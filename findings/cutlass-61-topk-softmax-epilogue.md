# CUTLASS Example 61: Hopper GEMM + Top-K + Softmax fusion

Reference for fused GEMM + selection kernels. Source files analyzed (CUTLASS `main`, 2024-2026 copyright):

- `examples/61_hopper_gemm_with_topk_and_softmax/61_hopper_gemm_with_topk_and_softmax.cu` (540 lines)
- `include/cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp` (~600 lines, the actual EVT node)
- `include/cutlass/epilogue/fusion/operations.hpp` (the `LinCombTopKSoftmaxCol` metadata tag)

No `README.md` exists in the example directory (only the `.cu` file and `CMakeLists.txt`).

---

## 1. What the kernel computes

A standard GEMM `D = alpha * (A @ B) + beta * C`, with `alpha = 1/k`, `beta = 0`, followed by a fused
**top-K + masked softmax over the N (column) dimension**:

- For each row `m`, find the top-K largest values across N.
- Keep only those K values; set everything else to 0.
- Apply softmax over the surviving K values: `softmax(x_i) = exp(x_i - logsumexp)` where
  `logsumexp = m + log(sum_j exp(x_j - m))`.

Shapes (default test config, `Options`):
- A: `[M, K]` row-major, fp16. Default M=16.
- B: `[N, K]` column-major (= `[K, N]` storage), fp16. Default N=8.
- C: `void` (beta=0, no source matrix).
- D: `[M, N]` row-major, fp16. Holds the softmax-normalized top-K values (non-top-K entries are 0).
- Problem shape is `Shape<int,int,int,int>` = `{M, N, K, L}` (batched).
- `TopK` is a compile-time constant, default 2.

So the fusion is: GEMM mainloop -> linear combination -> per-row top-K selection -> masked softmax, all in one kernel launch.

---

## 2. Epilogue visitor structure

### 2a. Two layers: a metadata tag + a real EVT node

There are **two separate things named "LinCombTopKSoftmaxCol"** in different roles:

1. **`cutlass::epilogue::fusion::LinCombTopKSoftmaxCol`** (in `operations.hpp`) is a **metadata tag struct**.
   It inherits from `LinearCombination` and adds **zero** fields. It only exists so the
   `CollectiveBuilder` can pattern-match on it at host compile time and wire up the real visitor tree:

   ```cpp
   // D = softmax(top_k(alpha * acc + beta * C))
   template<
     int TopK,
     class ElementOutput_,
     class ElementCompute_,
     class ElementSource_  = ElementOutput_,
     class ElementScalar_  = ElementCompute_,
     FloatRoundStyle RoundStyle_ = FloatRoundStyle::round_to_nearest
   >
   struct LinCombTopKSoftmaxCol
       : LinearCombination<ElementOutput_, ElementCompute_, ElementSource_, ElementScalar_, RoundStyle_> {
   };
   ```

2. **`Sm90TopKSoftmaxColReduction`** (in `sm90_visitor_topk_softmax.hpp`) is the actual device-side
   EVT reduction node. The `CollectiveBuilder` instantiates this when it sees the tag, and composes it
   into the full Epilogue Visitor Tree alongside the standard `LinComb` compute node and the store node.

So the EVT is a **tree**: `[LinComb compute] -> [Sm90TopKSoftmaxColReduction col-reduce] -> [store]`.
The user only ever types the tag; the builder assembles the tree.

### 2b. The reduction node declaration

```cpp
template <
  int TopK,
  int FragmentSize,
  class CtaTileShapeMNK,
  class EpilogueTile,
  class ElementOutput,
  class ElementCompute,
  FloatRoundStyle RoundStyle,
  int Alignment = 128 / sizeof_bits_v<ElementOutput>,
  bool UseButterflyReduce = true
>
struct Sm90TopKSoftmaxColReduction {
  static_assert(is_same_v<ElementCompute, float>, "Fused Top-K + Softmax reduction requires FP32 accumulation.");
  static_assert(TopK == 2 || TopK == 4,
    "Fused Top-K + Softmax reduction only allows K=2 and K=4, ...");
  static_assert(Alignment * sizeof_bits_v<ElementOutput> % 128 == 0, "sub-16B alignment not supported yet");
  ...
};
```

It satisfies the EVT `ColReduction` interface: `SharedStorage`, `Arguments`, `Params`,
`to_underlying_arguments`, `can_implement`, `get_workspace_size`, `initialize_workspace`,
`is_producer_load_needed`, `is_C_load_needed`, `get_producer_load_callbacks`,
`get_consumer_store_callbacks`. No workspace is used (`get_workspace_size` returns 0); the whole
reduction lives in registers + warp shuffles.

### 2c. Top-K implementation: hand-written PTX, NOT CUB

This is the most important architectural fact. **The top-K does not use `cub::BlockRadixSort`,
`cub::BlockMergeSort`, or any sorting primitive.** It is a custom descending-sorted-array merge with
fast paths for K=2 and K=4 written in **inline PTX**.

The four primitives (all `CUTLASS_DEVICE`, operate on `Array<float, K>` sorted descending):

- `top_2_reduce_scalar(a, b)` - insert one scalar into a sorted 2-array.
- `top_2_reduce(a, b)`        - merge two sorted 2-arrays into the top-2.
- `top_4_reduce_scalar(a, b)` - insert one scalar into a sorted 4-array.
- `top_4_reduce(a, b)`        - merge two sorted 4-arrays into the top-4 (the most complex PTX block).

Example, the full K=2 scalar-insert PTX:

```cpp
CUTLASS_DEVICE
Array<float, 2> top_2_reduce_scalar(Array<float, 2> a, float scalar) {
  Array<float, 2> out;
  asm volatile(
      "{\n"
      "  .reg .f32 mx;\n"
      "  .reg .pred p;\n"
      "  max.f32 mx, %3, %4;\n"
      "  setp.gtu.f32 p, %2, %4;\n"
      "  selp.f32 %1, mx, %2, p;\n"
      "  selp.f32 %0, %2, %4, p;\n"
      "}\n" : "=f"(out[0]), "=f"(out[1]) : "f"(a[0]), "f"(a[1]), "f"(scalar));
  return out;
}
```

The generic fallback (for K not in {2,4}) is a branchy shift-down insertion sort step:

```cpp
template <typename Element, int N>
CUTLASS_DEVICE
void add_element_to_desc_sorted_array(cutlass::Array<Element, N>& a, Element b) {
  if constexpr (N == 2 && is_same_v<Element, float>)      { a = top_2_reduce_scalar(a, b); }
  else if constexpr (N == 4 && is_same_v<Element, float>) { a = top_4_reduce_scalar(a, b); }
  else {
    CUTLASS_PRAGMA_UNROLL
    for (int k = 0; k < N; ++k) {
      if (a[k] < b) {
        CUTLASS_PRAGMA_UNROLL
        for (int l = N - 1; l > k; --l) { a[l] = a[l-1]; }
        a[k] = b;
        break;
      }
    }
  }
}
```

`merge_desc_sorted_arrays` dispatches the same way to `top_2_reduce` / `top_4_reduce`.

### 2d. Index tracking: THERE IS NONE

This is critical and the example calls it out explicitly. The top-K stores **values only**, not
indices. From the verify() comment in the `.cu`:

> "This formulation of top-K + softmax only works when it is guaranteed that none of the top-K
> elements are repeated! If this is the case, the device kernel can also make mistakes, because
> A. Once the top-K values are reduced, and the operation is being applied, there is no way to tell
> repeated elements apart, so none are masked.
> B. The softmax sum of exps will be incorrect (because the repeated elements are not repeated in it.)"

The reduction struct confirms it: `TopKResult { Array<ElementCompute, TopK> top_k_; }` - values only,
no index array. This is a hard limitation for any downstream use that needs "which column" the top-K
came from. **For a sampling kernel that needs to return token ids, this EVT node is not directly
reusable**; you would need to extend it to track `(value, index)` pairs through the PTX merge.

---

## 3. The softmax part: online logsumexp, not per-tile

Softmax is computed **only over the final top-K set**, not over all N elements. The trick that makes
this correct is:

- The top-K values are reduced across the whole row first (via the visit/warp-shuffle path below).
- Then `logsumexp` is computed from just those K values:

```cpp
// m + log(1 + sum_{i != m}(x_i - x_m))   -- one fewer exp than naive
template <typename Element, int N>
CUTLASS_DEVICE
Element topk_logsumexp(cutlass::Array<Element, N> a) {
  Element sum = Element(1.0);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 1; i < N; ++i) { sum += fast_exp(a[i] - a[0]); }
  return a[0] + fast_log(sum);
}
```

- The final per-row reduction result is **always 2 scalars regardless of K**: `{min of top-K (used as
  the mask threshold), logsumexp}`. Packed as `ReductionResult { ElementCompute min_; ElementCompute
  logsumexp_; }` (8 bytes, so it can be shuffled as a single `uint64_t`).
- Masking + softmax is fused into one PTX helper `fast_masked_softmax(value, minimum, logsumexp)`:

```cpp
CUTLASS_DEVICE
float fast_masked_softmax(float value, float minimum, float logsumexp) {
  float new_value;
  asm volatile(
      "{\n"
      "  .reg .pred p0;\n"
      "  setp.geu.f32 p0, %1, %2;\n"          // value >= minimum (keep if in top-K)
      "  ... expf lowering via ex2.approx ...\n"
      "  selp.f32 %0, %%f10, 0f00000000, p0;\n" // mask ? softmax_val : 0
      "}\n" : "=f"(new_value) : "f"(value), "f"(minimum), "f"(logsumexp));
  return new_value;
}
```

Because all non-top-K elements become 0 and the softmax denominator only includes the K survivors,
this is **equivalent** to softmax over the full row when the top-K values are distinct.

This is NOT "online softmax over K-reduction tiles" in the flash-attention sense. The GEMM K dimension
is reduced normally by the MMA accumulator; the top-K/softmax happens once, after accumulation, in the
epilogue's N-reduction pass.

---

## 4. Key code patterns

### 4a. Per-fragment visit (accumulate into per-thread top-K)

Called once per epilogue sub-fragment. `tCrTopK` is a register tensor laid out so that N-modes have
0-stride, meaning all threads contributing to the same row alias the same `TopKResult`.

```cpp
template <typename ElementAccumulator, typename ElementInput>
CUTLASS_DEVICE auto
visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n,
      Array<ElementInput, FragmentSize> const& frg_input) {
  auto& [tCrTopK, tCrSoftmax, tCcCol, cCol,
          lane_layout_MN, lane_mn,
          residue_cCol, residue_tCcCol] = args_tuple;
  Tensor tCcCol_mn = tCcCol(_,_,_,epi_m,epi_n);

  using ConvertInput = NumericArrayConverter<ElementCompute, ElementInput, FragmentSize, RoundStyle>;
  Array frg_I = ConvertInput{}(frg_input);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < FragmentSize; ++i) {
    auto thread_crd = tCcCol_mn(epi_v * FragmentSize + i);
    if (elem_less(thread_crd, residue_tCcCol)) {
      TopKResult& tCrCol_vmn = tCrTopK(epi_v * FragmentSize + i);
      detail::add_element_to_desc_sorted_array(tCrCol_vmn.top_k_, frg_I[i]);
    }
  }
  return frg_input;   // NOTE: returns the input unchanged; softmax applied later in reduce()
}
```

### 4b. The reduction (butterfly over warp lanes, then re-visit)

Triggered from `reduce(...)` with `is_last_iteration`. Two strategies gated by `UseButterflyReduce`:

```cpp
if constexpr (UseButterflyReduce) {
  // 1. Butterfly reduction: log2(warp_width) shuffle-xor rounds, all lanes end up with the merged top-K.
  CUTLASS_PRAGMA_UNROLL
  for (int j = 1; j < size<1>(lane_layout_MN); j *= 2) {
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size(tCrTopK_f); ++i) {
      tCrTopK_f(i).shuffle_xor_sync(j);   // merges via top_2_reduce / top_4_reduce
    }
  }
  // 2. Each lane computes its own logsumexp from the fully-reduced top-K.
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < size(tCrSoftmax_f); ++i) {
    tCrSoftmax_f(i) = tCrTopK_f(i).reduce_final();   // ReductionResult{min, logsumexp}
  }
}
// ... else: warp-shuffle-down + broadcast variant ...

// 4. Re-visit cached visit_results and apply masked softmax with the reduced scalars.
CUTLASS_PRAGMA_UNROLL
for (int epi_v = 0; epi_v < size(visit_results); ++epi_v) {
  auto& visit_frag = visit_results(epi_v);
  CUTLASS_PRAGMA_UNROLL
  for (int i = 0; i < FragmentSize; ++i) {
    visit_frag[i] = detail::masked_softmax(
      visit_frag[i],
      tCrSoftmax(epi_v * FragmentSize + i).min_,
      tCrSoftmax(epi_v * FragmentSize + i).logsumexp_);
  }
}
```

The shuffle methods pack the top-K array into `uint64_t` words to issue fewer shuffles
(`sizeof(TopKResult) == sizeof(uint64_t)` for K=2, `2 * sizeof(uint64_t)` for K=4):

```cpp
CUTLASS_DEVICE
void shuffle_xor_sync(int laneMask) {
  if constexpr (TopK == 2) {
    uint64_t top_k = reinterpret_cast<uint64_t&>(*this);
    top_k = __shfl_xor_sync(0xFFFFFFFF, top_k, laneMask);
    auto synced_v = reinterpret_cast<TopKResult&>(top_k);
    detail::merge_desc_sorted_arrays(top_k_, synced_v.top_k_);
  }
  else if constexpr (TopK == 4) {
    uint64_t* p = reinterpret_cast<uint64_t*>(this);
    uint64_t a = __shfl_xor_sync(0xFFFFFFFF, p[0], laneMask);
    uint64_t b = __shfl_xor_sync(0xFFFFFFFF, p[1], laneMask);
    uint64_t arr[2] = {a, b};
    detail::merge_desc_sorted_arrays(top_k_, reinterpret_cast<TopKResult&>(arr).top_k_);
  }
  ...
}
```

Note the static_assert inside `get_consumer_store_callbacks`:

```cpp
static_assert(decltype(size<1>(warp_layout_MN))::value <= 1);
// "Make sure there's only one warp across N so we can use warp shuffle intrinsics for reduction."
```

So the reduction is **intra-warp across N only**. Multiple warps across N are forbidden; cross-CTA
reduction is explicitly disallowed in `can_implement` because "there is no guarantee that all CTAs run
concurrently."

### 4c. `end_loop` resets per-tile state, `end` is a no-op

```cpp
CUTLASS_DEVICE void end_loop(int epi_m, int epi_n) {
  // Reset reduced top-K values for next tile.
  // Must be done because we only assume a single epilogue tile across N, but not M.
  fill(tCrTopK, TopKResult());
}
CUTLASS_DEVICE void end() { }
```

### 4d. Host setup is empty

```cpp
struct Arguments { };
struct Params { };

template <class ProblemShape>
static constexpr Params
to_underlying_arguments(ProblemShape const&, Arguments const&, void*) { return {}; }

template <class ProblemShape>
static size_t get_workspace_size(ProblemShape const&, Arguments const&) { return 0; }
```

All state is register-only; no workspace, no host preparation.

### 4e. `can_implement` - the shape constraints, enforced

```cpp
template <class ProblemShape>
static bool can_implement(ProblemShape const& problem_shape, Arguments const&) {
  auto [M, N, K, L] = problem_shape;
  auto [tile_M, tile_N, tile_K] = CtaTileShapeMNK{};
  auto [epi_M, epi_N] = EpilogueTile{};
  return N <= tile_N && N <= epi_N && N >= TopK;
}
```

---

## 5. Constraints / limitations

Stated in the `.cu` docstring and enforced in code:

1. **Fusion is over the N dimension only.** Reduces columns, one row at a time.
2. **Top-K is a compile-time constant.** Different K needs a different kernel.
3. **Only K=2 and K=4 are performance-optimized** (static_assert; other K falls through to a generic
   branchy sort that "can lead to serious performance implications" and register spill). The docstring
   says you can delete the assert to enable the generic path.
4. **`CTA_N >= N` and `EPI_N >= N`** - the entire reduction dimension must fit in a single CTA tile
   AND a single epilogue tile. This is why the example uses `TileShape = <64,64,128>` and defaults
   `N=8`. In practice this caps the reducible N at ~128-256.
5. **Only one warp across N** (`static_assert(size<1>(warp_layout_MN) <= 1)`), because reduction uses
   warp shuffle intrinsics. Multiple warps across M are fine.
6. **No cross-CTA reduction.** "There is no guarantee that all CTAs run concurrently."
7. **`ElementCompute` must be `float`** (static_assert).
8. **128-bit alignment** required on the output (`Alignment * sizeof_bits % 128 == 0`).
9. **No index tracking** - values only (see 2d). Distinct top-K values are assumed.
10. **`EpilogueTileType = take<0,2>(TileShape{})`** - the epilogue tile must equal the mainloop M,N
    tile. No epilogue sub-tiling allowed, because the reduction revisits accumulated fragments from a
    single epilogue tile.
11. Register pressure: the K=4 PTX is already large; the docstring warns the generic path causes
    register spill. The `ReductionResult` and `TopKResult` are deliberately kept at 8/16 bytes so they
    pack into 1-2 `uint64_t` shuffle words.

---

## 6. How the EVT is wired into the GEMM

All in the `.cu`, file-scope type aliases (no runtime hook):

```cpp
using TileShape        = Shape<_64,_64,_128>;
using ClusterShape     = Shape<_1,_1,_1>;
using KernelSchedule   = cutlass::gemm::KernelTmaWarpSpecialized;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized;

// The fusion op tag. LinCombTopKSoftmaxCol<K,...> inherits from LinearCombination.
using FusionOperation = std::conditional_t<EnableTopKSoftmax,
  typename cutlass::epilogue::fusion::LinCombTopKSoftmaxCol<TopK, ElementD, ElementCompute>,
  typename cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute>
>;

// KEY: the epilogue tile must be the mainloop's M,N tile, no sub-tiling.
using EpilogueTileType = decltype(cute::take<0,2>(TileShape{}));

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    TileShape, ClusterShape,
    EpilogueTileType,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    EpilogueSchedule,
    FusionOperation               // <-- tag drives the EVT construction
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))
    >,
    KernelSchedule
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue
>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
```

Runtime arguments - note the epilogue argument block is just `{alpha,beta}, C-ptr, stride, D-ptr,
stride`. The top-K/softmax node takes **no extra arguments** (its `Arguments` is empty):

```cpp
typename Gemm::Arguments arguments{
  cutlass::gemm::GemmUniversalMode::kGemm,
  {options.m, options.n, options.k, options.l},
  {tensor_A.device_data(), stride_A, tensor_B.device_data(), stride_B},
  {
    {options.alpha(), 0.f},        // alpha, beta
    nullptr, stride_D,             // C (nullptr since ElementC=void)
    tensor_D.device_data(), stride_D
  }
};
```

Host reference (for verification) does top-K + softmax manually, row by row, then compares with
`TensorRelativeErrorMetric` at `eps=1e-5`.

---

## 7. Blackwell (sm_100) portability

**No out-of-the-box Blackwell support.** Evidence:

- The fusion directory (`include/cutlass/epilogue/fusion/`) contains `sm90_visitor_topk_softmax.hpp`
  but has **no** `sm100_visitor_topk_softmax.hpp` or `sm120_visitor_topk_softmax.hpp` counterpart.
- The example `.cu` hard-guards on `props.major != 9 || props.minor != 0` and exits otherwise.
- The reduction node class is named `Sm90TopKSoftmaxColReduction` and includes
  `sm90_visitor_tma_warpspecialized.hpp` (the SM90 EVT base).
- The `LinCombTopKSoftmaxCol` tag in `operations.hpp` carries no arch field, so in principle the
  `CollectiveBuilder` for sm100/sm120 *could* dispatch it - but there is no sm100/sm120 reduction
  visitor implementation to dispatch to, so it would fail to build or fall through to an unsupported
  path.

To port to Blackwell you would need to write a new `Sm100TopKSoftmaxColReduction` (or similar) that
inherits from the sm100 EVT base (`sm100_visitor_store_tma_warpspecialized.hpp`,
`sm100_visitor_compute_tma_warpspecialized.hpp`) and replicates the same
visit -> warp-shuffle-reduce -> revisit-with-softmax pattern. The PTX merge primitives
(`top_2_reduce`, `top_4_reduce`, `fast_masked_softmax`) are architecture-neutral `.f32` PTX and should
lift directly; the layout math in `get_consumer_store_callbacks` is also arch-neutral. The main work
would be adapting to the sm100 epilogue callback signatures and TMA store path.

The user's note in `findings/tma-store-blackwell-singleton-dims.md` (TMA store silently no-ops with
singleton dims on B200) is relevant: this EVT re-visits register fragments rather than re-loading from
shared memory, so it sidesteps that class of bug, but any Blackwell port that stores per-tile top-K
intermediates through TMA would need to re-check store correctness.

---

## Relevance to FlashSampling (FMMS)

What carries over to a fused GEMM + sampling kernel that needs tile-local
top-K:

- **The PTX merge primitives are directly reusable** for `TopK=20`-style selection IF you compile the
  generic path or write a K=20 fast path. The generic `add_element_to_desc_sorted_array` /
  `merge_desc_sorted_arrays` work for any N but are branchy.
- **The butterfly-shuffle reduction over warp lanes** is a useful pattern
  for intra-CTA top-k merging, but its lane layout reduces N.
- **The revisit-and-mask structure is not needed for Gumbel-Max.**
  FMMS adds Gumbel noise before the max and writes only candidates.
  It does not revisit the accumulator tile to produce a masked dense output.
- **FMMS needs index tracking.**
  CUTLASS's node carries only values.
  FMMS needs `(value, global_row_index)` because vocabulary is GEMM M in the
  performant orientation.
- **The reduction axis is the decisive mismatch.**
  Example 61 reduces N, while FMMS computes `W[V,D] @ H[D,H]` and must
  reduce vocabulary M.
  Its visitor cannot be reused as the FMMS reduction skeleton.
  Swapping the output so vocabulary becomes N makes `CTA_N >= N`
  impossible for V=128K-152K and repeats the measured operand-swap
  regression.
- **Use `Sm90RowReduction` as the structural precedent.**
  It reduces M and already includes fragment, lane, warp, CTA, and optional
  cross-CTA reduction paths.
  Example 61 contributes top-k merge helpers only.
- **Softmax-over-top-K only** (logsumexp from K survivors) matches FMMS's per-tile Gumbel-max trick
  conceptually, but FMMS samples rather than normalizes.

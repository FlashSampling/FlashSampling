# CUTLASS FMMS Kernel Plan

A production-grade CUDA/CUTLASS implementation of FlashSampling that closes
the Triton-vs-cuBLAS GEMM gap, supports top-k, and extends naturally to the
tensor-parallel path.

This plan is informed by deep research into the current state of CUTLASS 4.x,
CCCL/CUB, Blackwell sm_100, and the closest reference implementations
(CCE, Liger, FlashInfer, Quack). Research notes live in
`findings/cutlass/02-topk-softmax-epilogue.md` and the four research reports
that produced it.

## 2026-07-31 revision: reopen GEMM kernel discovery

The deterministic reduction and production integration work through Gate 2a
is still valid.
The Gate 18 performance stop applies only to the small manually selected
family that was measured: six tile/schedule controls plus explicit stage
counts, all with a `1x1x1` cluster and no explicit Blackwell 2-SM coverage
audit.
It does not establish that CUTLASS cannot match `torch.mm`.

The earlier plan treated stage-count improvement as a prerequisite for cluster
tuning and rejected several scheduler families from qualitative reasoning.
That was too restrictive.
NVIDIA's supported workflow treats CTA and instruction shapes, persistent
schedules, 1-SM versus 2-SM MMA, clusters, split-K, raster order, and swizzle
as a joint kernel-selection problem.
The revised plan therefore reopens Gate 2 with an official
`nvidia-matmul-heuristics` plus `cutlass_profiler` search before any more FMMS
features are implemented.

This revision makes two distinctions explicit:

1. `torch.mm` is the sole strong matmul baseline.
   It exercises the production PyTorch path and dispatches to NVIDIA's cuBLAS
   backend, which is the comparison that matters for this project.
2. A plain CUTLASS winner is only a schedule donor.
   The custom candidate epilogue must still revalidate correctness and measure
   its own performance after the schedule is transplanted.

## 2026-07-31 priority: complete Blackwell before Hopper

The active implementation target is B200 (`sm_100a`) only.
The goal is a complete B200 CUTLASS provider, not a cross-architecture partial
implementation.

The B200 milestone includes:

- Plain-GEMM kernel discovery and production dispatch.
- Greedy TP1 FMMS.
- Stateless Philox and unrestricted Gumbel-Max sampling.
- TP2, TP4, and TP8 correctness and performance.
- Fixed-K top-k and the supported top-p-on-survivors sampling path.
- Memory-traffic validation and end-to-end vLLM benchmarks.
- Every small and large benchmark dimension and every declared hidden-state,
  tensor-parallel, and sampling configuration supported by Triton
  FlashSampling on B200.

Hopper implementation work is deferred until the B200 completion gate passes.
Existing H100 correctness and toolchain evidence remains valid historical
coverage, but it does not impose work on the B200 critical path.

The B200 completion gate is intentionally strict.
The CUTLASS implementation must beat the corresponding Triton FlashSampling
provider pointwise in every declared B200 performance cell.
Do not use an average, geometric mean, pointwise minimum across runs, or a win
in one regime to offset a loss in another.
Correctness, supported features, output contracts, and memory safety must also
match before the Hopper port begins.

After B200 passes, Gate 8 ports the complete design to H100/H200.
The Hopper phase repeats kernel discovery and accumulator-layout derivation for
sm_90 rather than forcing Blackwell schedule or ownership assumptions onto
Hopper.

## TL;DR

Build a CUTLASS C++ GEMM with a **custom Epilogue Visitor Tree (EVT)** that
performs per-tile Gumbel-noise argmax (or top-k) on output tiles while they
are still in registers, then writes a tiny `[num_samples, num_tiles, H, k]`
candidates tensor to HBM. The kernel loops over `num_samples` internally up
to a compile-time bound; larger sample counts are batched across launches.
A small second stage (the existing Triton/PyTorch merge) picks the global
winner and returns samples with the same `[H, num_samples]` shape as every
other provider.

Use a custom **M-axis reduction visitor** derived from CUTLASS
`Sm90RowReduction`.
The GEMM is `W[V,D] @ H[D,H]`, so vocabulary is M and hidden states are N.
CUTLASS example 61 reduces N and therefore cannot be copied as the reduction
skeleton for this layout.
It remains useful for its register-resident top-k merge primitives.

After the single-CTA kernel is measured, optionally test a **DSMEM cluster
reduction** across adjacent M tiles.
This can reduce the number of candidates written to HBM and exchanged in TP,
but it does not reduce the per-CTA accumulator shape or inherit Quack's
register-spill recovery mechanism.

This design:

- Targets the cuBLAS backend gap (the AC's sharpest implementation concern).
- Adds top-k as a real feature, not a future-work bullet (99UK, AC ask #1).
- Preserves the existing TP symmetric-memory machinery while requiring the
  CUTLASS overlap benefit to be measured independently.
- Keeps the performant GEMM orientation, with V on M and H on N.
- Uses the shipped M-reduction visitor machinery rather than applying an
  N-reduction visitor to the wrong axis.
- Treats DSMEM candidate compression as an empirical optimization, with no
  assumed register-pressure benefit.

The implementation is organized as stage gates.
Each gate produces the smallest artifact that can disprove the current
approach, has a numerical success threshold, and preserves a fallback.
The critical path is the completed cross-architecture toolchain and reduction
foundation, followed by B200-only plain-GEMM kernel discovery, greedy FMMS,
stateless Philox, Gumbel-Max, TP, top-k, and end-to-end validation.
Top-k follows the working sampling kernel.
DSMEM is outside the critical path and is attempted only if profiling shows
candidate traffic is material.

## Why CUTLASS C++ (not CuTe-DSL, not raw CUDA)

A careful comparison of four implementation languages (see research reports)
lands on CUTLASS C++ for one decisive reason: **RNG**.

The FMMS epilogue must generate Gumbel noise `g = -log(-log(u))` per element.
The options:

| Language | RNG story | Verdict |
|---|---|---|
| **CUTLASS C++** | Native `curand_*` device API. Works in any device function including EVT visitors. | **Chosen.** |
| CuTe-DSL | **No RNG primitive exists.** Three fallbacks: (1) pre-generated noise tensor (kills fusion, adds HBM traffic that defeats the kernel's purpose), (2) hand-rolled Philox (feasible but requires re-validating distribution statistics), (3) link curand bitcode via `cute.ffi.extern(BitCode(...))` (fragile). | Rejected. |
| Triton (status quo) | `tl.rand`. Kernel already written and tuned. | Stays as the reference/portability kernel. |
| Raw CUDA | `curand_*`. Maximum control. | Loses Python iteration speed and EVT abstraction. |

CuTe-DSL is the wrong move *for this kernel* because the RNG gap forces a
fundamental algorithmic change. It would be the right move for a different
kernel family (pure reductions, dense GEMM with composable epilogue). The
absence of `curand` in `cute.arch` is documented upstream as of `main`.

CUTLASS C++ also gives us:

- **EVT (Epilogue Visitor Tree)**, the right abstraction for composing
  per-tile scale/noise/argmax/store.
  The reduction skeleton is `Sm90RowReduction`, which reduces M.
  Example 61's `Sm90TopKSoftmaxColReduction` reduces N and supplies only
  reusable top-k helper ideas.
- **CollectiveBuilder GEMMs** with TMA, warp specialization, tensor cores,
  Hopper WGMMA, and Blackwell tcgen05.mma.
  Whether the custom kernel reaches cuBLAS parity is an empirical target.
- **Mature tooling**: cuda-gdb, Nsight Compute, stable API.
- **Upstream vLLM merge path**: vLLM already ships CUTLASS-based kernels
  (FP8 GEMMs). A CUTLASS FMMS kernel is mergeable in a way a one-off Triton
  kernel is not.

## Background: choosing the correct CUTLASS reduction axis

### Why example 61 cannot be the reduction skeleton

Example 61 uses `Sm90TopKSoftmaxColReduction`
(`include/cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp`). Its
`can_implement` enforces:

```cpp
return N <= tile_N && N <= epi_N && N >= TopK;
```

i.e. the **entire N reduction axis must fit inside one CTA tile**.
The example source explicitly says that fusion is over N.

FMMS computes `W[V,D] @ hidden_states.T[D,H]`, so its output has shape
`[V,H]`.
Vocabulary is M and hidden states are N.
Applying the example-61 visitor to this output would reduce hidden states,
not vocabulary.
Swapping the operands/output to place V on N would make the example's
`N <= tile_N` constraint impossible for V=128K-152K and would repeat the
operand-swap regression documented in
`findings/cutlass/00-2cta-mma-operand-swap-regression.md`.

### The correct starting point: `Sm90RowReduction`

CUTLASS ships `Sm90RowReduction` in
`sm90_visitor_store_tma_warpspecialized.hpp`.
In CUTLASS terminology, this reduces the M rows and emits a result for each
N column, which is exactly FMMS's vocabulary reduction.
It supports:

- Register reduction within each lane's fragments.
- Warp-shuffle reduction across M lanes.
- Shared-memory reduction when multiple warps cover M.
- Either a non-final per-CTA output or a workspace plus tile counters for a
  cross-CTA final reduction.

The FMMS visitor should derive its layout and reduction choreography from
`Sm90RowReduction`, replacing the scalar reduction value with a
`(noisy_logit, global_vocab_index)` pair and a max-with-index operation.
Initially use `FinalReduction=false` so every M tile writes one candidate per
N column.
The existing GPU Stage 2 then merges candidates across M tiles.

The stock node is templated on scalar reduce functors
(`RegReduceFn`/`ShuffleReduceFn`/`GmemReduceFn` over `ElementCompute`), so
carrying a `(value, index)` pair means forking its reduce internals rather
than instantiating it with a custom functor. Example 61's packed
`TopKResult` shuffles are the precedent for the pair exchange. The node's
coordinate tensors (`tCcRow`, `residue_tCcRow` in its args tuple) provide
the per-element global M coordinate and out-of-bounds predication that both
the global vocabulary index and the Philox counter mapping derive from.

Example 61 remains useful for its sorted-array insertion and K=2/K=4 PTX
merge helpers, but not for its N-axis visitor layout or `can_implement`
condition.

## Building blocks

### CUTLASS 3.x/4.x EVT

Canonical references (all in `NVIDIA/cutlass` on `main`):

- `include/cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp`:
  the base classes: `Sm90VisitorImpl`, `Sm90TreeVisitor`, `Sm90EVT`,
  callback shells. ~900 lines.
- `include/cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp` - the
  source of reusable top-k merge helpers, not the FMMS visitor layout.
- `include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp`
  - `Sm90AccFetch`, `Sm90ScalarBroadcast`, `Sm90AuxLoad`.
- `include/cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp`
  - `Sm90AuxStore`, `Sm90ScalarReduction`, `Sm90RowReduction`.
- `include/cutlass/epilogue/fusion/sm100_callbacks_tma_warpspecialized.hpp`
  - Blackwell callback definitions.
  It reuses some SM90 compute nodes, but it does not provide
  `Sm100TopKSoftmaxColReduction` or establish that a custom SM90 reduction
  visitor carries over unchanged.
- `examples/71_blackwell_gemm_with_collective_builder/` - the canonical
  EVT construction example for Blackwell.

The visitor lifecycle (from `sm90_visitor_tma_warpspecialized.hpp`):

| Callback | Granularity | When |
|---|---|---|
| `begin()` | once per CTA | top of store loop |
| `begin_loop(epi_m, epi_n)` | per subtile | iteration start |
| `previsit(...)` | per subtile | after producer loads visible |
| `visit(frg_acc, epi_v, epi_m, epi_n[, frg_inputs...])` | per fragment | many per subtile |
| `reduce(smem, sync_fn, epi_m, epi_n, is_last, visit_results)` | per subtile | after all `visit` calls |
| `postreduce(...)` | per subtile | after smem fence |
| `tma_store(...)` | per subtile | before TMA commit |
| `end_loop(epi_m, epi_n)` | per subtile | end of iteration |
| `end()` | once per CTA | end of all subtiles |

The **revisit-and-apply** mechanism in `reduce()` is what lets us compute a
per-tile argmax then write only the winner: `visit()` accumulates into a
register-array state, `reduce()` does the warp/block shuffle merge, and
either `postreduce()` writes the result or `end()` does.

### CUB / CCCL primitives

`cub::detail::block_topk` is a possible later optimization, not the initial
implementation.
It is a radix-select "AIR TopK" algorithm:

- File: `cub/cub/block/block_topk.cuh`, thin wrapper around
  `block_topk_air.cuh`.
- Added in **CCCL 3.3.0** (CUDA 13.3). Public entry point is
  `cub::DeviceTopK` / `cub::DeviceBatchedTopK`. The block-level primitive
  lives in `detail::` and is undocumented but functional.
- Algorithm: MSB-to-LSB radix passes that narrow the candidate set; only
  the final pass moves data. Roughly half the work of full `BlockRadixSort`
  for f32 (4 passes vs 8).
- API:

  ```cpp
  template <typename KeyT, int BlockDimX, int ItemsPerThread, typename ValueT = NullType>
  class block_topk {
    struct TempStorage { ... };
    block_topk(TempStorage& storage);
    template <bool IsFullTile>
    void max_pairs(KeyT (&keys)[IPT], ValueT (&values)[IPT],
                   int k, int num_valid, int begin_bit=0, int end_bit=sizeof(KeyT)*8);
  };
  ```

- **Risk**: it lives in `detail::`, is undocumented, and assumes a
  block-level collective.
  Do not put it on the critical path.
  Start with a fixed-K warp-group merge derived from example 61.

For **argmax only** (k=1), skip CUB and use the inline-PTX warp-shuffle
pattern from `sm90_visitor_topk_softmax.hpp` (the `top_2_reduce_scalar`
helpers), extended to carry indices. This matches what FlashInfer's
`SamplingFromLogitsKernel` does with `DataAndIndex<DType, IdType>` and
`cub::BlockReduce<DataAndIndex, ...>::Sum` (where `+` is overloaded to
max-with-index-carry).

For **cumulative sum** (top-p on the surviving k candidates, in stage 2):
`cub::BlockScan<float, N>::InclusiveSum`. Stage-2 only, not in the EVT
visitor.

### Reference implementations

- **CCE** (`apple/ml-cce`, Triton): the architectural template. Its forward
  kernel does tiled matmul + per-tile LSE + atomic-locked per-row merge
  across V tiles. To convert to Gumbel-Max: swap `logaddexp` for
  `max-with-argmax` (carry `(val, idx)`, compare-and-swap on val).
  Group swizzling (GROUP_B=8) is already optimal for the hidden-state reuse
  / weight streaming pattern at low batch.

- **FlashInfer `SamplingFromLogitsKernel`** (`flashinfer/include/sampling.cuh`):
  the **direct template for the per-tile argmax epilogue**. CUDA C++ using
  `cub::BlockReduce<DataAndIndex<DType, IdType>>::Sum` where `+` is
  max-with-index-carry. Gumbel noise via `curand_uniform4` +
  `-kLOG2 * log2f(-log2f(x * kSCALE))` (one PTX `LG2` instruction instead
  of `logf`).

  ```cpp
  // The entire Gumbel-Max sampler, one block per row (FlashInfer)
  DataAndIndex<DType, IdType> max_data = {-inf, 0};
  for (chunks over V) {
    logits_vec.cast_load(...);
    gumbel_noise = GenerateGumbelNoise<DType, VEC_SIZE>(seed, offset, subsequence);
    DataAndIndex<DType, IdType> cur[VEC_SIZE];
    for (j in VEC_SIZE) {
      cur[j].data = logits[j] + gumbel[j];
      cur[j].index = token_idx;
    }
    max_data += BlockReduce<DataAndIndex<...>>(temp).Sum<VEC_SIZE>(cur);
    __syncthreads();
  }
  if (tx == 0) output[bx] = max_data.index;
  ```

  Our per-tile EVT visitor is one block of this loop, with the surrounding
  loop replaced by the GEMM's tile iteration.

- **Quack** (`Dao-AILab/quack`, CuTe-DSL): reference for memory-bound
  speed-of-light patterns. Their `reduce.py` reduction template
  (thread -> warp shuffle -> SMEM block -> cluster DSMEM) is the right
  mental model, but we will use CUB primitives rather than vendoring Quack.

- **CUTLASS `Sm90RowReduction`**: reference for the EVT visitor layout and
  M-axis reduction choreography.
  It already implements register, warp, block, and optional cross-CTA
  reduction over M.
  FMMS must extend its scalar state to carry `(value, index)`.

- **CUTLASS example 61**: reference only for register-resident top-k data
  structures and the PTX merge primitives (`top_2_reduce`, `top_4_reduce`).
  Its visitor reduces N, the wrong axis for the chosen FMMS GEMM layout.
  Its two reusable gaps are index tracking and efficient K>4 selection.

### Hardware targets

| Arch | Chips | CUTLASS support | FMMS path |
|---|---|---|---|
| sm_90 | H100, H200 | CUTLASS 3.x WGMMA + TMA + clusters | Deferred Gate 8 target after the complete B200 gate passes. Existing correctness evidence is retained. |
| sm_100 | B100, B200, B300 | CUTLASS 4.x tcgen05.mma + TMEM + clusters (max 8) | Active target. Develop and approve on B200 first. |
| sm_103 | B300 Ultra | Ultra FP4 paths | Not relevant for BF16 FMMS. |
| sm_120 | RTX 5080 (GeForce) | No TMA multicast, no clusters | Use `ClusterShape<1,1,1>`; same kernel runs. |
| sm_80 (A100) | A100 | CUTLASS 3.x SIMT path | Out of scope for the CUTLASS port. The Triton kernel remains the A100 fallback. |

**Clarification: cluster MMA vs DSMEM candidate reduction.** These are
distinct mechanisms:

- **GEMM scheduling clusters and Blackwell 2-SM MMA**: required candidates in
  the reopened plain-GEMM search.
  A cluster along M can multicast the hidden-state B tile reused by adjacent
  vocabulary tiles, even though it does not reduce the streaming weight
  traffic.
  Blackwell `tcgen05.mma cta_group::2` must be tested through CUTLASS's native
  2-SM dispatch and legal tile shapes.
  The rejected experiment in
  `findings/cutlass/00-2cta-mma-operand-swap-regression.md` swapped GEMM
  operands to keep a Triton reduction inside one CTA.
  Its 6-23% low-H regression does not test CUTLASS's native 2-SM GEMM
  orientation and cannot exclude that family.
- **DSMEM candidate reduction in the epilogue**: optional experiment.
  A cluster along M can merge adjacent vocabulary-tile candidates before
  HBM.
  This reduces candidate count, but the CUTLASS CTA already has a fixed
  `tile_N`; clustering does not change its accumulator shape or solve the
  current persistent Triton kernel's H=256 spill.
  See `findings/cutlass/03-dsmem-cluster-reduction.md`.

**Required B200 plain-GEMM search dimensions**:

- Blackwell 1-SM and 2-SM dense BF16 dispatch policies with their legal CTA,
  instruction, epilogue, and cluster shapes.
- Static clusters and preferred/fallback flexible clusters on sm_100a.
- Runtime raster order and tile swizzle.
- Split-K or Stream-K candidates when NVIDIA's heuristic emits them.
  The large M dimension may make them unnecessary, but that is a measured
  outcome rather than a prior exclusion.

Do not infer that a mechanism is irrelevant solely because the arithmetic
intensity model labels a case memory-bound.
Scheduling, multicast, and rasterization can still change reuse, memory-level
parallelism, and the fraction of peak bandwidth achieved.

Hopper non-persistent, persistent ping-pong, and persistent cooperative
schedules move to the Gate 8 search after B200 completion.

## The kernel design

### Single-rank, k=1 (Gumbel-Max)

```
┌───────────────────────────────────────────────────────────────────┐
│ CUTLASS GEMM (CollectiveBuilder, cuBLAS parity)                   │
│                                                                   │
│   W[V_tile, D] @ H[D, H_tile]   → accumulator in registers/TMEM   │
│   (TMA loads, warp-specialized producer/consumer, tensor cores)   │
│                                                                   │
│ Epilogue Visitor Tree (per output tile, in registers):            │
│                                                                   │
│   Sm90EVT<                                                        │
│     FmmsArgmaxVisitor,                                            │
│     Sm90Compute<divides, ...>,                                    │
│     Sm90ScalarBroadcast<temperature>,                             │
│     Sm90AccFetch                                                  │
│   >                                                               │
│                                                                   │
│   FmmsArgmaxVisitor::visit(frg_acc, epi_v, epi_m, epi_n,          │
│                            frg_scaled):                           │
│     1. stateless_philox(global_output_coordinate) -> u           │
│     2. gumbel = -LOG2E * log2f(-log2f(u * SCALE))                 │
│     3. noisy = frg_scaled + gumbel                                │
│     4. add_element_to_desc_sorted_array(                         │
│            thread_local_argmax,                                   │
│            {noisy, global_v_idx})                                 │
│                                                                   │
│   FmmsArgmaxVisitor::reduce(...):                                 │
│     1. reduce across M lanes using Sm90RowReduction's layout      │
│        and warp-shuffle choreography                              │
│     2. smem block reduction across the warps that own M lanes     │
│        while carrying (val, global_vocab_idx) pairs               │
│                                                                   │
│   FmmsArgmaxVisitor::end_loop(epi_m, epi_n):                      │
│     per sample: write (val, idx) to candidates                    │
│       [sample_idx, pid_v, h_idx, 0]                               │
│     (plain st.global, not TMA - the payload is 8 bytes)           │
└───────────────────────────────────────────────────────────────────┘
                                 ↓
        [num_samples, num_tiles_v, H, 1] candidates tensor (HBM)
                                 ↓
┌───────────────────────────────────────────────────────────────────┐
│ Stage 2 (unchanged from current Triton implementation):           │
│   max over the num_tiles dimension, gather the winning indices    │
│   Returns samples with shape [H, num_samples]                     │
└───────────────────────────────────────────────────────────────────┘
```

The visitor is structurally based on `Sm90RowReduction`, with three changes:
1. Stateless Philox parameters are added to `Arguments`.
   Each element derives its counter from global output coordinates.
2. The CTA-local reduction state carries FP32 value plus an i32 vocabulary
   index, packed into 64 bits for shuffle exchange.
   V is far below the signed i32 limit.
   Widen the index to the public int64 output type only when storing.
3. `FinalReduction=false` writes one candidate per M tile, N column, and
   sample.

For top-k, reuse example 61's sorted-array merge concepts inside this
M-reduction layout.

### Single-rank, k>1 (top-k)

Same structure. Two changes:

1. The first implementation uses a compile-time fixed K, initially K=20 or
   padded K=32, and a custom warp-group sorted-array or merge network derived
   from example 61.
   `cub::detail::block_topk` is evaluated only after the fixed-K path works.

2. `end_loop()` writes k pairs per sample to
   `candidates[sample_idx, pid_v, h_idx, 0..k-1]`.

Stage 2 changes from `max(dim=0)` to top-k merge across `num_tiles*k`
candidates + softmax + (optional) top-p + multinomial sample, exactly as
in the current `tl_fused_mm_topk.py::_topk_merge_and_sample`.

The example-61 `N <= tile_N` constraint does not apply to this custom
M-reduction visitor.
Each CTA selects top-k within its own `tile_M` vocabulary rows, and Stage 2
merges across M tiles.

### Tensor-parallel variant

The current TP path in `tensor_parallel_reduce.py` works as follows:
the kernel output buffers (`maxs`, `maxs_idx`) are allocated in symmetric
memory, so the kernel's existing TMA stores write directly to NVLink-mapped
peer addresses. After the kernel completes, a host-side barrier ensures all
ranks' writes are visible; each rank then reads all ranks' per-tile outputs
and reduces.

The CUTLASS port first writes candidates locally and reuses a separate P2P
fan-out kernel.
This is the correctness-preserving fallback.
Only after TP2 is stable does the EVT store visitor attempt direct writes to
peer symmetric-memory buffers.

The host-side `_local_reduce` and `_stack_and_select_winner` are unchanged
(they operate on a small `[world_size, num_samples, num_tiles, H, k]`
candidates tensor already in local HBM after the symm-mem barrier).
The internal candidates layout is an implementation detail; the only shape
contract is the final samples output `[H, num_samples]`, shared with every
other provider.

The existing Triton overlap result does not automatically transfer to the
CUTLASS backend.
Measure local output plus fan-out against direct peer stores over the full
kernel, communication, barrier, and reduction path.

**DSMEM clustering can reduce TP candidate traffic.** When each rank's
kernel uses DSMEM reduction across adjacent M tiles, the per-rank candidate
count drops from
`num_tiles` to `num_clusters` (e.g., 1000 -> 125 at cluster_m=8). The
inter-rank symm-mem payload shrinks 8x, and the host-side reduction
operates on `world_size * num_clusters` candidates instead of
`world_size * num_tiles`. The two layers are independent: DSMEM cluster
reduction is intra-rank; symm-mem P2P is inter-rank over NVLink.
Whether this saves measurable latency must be benchmarked because candidate
traffic is already small relative to weight reads.

**Explicitly rejected**: Hopper+ cluster DSMEM reduction (example 93
pattern) for *inter-rank* TP. Clusters are strictly intra-GPU (one CUDA
context, one GPC). Inter-rank communication must still go over NVLink, so
symmetric memory remains the right mechanism. Cluster DSMEM is also
limited to 8-16 CTAs, far below the 1000+ V tiles in a V=128k problem.

## DSMEM cluster reduction: optional candidate compression

Deep dive in `findings/cutlass/03-dsmem-cluster-reduction.md`. Summary of how
it changes the design.

### What the source review changed

The current Triton kernel spills at H=256 because each persistent program
holds a `[BLOCK_SIZE_V, BLOCK_SIZE_H]` tile with `BLOCK_SIZE_H=64`.
That does not imply that a CUTLASS cluster can reduce the CUTLASS kernel's
register footprint.

A conventional CUTLASS GEMM already tiles N.
For fixed `tile_N=64`, increasing global H from 64 to 256 launches more N
tiles; it does not make one CTA hold 256 columns.
A cluster along M can merge vocabulary candidates, but every member still
computes the same fixed `[tile_M,tile_N]` accumulator tile.
A cluster along N computes different hidden-state columns and cannot merge
them into the same sample.

Therefore the plan makes no register-spill or 1.5x bandwidth-recovery claim
for DSMEM.
The single-CTA CUTLASS kernel must be profiled directly.

### What clusters cannot do

- **Cannot reduce weight matrix HBM traffic.** The bottleneck is reading
  W[V,D] (~2 GB). Cluster MMA partitions the output but does not reduce
  per-element W reads. Confirmed empirically by
  `findings/cutlass/00-2cta-mma-operand-swap-regression.md`.
- **Cannot help with inter-rank TP.** Clusters are strictly intra-GPU
  (one GPC, one CUDA context, max 8 portable / 16 non-portable CTAs).
  Inter-rank reduction still needs NVLink + symm-mem.
- **Cannot do cluster-wide TMEM access.** TMEM is per-CTA (or per
  cta_group::2 pair for 2-CTA MMA). No general cluster-TMEM gather.

### What clusters can do

1. **Reduce Stage-1 candidate writes and Stage-2 input size** by merging
   adjacent M-tile candidates before HBM.
   This does not remove Stage 2 unless one cluster spans the full vocabulary,
   which is impossible at the portable cluster-size limit.
2. **TMA multicast hidden states across CTAs clustered along M.**
   These CTAs consume the same hidden-state N tile while reading different
   weight rows.
   The possible saving is small because weight traffic dominates.

### The clustered kernel design (additive to the baseline)

The clustered variant layers on top of the proven single-CTA EVT visitor.
It is evaluated only after TP and top-k are stable and profiling admits the
optional DSMEM gate.

```
Cluster shape: (cluster_m, 1, 1)
  Adjacent CTAs cover different vocabulary M tiles and the same N tile.

Per CTA (within cluster), additions to the EVT visitor:
  4. EVT visitor postreduce(): DSMEM cluster_reduce
     - Pack (val: f32, idx: i32) -> i64  (halves DSMEM traffic)
     - st.async.shared::cluster.mbarrier to peer CTAs (Quack pattern)
     - mbarrier_wait
     - Local reduce across cluster_m pairs -> cluster winner (val, idx)
  5. CTA rank 0 writes cluster winner to HBM
     (or to peer symm-mem in TP, replacing the per-tile write)

The DSMEM choreography (one mbarrier round-trip for the (val, idx) tuple):
  - elect_one thread arms mbarrier_arrive_and_expect_tx(combined_byte_count)
  - each lane < cluster_m does store_shared_remote to peer CTA's slot
  - mbarrier_wait blocks until all peers' bytes arrive
  - local reduce across cluster_m pairs (trivial)
```

### Cluster sizing

Cluster size is not selected from H to manage register pressure.
Treat it as a tuning parameter for candidate compression and hidden-state
multicast:

```python
def fmms_cluster_m(arch, tile_m, tile_n):
    if arch < sm_90: return 1
    # Benchmark 1, 2, 4, and 8. Default to 1 until a repeatable win exists.
    return 1
```

Any enabled value must be chosen from measured end-to-end latency, not
inferred from the Triton spill.

### Top-k variant

Each CTA produces k candidates per tile. The cluster reduction merges
`cluster_m * k` candidates into the top-k. Two options:

1. **Bitonic merge via DSMEM** (preferred for k=20-50): pair-wise bitonic
   merge across cluster CTAs, log2(cluster_m) rounds, each round one DSMEM
   exchange.
2. **Radix select via DSMEM histogram combine** (FlashInfer pattern): each
   CTA builds a local histogram, DSMEM combines histograms, find global
   threshold bin. Better for large k.

### Risks

1. **EVT + DSMEM choreography is novel.** No shipped CUTLASS EVT visitor
   does DSMEM cluster reduction. The Quack `cluster_reduce` is hand-rolled
   CuTe-DSL; embedding the same choreography in a CUTLASS EVT visitor
   (warp-specialized consumer warp) requires careful mbarrier timing.
   Mitigation: prototype standalone first, integrate second.
2. **Cluster launch constraints.** Cluster shape must be compatible with
   the GEMM tile shape (CUTLASS example 73 constraints).
3. **DSMEM bandwidth saturation.** At cluster_m=16, DSMEM bandwidth drops
   below HBM (ClusterFusion Fig 5). Stay at cluster_m=4-8.
4. **RNG sequence correctness.** Philox subsequence must include both
   V-tile index and cluster rank to keep CTAs' noise independent.

### Expected impact

No speedup range is predicted.
Quack demonstrates that clusters can recover bandwidth when they divide a
large per-CTA reduction domain.
The proposed CUTLASS FMMS cluster does not divide its fixed CTA accumulator
tile, so Quack's measured recovery is not transferable.

## The novel piece: RNG inside an EVT visitor

No shipped CUTLASS EVT visitor uses RNG. This is the one genuinely new
thing we have to build and validate.

### Design

```cpp
template <class ElementCompute, FloatRoundStyle RoundStyle>
struct FmmsArgmaxVisitor : Sm90VisitorImpl<> {
  struct Arguments {
    ElementCompute const* temperature_ptr = nullptr;  // 0-d GPU tensor
    uint64_t seed = 0;                                 // host scalar
    int vocab_size = 0;
    int num_samples = 1;  // runtime value, <= NUM_SAMPLES_MAX
    // Output pointers (local HBM or peer symm-mem in TP):
    ElementCompute* candidates_vals_ptr;
    int64_t*        candidates_idxs_ptr;
  };

  using Params = Arguments;

  // ... standard can_implement / get_workspace_size / initialize_workspace ...

  struct ConsumerStoreCallbacks {
    int block_v_start;
    int h_idx;

    CUTLASS_DEVICE void begin() {
      // Initialize any cached RNG parameters.
      // Do not derive the stream from blockIdx or scheduler order.
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc,
          int epi_v, int epi_m, int epi_n,
          Array<ElementCompute, FragmentSize> const& frg_scaled) {
      // For each element in the fragment:
      //   1. Read the global (v_idx, h_idx) coordinate from the node's
      //      coordinate tensors (tCcRow gives per-element global M/N,
      //      residue_tCcRow gives OOB predication).
      //   2. Map (seed, sample_idx, h_idx, global_v_idx) to a unique
      //      Philox counter and draw u ~ U(0,1)
      //   3. gumbel = -LOG2E * log2f(-log2f(u * (1 - epsilon)))
      //   4. noisy = frg_scaled[i] + gumbel
      //   5. Insert (noisy, global_v_idx) into the per-sample thread-local
      //      desc-sorted argmax array (top-1 or top-k).
      // Returns frg_scaled unchanged; the argmax is finalized in reduce().
    }

    CUTLASS_DEVICE void reduce(...) {
      // Reduce over M using the Sm90RowReduction lane/warp layout.
      // Exchange packed (f32 value, i32 index) states with 64-bit shuffles,
      // then use its shared-memory path across M warps.
    }

    CUTLASS_DEVICE void end_loop(int epi_m, int epi_n) {
      // Per sample s: write k (val, idx) pairs to
      // candidates[s, pid_v, h_idx, 0..k-1].
      // Plain st.global, not TMA (payload is tiny: 8 bytes for k=1,
      // ~200 bytes for k=25). Avoids the Blackwell TMA-store singleton
      // bug documented in findings/tma-store-blackwell-singleton-dims.md.
    }
  };
};
```

**Sample loop.** The epilogue iterates over `num_samples` within each
output tile, mirroring the Triton kernel's
`for sample_idx in range(num_samples)`. The accumulator fragments are
revisited per sample (the same revisit mechanism example 61 uses in
`reduce()`), so per-sample argmax state does not scale with
`num_samples`. `num_samples` is a runtime value bounded by a compile-time
`NUM_SAMPLES_MAX` sized to the test and benchmark values; the Python
wrapper batches larger requests across launches and concatenates, keeping
the provider call signature identical to the other backends. The
10M-sample large-vocabulary test already batches this way via
`SAMPLES_PER_CALL`.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| Stateless Philox arithmetic is expensive inside the epilogue | Prototype it outside GEMM and measure instruction count and SM/SFU pipe utilization before integration. |
| RNG sequence changes with tile or schedule | Derive counters only from global `(seed, sample, h, vocab)` coordinates. |
| RNG increases register pressure | Compare NCU registers and local-memory traffic against the identical greedy kernel. |
| Distribution correctness at large V | Re-run the chi-squared test at V=128k (`make modal-verify-correctness-large-vocab VOCAB_SIZE=128000 NUM_SAMPLES=10000000`). Must pass the same decision threshold as the Triton kernel (its reference draw: reduced chi-squared 0.99844, p=0.6503). |

### Why not pre-generated noise

Pre-generating Gumbel noise in a separate kernel and passing it as a tensor
(analogous to the CuTe-DSL fallback) defeats the kernel's purpose: at V=128k,
H=1, S=1, the noise tensor is 512 KB; at H=256 it is 128 MB. The I/O budget
we save by not materializing logits would be spent on noise instead. In-kernel
RNG is the only choice that preserves the bandwidth win.

## Implementation steps

Each gate produces a measurable artifact, is verified independently before
moving on, and can stop or redirect the implementation before later
complexity is added.

### Initial production milestone

The first shippable scope is intentionally narrow:

- BF16.
- TP1 first, then TP2, TP4, and TP8.
- B200 only until the complete B200 gate passes.
- Greedy and unrestricted Gumbel-Max.
- The small and large model shape families already used by the benchmarks.
- The full hidden-state sweep (H=1 through H=256). All H values are in
  scope for the CUTLASS backend.
- Existing two-stage candidate merge.
- Triton remains the fallback backend.
- Offline kernel discovery and B200 per-shape dispatch are part
  of the initial milestone because one schedule is not expected to cover
  H=1 through H=256.

Candidate-compression DSMEM is optional within the B200 milestone.
A100, FP8/FP4, H200, and B300 are outside the active B200 scope.

### Testing strategy

The production sampling tests are **provider-agnostic**. Every sampler
plugs into `get_sampler(provider, weights)` (a match/case in `core.py`),
and every test parametrizes over the `provider` string.

The early gates also need small dedicated tests that do not construct a
sampler:

- Extension build/import and ordinary GEMM output.
- Standalone packed max-with-index reduction.
- Stateless Philox stream generation and coordinate mapping.
- B200 compile-and-run coverage for every new gate.
- Existing H100 compile-and-run evidence remains a regression record but is
  not rerun unless shared code changes make a targeted compatibility check
  necessary.

Only register `fused-cutlass` after greedy FMMS produces valid candidates.

The existing test inventory that the CUTLASS kernel must pass:

| Test | File | What it verifies | How to run |
|---|---|---|---|
| `test_sampling_distribution` | `tests/test_core.py` | Chi-squared goodness-of-fit against theoretical softmax. V in {100, 256, 512}, H in {1, 2}, 10k samples. | `pytest tests/test_core.py::test_sampling_distribution -k fused-cutlass -v` |
| `test_top_k_top_p` | `tests/test_core.py` | Top-k/top-p restricts samples to the allowed token set (verified against `reference_top_k_top_p`). V in {100, 200, 256}, H in {1, 2}. | `pytest tests/test_core.py::test_top_k_top_p -k fused-cutlass-topk -v` |
| `test_top_k_top_p_large_vocab` | `tests/test_core.py` | Top-k/top-p at V=151936 with real Qwen3-0.6b weights. Checks no crash, valid token range, top-k restricts output. | `pytest tests/test_core.py::test_top_k_top_p_large_vocab -k fused-cutlass-topk -v` |
| `test_greedy_sampling` | `tests/test_core.py` | Greedy (no noise) returns the argmax token. Deterministic complement to the chi-squared test. | Needs a CUTLASS greedy variant (see below). |
| `test_fused_triton_return_logits` | `tests/test_core.py` | Logits match a PyTorch fp32 matmul reference (atol=1e-4). | Needs a CUTLASS `return_logits=True` flag (see below). |
| `verify_sampling_distribution_tp` | `src/fused_mm_sampling/testing.py` | Chi-squared in TP context. Runs all providers across V in {100, 256, 512}, H in {1, 2} via torchrun. | `make modal-pytest-distributed GPU=b200` |
| `verify_greedy_tp` | `src/fused_mm_sampling/testing.py` | Greedy argmax in TP context. | `make modal-pytest-distributed GPU=b200` |
| Large-vocab chi-squared | `assert_sampling_distribution_large_vocab` | V=128k, 10M samples, Gaussian random weights. Validates the f32-max fix and RNG stream independence. | `make modal-verify-correctness-large-vocab VOCAB_SIZE=128000 NUM_SAMPLES=10000000` |
| Speed test smoke | `test_speed_test_smoke` | Runs `speed_test` with minimal config across all providers. Catches crashes and shape mismatches. | `pytest tests/test_core.py::test_speed_test_smoke` |

The chi-squared test (`assert_sampling_distribution`) is the load-bearing
correctness gate. It has already caught real bugs (the bfloat16 per-tile
maxima bias at V=128k, the f32-max fix, RNG stream collisions). Any new
provider that passes this test at V=128k with 10M samples has a correct
sampling distribution to within chi-squared statistical power.

**Two diagnostic features the CUTLASS kernel needs that the Triton kernel has but
are not yet in the parametrize lists:**

1. **`return_logits=True` flag**: the CUTLASS EVT visitor must support a
   mode that writes the full `[H, V]` fp32 logits tensor to HBM (defeating
   the fusion, but enabling the logits-vs-reference test). This is the
   same flag as `fused_triton_ret_logits` in the Triton kernel.
   It can be a build-test entry point rather than a production provider. The test
   `test_fused_triton_return_logits` verifies logits match `hidden_states.float() @ weights.float().T`
   to atol=1e-4. This catches GEMM correctness bugs before the sampling
   epilogue is added (useful in Gates 0-2).

2. **`greedy_sampling=True` flag**: the CUTLASS EVT visitor must support a
   mode that skips Gumbel noise and returns the plain argmax. This is the
   same flag as `fused_triton_greedy`. The test `test_greedy_sampling`
   verifies the result matches `logits.argmax(dim=-1)`. This is a
   deterministic complement to the statistical chi-squared test and is
   especially useful for debugging the EVT visitor because it isolates
   the argmax reduction from the RNG.

3. **Cross-provider consistency** (optional, best-effort): the Triton and
   CUTLASS kernels use different, independently valid RNG streams (see
   "RNG sequence contract" below), so bit-identical samples are not
   expected and not required. If both kernels ever share a counter
   assignment, `test_cross_provider_seed_consistency` can assert
   `torch.equal` over a few thousand same-seed samples as a
   reproducibility bonus. It is never a gate; the chi-squared test is the
   only distribution-correctness gate.

### RNG sequence contract

The CUTLASS and Triton kernels use different, independently valid RNG
streams. Bit-identical samples across providers are not expected and not
required.

- CUTLASS kernel: a stateless Philox stream whose counter is derived only
  from global `(seed, sample_idx, h_idx, global_vocab_idx)` coordinates.
  It must not depend on block index, warp/lane ownership, tile scheduler
  order, autotuned tile shapes, or cluster size. Seed is a host scalar in
  `Arguments`.
- Triton kernel (status quo, unchanged): `_gumbel_noise` in `core.py`
  draws `tl.rand(seed + sample_idx, tile_noise_offsets)`, where the
  offsets are linearized from tile coordinates and the autotuned
  `BLOCK_SIZE_V`/`BLOCK_SIZE_H`. Its stream is therefore tile-shape
  dependent and cannot be reproduced by a kernel with different tiling.

Distribution correctness is gated per provider by the chi-squared test.
Cross-provider bit equality would additionally require replicating
Triton's tile-dependent counter linearization and its exact
uniform-to-Gumbel floating-point sequence; it is an optional
reproducibility experiment, never a gate.

### Stage-gate policy

Do not start a later gate until the current gate has a complete local validation
packet, correctness evidence, benchmark logs, and a written decision.
Commit the reproducible runner and finding, but keep generated validation
packets under `benchmarking/modal-results/` out of Git.
Record failed approaches rather than silently carrying them forward.

Initial thresholds:

| Gate | Required result | Failure response |
|---|---|---|
| Toolchain baseline | Reproducible H100 and B200 builds | Fix the build before kernel work |
| M reduction | Exact max-with-index across boundary shapes | Leave generic EVT and use a handwritten epilogue |
| B200 plain-GEMM discovery | A measured CUTLASS dispatch is within 5% of `torch.mm` for all 18 B200 model-shape and H cells | Expand only the official heuristic-selected population or document the remaining unsupported Blackwell family before stopping |
| Greedy FMMS | No more than 5% slower than plain CUTLASS GEMM plus the unavoidable reduction work | Rework or abandon the epilogue design |
| Gumbel-Max | Correct large-vocab distribution with no RNG-caused spill | Replace the RNG implementation or schedule |
| Top-k | Matches or beats the existing Triton top-k path | Keep Triton top-k as the dispatched fallback |
| TP | Correct TP2 with competitive total latency | Use local output plus a separate fan-out kernel |
| Optional optimization | At least 3% repeatable end-to-end improvement | Drop it |

The 5% and 3% values are starting policy, not facts.
Change them only before seeing the experiment being judged.

### Required validation packet for every formal gate

Do not mark a gate complete from a successful process exit alone.
Every formal gate or milestone must leave a human-readable validation packet under
`benchmarking/modal-results/cutlass/<number>-<gate-name>/`.
The packet must contain:

- `VERIFY.md`, as the entry point for the human verifier.
  It must give the review order, expected outcome, actual outcome, explicit
  failure criteria, and copy-paste commands that check the packet.
- `summary.json`, with the expected result, actual result, pass or fail status,
  architectures, dimensions, test counts, failure count, and exact commands.
- `case-summary.csv`, with one human-scale row per architecture, test case, and
  relevant configuration.
  It must contain separate expected, actual, error or mismatch count, and pass
  columns, plus the number of raw observations represented by the row.
- `cases.csv`, with the complete case-level or observation-level evidence used
  to produce `case-summary.csv`.
  This is the debugging record, not the primary review surface.
- `log.txt`, with the complete build and execution output.
- A finding under `findings/` that explains what was tested, what passed, what
  could have failed, why the gate's constraints remain useful, and what the
  result does not prove.

Performance gates must additionally save raw per-repetition measurements,
environment metadata, and profiler reports.
Their compact report must include repetition counts, the declared threshold,
the statistic used for the decision, uncertainty or variability where
applicable, and an explicit pass column.
Distribution gates must additionally save sample counts and the complete test
statistics.
Their compact report must include the test statistic, degrees of freedom,
p-value or other declared decision statistic, threshold, covered probability
mass where applicable, sample count, and an explicit pass column.
If an artifact does not apply, `summary.json` must say why instead of silently
omitting it.

A human verifier should be able to check a gate without reading kernel source:

1. Start with `VERIFY.md` and confirm that it states the expected and actual
   outcomes instead of only describing how the test works.
2. Confirm that `summary.json` names every required architecture and test case.
3. Confirm that the expected and actual counts match and `failure_count` is
   zero.
4. Inspect `case-summary.csv` for every required case, worst errors, and
   boundary cases, not only its final pass column.
5. Search `log.txt` for compiler errors, launch errors, sanitizer failures,
   skipped tests, NaNs, and unexpected fallbacks.
6. Confirm that the finding states the gate's limitations before approving the
   next gate.

The verifier must not need to inspect `cases.csv` or other raw measurements to
approve a passing gate.
The compact report must expose every required case and make missing coverage,
threshold failures, and mismatches obvious.
Raw evidence remains mandatory so a failure or surprising aggregate can be
investigated without rerunning the gate.

### Incremental code and artifact lifecycle

Incremental gates intentionally create diagnostic kernels, runners, and
validation packets.
Keep them organized so that completed experiments do not become an accidental
second implementation of the production kernel.

Use these rules:

1. Give each formal gate or milestone one canonical source or test harness,
   one runner, one Make target, one artifact directory, and one finding.
   Keep narrower correctness checks as test families in that harness.
   Do not retain alternate files with suffixes such as `new`, `fixed`, or
   `final`.
2. Keep gate-only CUDA harnesses under
   `src/fused_mm_sampling/csrc/cutlass/`.
   Keep their Modal entrypoints under
   `src/fused_mm_sampling/modal_lib/cutlass/`, and write all generated evidence
   only under the matching
   `benchmarking/modal-results/cutlass/<number>-<gate-name>/` directory.
3. When a primitive becomes part of the next gate or the production kernel,
   move its reusable implementation into one shared header or production
   source.
   Make the earlier harness test that shared implementation instead of copying
   it.
4. Reuse the pinned image, compilation helpers, packet-writing helpers, and
   validation schemas across gates.
   Extract a shared helper after the second real use, rather than cloning a
   runner and allowing the copies to diverge.
5. Keep only artifacts needed for human verification and failure diagnosis.
   Do not duplicate logs or raw measurements in findings, runner output
   folders, and ad hoc scratch directories.
   Findings summarize and link to the canonical packet.
6. After a gate is approved, remove failed prototypes, superseded runners,
   unused build commands, stale binaries, temporary output directories, and
   abandoned artifact formats before starting the next gate.
   Prefer `trash-put` for local cleanup when the target can reasonably be
   recovered.
7. Retain the approved gate's minimal reproducible harness, canonical runner,
   validation packet, and finding until equivalent or stronger coverage exists
   in the permanent test suite.
   Once permanent coverage replaces the harness, remove the redundant harness
   and runner, record the replacement in the finding, and keep the compact
   historical packet.
8. At the end of every gate, run `git status --short` and inspect the gate's
   artifact directory.
   The handoff must identify every retained gate-specific file and explain why
   it remains.

A gate is not ready for approval while unexplained scratch files, duplicate
runners, stale packet formats, or copied implementations remain.
Cleanup must not remove the only reproducer for an unresolved failure or the
only evidence supporting a completed gate.

### Gate 0: establish a reproducible toolchain baseline

Use a versioned Modal image and pin the CUTLASS revision used by each
benchmark series.
Record the base image, CUTLASS revision, CUDA toolkit, CUDA runtime, PyTorch,
host compiler, GPU, and architecture in every run.
Dependencies may be upgraded whenever the implementation needs a newer fix or
feature.
Treat an upgrade as a new baseline, record the new versions, and rerun the
ordinary-GEMM smoke checks before comparing kernel changes across it.

Build minimal ordinary-output GEMMs for H100 and B200.
Verify their outputs against an independent reference and preserve the build
commands in the Modal runner.
Do not expose a sampling provider yet.

**Exit:** both architectures build reproducibly and a small GEMM passes.

**Human verification:** check that the log records all pinned versions and
contains successful independent reference checks for both SM90a and SM100a.
Treat a missing architecture, an unrecorded dependency version, a reference
mismatch, or a build that reused an unexplained binary as a failure.
This gate proves only the ordinary GEMM toolchain, not the custom epilogue.

**Current baseline (validated 2026-07-29):**

- Modal base image: `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`.
- CUTLASS: 4.6.1 at commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`.
- CUDA toolkit: 13.0.88.
- PyTorch: 2.11.0+cu130.
- Host compiler: GCC 13.3.0.
- Targets: SM90a on H100 and SM100a on B200.

Run `make modal-cutlass GATE=toolchain` after changing any component of this
baseline.
The runner builds CUTLASS examples 48 and 71, executes an M=512, N=64, K=256
ordinary GEMM on each architecture, checks CUTLASS's independent device
reference, and writes the complete log to
`benchmarking/modal-results/cutlass/00-toolchain/smoke.txt`.
Both architecture checks passed on 2026-07-29.
The Gate 2a provider matrix and its shared greedy pytest cases were also rerun
on both architectures after the 4.6.1 upgrade.

### Gate 1: standalone M-axis max-with-index

Gate 1 was initially divided into micro-gates, each with a local validation
packet, focused tests, and a written result.
Gates 1a through 1f retain those original boundaries as historical records.
In hindsight, Gates 1b through 1f were useful development checkpoints but
were narrower than necessary as formal gates.
Future work uses one formal gate per integration boundary or distinct
mechanism and keeps thread, warp, CTA, column, boundary, and tie cases as test
families inside the relevant packet.

#### Gate 1a: map the accumulator fragment layout

**Status:** complete on 2026-07-29.

Build a diagnostic EVT that replaces accumulator values with ownership
metadata and lets the ordinary CUTLASS epilogue store it.
Record the mapping from thread, fragment slot, and epilogue iteration to
global `(M, N)` coordinates.
Run it independently on SM90 and SM100 because SM100 loads its accumulator
from TMEM into registers before the callback.
The checked-in runner is `make modal-cutlass GATE=accumulator-layout`, and the
observed layouts are documented in
`findings/cutlass/04-accumulator-layout.md`.

**Exit:** every output coordinate in one complete CTA tile has exactly one
owner, and the SM90 and SM100 mappings are saved and documented.

**Human verification:** inspect `sm90.csv` and `sm100.csv` for 16,384 rows,
16,384 unique `(m, n)` pairs, zero missing coordinates, and zero duplicate
coordinates.
Confirm that each summary reports the expected thread, fragment, and epilogue
iteration ranges.
Missing or duplicate owners, corrupted metadata, a GPU launch failure, or an
architecture absent from the packet fails the gate.
This gate maps ownership only and does not prove any reduction.

#### Gate 1b: reduce values owned by one thread

**Status:** complete on 2026-07-29.

Implement deterministic max-with-index over only the values owned by one
thread.
Do not use warp shuffles, shared memory, or partial tiles.
The checked-in runner is `make modal-cutlass GATE=thread-local-max`, and the result
is documented in `findings/cutlass/05-thread-local-max.md`.

**Exit:** exact value and lowest-index tie agreement with a CPU reference for
every thread-local fragment.

**Human verification:** require separate rows for unique maxima, negative
values, maxima in every fragment slot, and ties in both index orders on SM90
and SM100.
Expected and actual FP32 bit patterns and indices must match exactly.
This gate does not exercise warp communication or shared memory.

#### Gate 1c: reduce within one warp

**Status:** complete on 2026-07-29.

Add shuffle-based max-with-index across the M lanes of one warp.
Test unique maxima, ties, negative values, and maxima that cross lane
boundaries.
The checked-in runner is `make modal-cutlass GATE=warp-max`, and the result is
documented in `findings/cutlass/06-warp-max.md`.

**Exit:** exact agreement for every warp-local M domain.

**Human verification:** require cases whose winner originates in every
participating lane, plus cross-lane ties and all-negative inputs.
Check exact values and lowest indices against the CPU reference for every
warp.
A result from lane zero alone or a test that never moves the winner across a
shuffle boundary is insufficient.
This gate does not prove cross-warp reduction.

#### Gate 1d: reduce within one CTA

**Status:** complete on 2026-07-29.

Combine warp results through shared memory.
Restrict the problem to one complete M tile and one N column.
The checked-in runner is `make modal-cutlass GATE=cta-max`, and the result is
documented in `findings/cutlass/07-cta-max.md`.

**Exit:** exact max-with-index for one full CTA M tile.

**Human verification:** require winners from every contributing warp and ties
between different warps.
Check the complete output against the CPU reference and search the log for
race-check or synchronization failures.
This gate covers one full M tile and one N column only.

#### Gate 1e: support multiple N columns

**Status:** complete on 2026-07-29.

Extend the CTA reduction to every N column in the epilogue tile.
Verify that columns remain independent.
The checked-in runner is `make modal-cutlass GATE=cta-multi-column-max`, and the
result is documented in `findings/cutlass/08-cta-multi-column-max.md`.

**Exit:** exact results for one full CTA tile across its complete N extent.

**Human verification:** require independent maxima at different M positions
for every N column.
Inspect per-column expected and actual values and indices, including columns
at both ends of every epilogue N iteration.
Cross-column contamination or an untested column fails the gate.
This gate still excludes boundary tiles.

#### Gate 1f: handle boundary tiles

**Status:** complete on 2026-07-29.

Add explicit predication for partial M and N tiles.
Test dimensions immediately below, at, and above tile boundaries.
The checked-in runner is `make modal-cutlass GATE=cta-boundary-max`, and the
result is documented in `findings/cutlass/09-cta-boundary-max.md`.

**Exit:** exact results for M in `{100, 127, 128, 129, 255, 256, 257}` and
N in `{1, 2, 63, 64, 65, 127, 128, 129}`.

**Human verification:** confirm that `cases.csv` contains the full Cartesian
product of the declared M and N sets on both architectures.
Require winners at the final valid M coordinate and sentinel maxima in padded
coordinates, so broken predication cannot pass accidentally.
Any missing shape, out-of-bounds report, sentinel winner, or mismatch fails
the gate.

#### Gate 1g: integrate global candidates into a minimal EVT

**Status:** complete on 2026-07-29.

Feed ordinary GEMM accumulators into the proven CTA reduction and write one
candidate per M tile and N column.
Convert tile-local positions into global vocabulary indices in the same
integration.
Match `torch.max` tie behavior by choosing the lowest global index.
Use packed FP32 value plus i32 row index internally and widen the index only
at the public output.
Keep `FinalReduction=false`.
Do not add Stage 2 or sampling.

This gate has four required test families in one packet:

- Nonzero M-tile offsets and winners at both tile edges.
- Complete and partial M and N tiles.
- Negative and tied logits within a tile.
- Equal maxima in different tiles, with each emitted candidate retaining its
  correct global index before the later merge.

**Exit:** every candidate from every `(m_tile, n)` coordinate matches the
corresponding slice of `torch.matmul`, including its global index.

**Human verification:** save the reference logits or a reproducible input
seed and record every candidate rather than only the eventual winner.
Compare every candidate value bit pattern and global index with the
corresponding PyTorch slice on H100 and B200.
Require nonzero tile offsets, boundary winners, negative inputs, and
deterministic ties.
A correct final winner cannot hide an incorrect losing tile candidate.
This gate does not validate Stage 2.

The checked-in runner is `make modal-cutlass GATE=evt-candidates`, and the
result is documented in `findings/cutlass/10-evt-candidates.md`.

#### Gate 1h: integrate Stage 2 and close deterministic correctness

**Status:** complete on 2026-07-29.

Merge the Gate 1g candidates across M tiles with the existing GPU Stage 2.
Run the complete deterministic boundary and tie matrix on H100 and B200 in
the same packet.
A passing SM90 implementation does not imply that SM100 is correct.

The packet must retain both intermediate candidates and final outputs.
Required cases place the global winner in the first, middle, and last M tile
and include equal maxima in different tiles.

**Exit:** exact agreement with `torch.matmul(...).max(dim=0)` across the
complete deterministic matrix on both architectures.

**Human verification:** inspect intermediate candidates before final values
and indices.
Confirm that every declared case family and boundary shape is present on both
architectures, with no skips or architecture-specific omissions.
Group `cases.csv` by architecture and case family and verify zero candidate
and final-output failures.
Cross-tile ties must choose the lowest global index.
This gate closes deterministic reduction correctness but does not prove
sampling or production performance.

The checked-in runner is `make modal-cutlass GATE=stage2`, and the result is
documented in `findings/cutlass/11-stage2.md`.

**Fallback:** if a generic EVT stops expressing the reduction cleanly or
reliably during Gate 1, switch at that point to a handwritten CUTLASS
epilogue or kernel.
Do not add RNG to a questionable reduction.

### Gate 2: greedy TP1 FMMS and the performance feasibility gate

Gate 2 has two formal milestones because provider integration and the
performance go/no-go decision have different failure modes.

#### Gate 2a: expose the greedy TP1 provider

Wrap the deterministic Gate 1h kernel in the production sampler interface.
Support only BF16, TP1, greedy sampling, H100, B200, and the two primary model
shape families.
Run the provider-agnostic greedy tests and the full supported shape sweep.

**Exit:** the provider builds through the production path and produces exact
greedy outputs for every supported shape on both architectures.

**Human verification:** require the complete provider-agnostic result matrix,
including boundary shapes and deterministic ties.
Confirm that the production wrapper uses the candidate and Stage 2 path
validated in Gate 1h.
Missing shapes, fallback to another provider, compilation graph breaks, or any
output mismatch fails the gate.

The checked-in runner is `make modal-cutlass GATE=greedy-provider`, and the
passing result is documented in `findings/cutlass/12-greedy-provider.md`.

#### Gate 2b: greedy performance feasibility decision

**Status:** the current hand-selected schedule is no-go, but Gate 2 is reopened
for official kernel discovery in Gate 2c.

Profile the exact provider approved in Gate 2a without changing its
correctness path.

Treat Nsight Compute as a continuous kernel-development tool, not as a
one-time gate artifact or an optional follow-up after timing fails.
When a schedule, epilogue, reduction, or memory path changes materially, use
matched timing to locate the affected regime and profile representative fast
and slow cells before proposing a causal explanation or selecting the next
optimization.
Inspect instruction mix, atomic traffic, scheduler stalls, occupancy,
registers, local memory, cache and DRAM traffic, and tensor-core utilization as
applicable.
Refresh the profile whenever the implementation changes enough that an older
report no longer describes the production kernel.

The diagnostic FP32 `[V,H]` destination allocation and store inherited from Gate 1h have been removed.
The preferred callback-only implementation set `ElementD=void` while retaining the existing split-tree candidate EVT.
With CUTLASS 4.6.1, this compiles through the SM90 TMA collective after the EVT advertises a non-void `ElementAux` type and inherits its visitor constructors.
It does not compile through the SM100 TMA collective.
The SM100 builder accepts a void destination, but `sm100_epilogue_tma_warpspecialized.hpp` still instantiates D shared-memory layout arithmetic, D storage, and a D TMA descriptor with `ElementD=void`.
The first failures include division by zero in `StrideStageD` and attempts to construct a TMA store from a void element type.
The same implementation remains in NVIDIA CUTLASS `main` after the pinned 4.6.1 commit, so changing to an unreleased revision does not remove this blocker.

Do not approve an implementation that redirects the full D store into aliased or undersized scratch memory.
That would remove the logical allocation without removing the unwanted store traffic, and concurrent stores to the same addresses would introduce races.
Do not maintain separate SM90 and SM100 production semantics merely because the standard SM90 collective supports the void destination.

The implemented boundary is a narrow patch to the SM100 TMA epilogue collective that mirrors the existing SM90 void-D contract.
It substitutes the EVT auxiliary type for internal layout arithmetic, skips D descriptor construction and prefetch, disables shared-memory reuse that depends on D, and skips both the D shared-memory output copy and D TMA store.
The D-shaped shared-memory buffer remains as the workspace passed to `cst_callbacks.reduce`; it is not populated with or stored as the GEMM output.
The CUTLASS GEMM mainloop and the validated candidate reduction remain unchanged.
The production provider now uses the same `ElementD=void` semantics on SM90 and SM100.
The complete Gate 2a matrix passed again with 52/52 cases and 18/18 shared pytest cases per architecture.
The implementation and evidence are documented in `findings/cutlass/13-void-d-epilogue.md`.

The first complete H100 and B200 performance sweep failed the predeclared 5%
threshold in 24 of 36 configurations.
Profiling identified a serial 1,187-candidate dependency chain in Stage 2.
A cooperative one-CTA-per-column reduction reduced Stage 2 from 0.092-0.144
ms to 0.004-0.006 ms and passed the full Gate 2a regression.
The updated sweep passes 29 of 36 configurations.
The remaining seven failures are confined to H=1,2,4, and the worst ratio is
1.14.
Do not begin Gate 3 on the current implementation.
Before more fused-epilogue work, establish a dtype-matched ordinary CUTLASS
GEMM that remains within 5% of `torch.mm` for every primary shape and H value.
The current diagnostic CUTLASS GEMM plus argmax is slower than `torch.mm` plus
argmax in all 36 configurations, with median ratios of 1.17 on H100 and 1.27
on B200.
The initial comparison mixed FP32 CUTLASS output with BF16 `torch.mm` output, so
build a matched comparison before deciding whether CUTLASS tuning can close
the gap.
The bounded explicit-stage search did not pass the prerequisite.
That search covered only a manually selected 1-SM, `1x1x1` cluster family.
Its rule requiring a stage-count promotion before testing clusters was not a
valid exclusion of cluster scheduling because cluster multicast changes data
reuse independently of pipeline depth.
Treat `findings/cutlass/18-ordinary-gemm-stage-no-go.md` as the no-go record
for that narrow family, not for CUTLASS as a whole.
Only after Gate 2c passes should the approved plain-GEMM schedule be moved to
the fused epilogue experiment in Gate 2d.
The decision and evidence are documented in
`findings/cutlass/14-greedy-performance.md` and
`findings/cutlass/15-greedy-profile-stage2.md`.

**TODO:** Replace the downstream CUTLASS source patch with an upstream NVIDIA/CUTLASS fix.
Before opening the issue or pull request, extract a minimal SM100 `ElementD=void` reproducer with an EVT auxiliary output and add a focused regression test.
Keep the local patch pinned to the exact CUTLASS revision until an upstream release containing the fix is adopted and the full Gate 2a matrix passes without it.

Compare:

- Plain CUTLASS GEMM.
- CUTLASS GEMM plus a separate argmax.
- Greedy CUTLASS FMMS.
- Greedy Triton FMMS.
- `torch.mm` plus argmax.

Measure registers, local-memory traffic, occupancy, GEMM duration, Stage 2,
and total latency.
Measure the full hidden-state sweep (H=1 through H=256) on both
architectures, not only the compute-bound regime. Whether one GEMM kernel
covers the whole sweep or a separate GEMV path handles very small H (see
`findings/gemv-kernel-for-bsz1.md` and `tl_gemv.py`) is an implementation
choice decided from these measurements.
This milestone decides whether the custom epilogue preserves enough GEMM
performance to justify continuing.

**Exit:** the currently composed fused provider did not pass the predeclared
threshold.
Gate 2 remains open through Gate 2c and Gate 2d, while RNG, TP, and top-k stay
blocked.

**Human verification:** confirm that the Gate 2a correctness packet still
passes, then inspect raw repetitions, medians, dispersion, registers, local-memory
traffic, occupancy, component durations, and total latency for every provider
and H value.
The packet must compute the predeclared threshold directly and identify the
worst shape.
Do not accept pointwise minima, missing slow shapes, or a comparison made
across different toolchain baselines.
This gate does not prove RNG correctness.

#### Gate 2c: official plain-GEMM kernel discovery

**Status:** complete on 2026-08-01. Two independent confirmation runs each
selected a CUTLASS dispatch within 5% of `torch.mm` in all 18 B200 cells
(worst cell across both runs 1.049). The result, coverage audit, and
dispatch families are documented in
`findings/cutlass/19-gemm-heuristics.md`.
The checked-in runner is `make modal-cutlass GATE=ordinary-gemm-tuning`
with `PHASE=discover` and `PHASE=confirm RUN=<n>`; evidence is under
`benchmarking/modal-results/cutlass/14-ordinary-gemm-tuning/gate-2c-*`.

This gate replaces manual Cartesian tuning with NVIDIA's supported discovery
workflow.
It must run before any new fused-epilogue experiment.

##### 1. Define the baseline

Measure the exact BF16 operation `W[V,D] @ H[D,H]` with FP32 accumulation and
BF16 output through `torch.mm`.

`torch.mm` is the sole strong matmul baseline the provider must beat or stay
within 5% of.
It is the production PyTorch path and dispatches to NVIDIA's cuBLAS backend.
No separate library ceiling is in scope.

Log the PyTorch, CUDA, cuBLAS, driver, GPU, and clock versions.
Enable cuBLAS logging or use Nsight Systems to identify the kernel selected by
`torch.mm` where the tooling exposes it.
Kernel identification is diagnostic and must not change the baseline.

CUTLASS and `torch.mm` must use the same logical M/N/K, operand layouts,
BF16 inputs and output, FP32 accumulation, beta=0 semantics, alignment,
padding policy, preallocated buffers, stream, and cache state.
Record warm-L2 and cold-L2 results separately.
The production decision uses the state that matches decode-time execution;
the cold-L2 experiment remains useful for isolating weight streaming.
Never combine repetitions or baselines from different runs.

##### 2. Generate candidates with NVIDIA Matmul Heuristics

Create one problem JSON containing both primary `(V,D)` shapes and every
H value in `{1,2,4,8,16,32,64,128,256}` with the exact CUTLASS layout and
datatype description.
Run the generator on B200 for `sm_100a`.
Do not generate or profile Hopper candidates in this gate.

Build at least the top 16 heuristic configurations per problem initially.
If a failing shape's top candidates cluster tightly or omit a supported
family listed below, increase that shape to 32 candidates before making a
stop decision.
Use `CUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE`,
`CUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM`, and the emitted profiler
test list.
Do not translate heuristic results by hand before measuring them.

##### 3. Required candidate coverage audit

The generated manifest and profiler output must make the following dimensions
visible:

- CTA tile, instruction tile, pipeline stages, mainloop schedule, epilogue
  schedule, cluster shape, split-K, raster order, and swizzle.
- B200 1-SM and 2-SM dense BF16 schedules with their legal per-SM epilogues.
- Nontrivial static M-axis clusters on B200.
- Preferred and fallback flexible clusters on sm_100a when emitted or legal.
- Runtime raster order and swizzle variants for persistent schedulers.
- Split-K or Stream-K when emitted by the heuristic.

The heuristic is a search-space reducer, not a completeness proof.
If it omits one of these supported families for every problem, add a small
explicit control from that family or record the exact CUTLASS constraint that
makes it illegal.
Do not silently equate “not emitted” with “not useful.”

##### 4. Profile and select

Use `cutlass_profiler` with its emitted test list and retain complete
procedural kernel names plus runtime arguments.
Use locked clocks when the platform permits them.
Otherwise record clock samples and repeat any threshold-adjacent result.
Give each candidate enough profiling duration to make sub-5% distinctions;
NVIDIA's documented 50 ms fixed-duration example is the minimum starting
point, not a universal measurement guarantee.

Select the fastest measured CUTLASS candidate separately for each B200 model
shape and H.
Do not require one universal kernel.
Then determine the smallest piecewise dispatch that remains within 1% of the
per-case oracle and report the cost of simplifying that dispatch.

##### 5. Promotion and stop rules

Gate 2c passes when the selected CUTLASS dispatch is no more than 5% slower
than `torch.mm` in all 18 B200 production cases in one complete confirmation
run.

Repeat the full confirmation once.
Summarize runs independently with pandas and do not use pointwise minima
across runs.

A no-go decision is allowed only after:

1. The top-16 heuristic population has been measured for every problem.
2. Every failing problem has been expanded to top 32 or the heuristic has
   exhausted its distinct supported suggestions.
3. The required coverage audit has tested or formally excluded persistent
   B200 1-SM and 2-SM MMA, nontrivial clusters, flexible clusters,
   raster/swizzle, and heuristic-emitted split-K.
4. The `torch.mm` baseline has been measured under the matched protocol.
5. Two complete confirmation packets agree on the remaining failures.

##### 6. Gate 2c packet

The canonical packet must include:

- The B200 problem JSON and heuristic-generated test list.
- The generated kernel manifest and every rejection with its build,
  `can_implement`, or launch diagnostic.
- Raw CUTLASS profiler output and parsed pandas tables.
- Raw `torch.mm` timings.
- Per-candidate metadata and timings, the per-case oracle, and the simplified
  selected dispatch.
- Warm-L2 and cold-L2 summaries.
- Two independent full confirmation summaries.
- Toolchain, clock, workspace, padding, layout, and cache-policy metadata.

**Exit:** one reproducible plain-CUTLASS dispatch passes all 18 B200
production comparisons twice, or the fully audited Blackwell search records a
new no-go.

**Human verification:** confirm that every problem appears in both baseline
and CUTLASS tables, inspect the required-family coverage audit, verify that
the selected row comes from the same run as its baselines, and reject any
packet based on cross-run minima or undocumented padding/layout differences.

#### Gate 2d: transplant the winning B200 schedules into FMMS

**Status:** Complete on B200.
The per-CTA fused-EVT path plus cooperative Stage 2 passes 8,612 exact
intermediate and final comparisons, memcheck, racecheck, and the production
correctness suite.
The final H=1--256 performance packet passes the predeclared CUTLASS baseline
threshold with a worst ratio of 1.044.
At H=256 it is 3.2--4.2% faster than `torch.mm` plus argmax and 18.2--42.3%
faster than Triton.
The supporting timing and NCU evidence is in
`22-winning-schedule-performance.md`.

Reproduce only the measured B200 winning schedule families in the production
C++ builder.
Preserve their CTA and instruction shapes, 1-SM or 2-SM policy, cluster shape,
stage policy, tile scheduler, raster order, swizzle, and epilogue contract.

A changed schedule, CTA tile, epilogue tile, or 2-SM ownership model can change
which consumer thread owns each accumulator element.
Rerun Gate 1a on B200 first, derive the sm_100 ownership formulas from
evidence, and then rerun Gates 1b through 2a on B200.
Do not adapt the old ownership formulas by inspection.

After correctness passes, compare:

- Winning plain CUTLASS GEMM.
- Winning plain CUTLASS GEMM plus separate argmax.
- Greedy CUTLASS FMMS with the candidate epilogue.
- `torch.mm` plus argmax.
- Greedy Triton FMMS.

Measure GEMM, candidate epilogue, Stage 2, and total latency separately.
Collect registers, local memory, shared memory, occupancy, achieved tensor-core
throughput, HBM traffic, TMA traffic, and cluster utilization where supported.

**Exit:** the correctness-approved fused path passes the existing Gate 2b
threshold against both the winning plain CUTLASS schedule plus unavoidable
reduction work and the production `torch.mm` baseline.
Only then may Gate 3 begin.

**Human verification:** verify that the fused kernel really uses the winning
plain schedule and runtime parameters, that all dependent correctness gates
were rerun, and that the component timings explain the difference between
plain and fused performance.

### Gate 3: B200 stateless Philox prototype

Implement or adapt a counter-based Philox primitive from an established CUDA
implementation such as FlashInfer.
Map every random value from global
`(seed, sample_idx, h_idx, global_vocab_idx)` coordinates.
Do not keep long-lived `curandStatePhilox4_32_10_t` state in the epilogue and
do not derive streams from block, warp, lane, scheduler, or cluster order.

Gate 3 remains one formal gate with two ordered phases because both phases
approve the same standalone primitive.
The correctness phase must pass before cost profiling begins.

Phase A validates:

- Stream uniqueness across tiles and samples.
- Invariance under different launch and tile shapes.
- Uniform-distribution checks.
- Reproducibility on B200 across every promoted tile and scheduler family.

Phase B records:

- Generated instruction count and register footprint.
- SM issue-slot and SFU (MUFU) pipe utilization.

The Gumbel transform issues two `log2f` (LG2) operations per element on the
low-throughput SFU pipe.
At full weight-stream bandwidth the SFU could bind before HBM does, so measure
the pipe rather than only instruction count.

**Exit:** correct, stable streams with an acceptable measured cost.

**Human verification:** inspect stream-collision counts, reproducibility
comparisons, launch-shape invariance, uniformity statistics, instruction
counts, register use, and SFU utilization on B200.
The packet must record the predeclared statistical significance and cost
thresholds.
A p-value alone is insufficient without sample count, test statistic, and
multiple-test policy.
This gate tests uniform Philox output, not the Gumbel transform inside GEMM.

### Gate 4: B200 Gumbel-Max TP1

Add the validated stateless Philox and Gumbel transform to the working greedy
kernel.
Measure the incremental cost relative to greedy with the same GEMM
configuration.

Gate 4 has three ordered acceptance phases in one integration packet:

1. Run deterministic stream and provider-agnostic correctness tests.
2. Run the large-vocabulary 10M-sample chi-squared test.
3. Only after both pass, profile registers, local-memory traffic, latency, and
   SM/SFU pipe utilization against the identical greedy kernel.

**Exit:** correct sampling distribution on B200, no RNG-caused spill, and
acceptable incremental latency.

**Human verification:** require provider-agnostic test rows and the complete
10M-sample large-vocabulary statistics on B200.
Inspect reduced chi-squared, p-value, covered probability mass, excluded-bin
count, reproducibility, registers, local-memory traffic, SFU utilization, and
paired greedy-versus-Gumbel latency.
Distribution failure, new spill, or an undeclared latency regression fails the
gate.
This gate is TP1 and does not prove distributed stream uniqueness.

**Fallback:** try a different stateless Philox implementation or a
non-warp-specialized epilogue schedule.
Pre-generated noise is not an acceptable production fallback.

### Gate 5: B200 tensor parallelism

Implement TP before top-k and DSMEM.
Gate 5 has three formal milestones because each introduces a separate
distributed mechanism or scaling decision.

#### Gate 5a: TP2 correctness with local candidates

Implement and compare two local-candidate representations.
The per-tile fallback writes local candidates and uses a separate P2P fan-out kernel, matching the existing Triton communication structure without assuming its overlap benefit transfers.
The atomic-final-reduction alternative first reduces each rank's complete vocabulary shard to one packed candidate per hidden state.
It then performs an integer-MAX collective over only H packed 64-bit candidates before decoding the winning index.

Do not apply integer MAX directly to the existing raw `(FP32 bits, i32 index)` packing.
Encode the FP32 value with an order-preserving bit transform, invert the global token index so lower indices win ties, and account explicitly for signed versus unsigned 64-bit collective comparison semantics.
Reject NaNs or define and test their ordering before admitting this representation.
Validate the encoding exhaustively over positive and negative values, infinities, signed zero, adjacent FP32 values, and equal-value index ties before using it in a collective.

For the packed-MAX path, perform atomic reduction only into rank-local output.
Do not issue remote atomic operations from every vocabulary CTA.
The local winner is not complete until all GEMM CTAs finish, so the collective must follow GEMM completion and cannot claim tile-level compute/communication overlap.
The collective payload is `8H` bytes per rank and replaces per-tile candidate fan-out, the explicit symmetric-memory barrier, and the final world-size candidate reduction when the backend provides the required exact MAX semantics.
If a suitable collective is unavailable or slower, fan out the H final local candidates through symmetric memory and reduce the world-size candidates locally.

Measure four components separately for both representations: local GEMM and reduction, initialization, distributed exchange, and final decoding or reduction.
Compare the packed integer-MAX collective against symmetric-memory fan-out on the same hosts, because collective launch latency may dominate an H<=256 payload.
This comparison remains the correctness and performance fallback decision rather than assuming either transport wins.
Validate TP2 with `make modal-pytest-distributed`.

**Exit:** TP2 distributed correctness passes through both applicable paths, and one local-candidate exchange is selected from paired total-latency measurements.

**Human verification:** require per-rank candidates and outputs, plus explicit
confirmation that every rank selected the same global winner.
Include rank-local boundary winners, cross-rank ties, and distributed RNG
stream uniqueness.
For packed MAX, inspect the encoded keys and decoded global indices, including negative values and equal-value lower-index ties.
Record collective dtype and signed comparison semantics.
This milestone does not test direct peer stores from the GEMM epilogue or TP4/TP8.

#### Gate 5b: direct peer-store decision

Test direct stores to peer symmetric-memory buffers from the epilogue.
Compare them with the selected Gate 5a local-output path using identical
inputs and hosts.
Measure local compute, fan-out or direct stores, barrier, final reduction, and
total latency separately.
Do not assume the Triton overlap speedup transfers to CUTLASS.
The direct-store path exchanges per-tile candidates and preserves potential
overlap, while packed MAX exchanges only final local winners after GEMM.
Treat this as an end-to-end latency tradeoff between overlap and payload
reduction, not as a communication-only comparison.

**Exit:** retain direct stores only if they preserve correctness and pass the
predeclared total-latency threshold.
Otherwise keep the local-output fallback as the production path.

**Human verification:** inspect paired component and total timings rather than
only kernel duration.
Require identical results from both paths and record the selected production
path explicitly.

#### Gate 5c: TP4 and TP8 scaling

Expand the selected TP path to TP4 and TP8 only after TP2 is stable.
Run distributed correctness and paired scaling measurements at both world
sizes.

**Exit:** distributed correctness passes at TP4 and TP8 and the scaling packet
records the complete supported range.

**Human verification:** require per-rank agreement, complete host and topology
metadata, raw repetitions, component timings, and no missing world-size or
shape cells.
This milestone does not validate top-k.

### Gate 6: B200 fixed-K top-k and top-p

Gate 6 has two formal milestones because exact selection and integrated
sampling answer different correctness questions.

#### Gate 6a: exact fixed-K candidate selection

Implement the production-relevant fixed K first, initially K=20 or a padded
K=32.
Use a custom warp-group sorted-array or merge network derived from example
61.
Do not put the private `cub::detail::block_topk` API on the critical path.

**Exit:** exact top-k membership and deterministic tie ordering on B200.

**Human verification:** inspect adversarial membership cases at tile
boundaries, duplicate cutoff values, all-negative inputs, and K values around
the internal padded K.
Save every selected value and global index.
A later distribution pass cannot substitute for exact membership.

#### Gate 6b: integrated top-k sampling and performance

Reuse `_topk_merge_and_sample` for the global Stage 2.
Validate against the renormalized global top-k distribution and compare with
the existing Triton top-k implementation.
Exercise every production top-p value supported on the selected top-k
survivors and verify the final allowed set and distribution.

**Exit:** the sampling distribution passes and the predeclared performance
target passes on B200.

**Human verification:** retain the approved Gate 6a membership packet, then
inspect normalized probabilities, distribution statistics, and raw latency
repetitions for CUTLASS and Triton.
Record the exact K and padded internal K for every row.

**Development fallback:** dispatch top-k to the Triton backend while keeping
CUTLASS for greedy and unrestricted Gumbel-Max.
This hybrid path cannot pass Gate 7 and therefore cannot unlock Hopper work.

### Optional B200 optimization before Gate 7: DSMEM candidate compression

This optional gate concerns candidate reduction through DSMEM after the GEMM.
It is distinct from the GEMM scheduling clusters and TMA multicast required in
Gate 2c.
Profile candidate writes, Stage 2, and TP exchange first.
Attempt candidate compression only if those operations consume enough latency
for a 3% total improvement to be plausible or if it is needed to win a
specific declared Gate 7 cell.

Prototype the DSMEM max-with-index exchange outside GEMM, then integrate a
cluster along M.
Compare cluster sizes 1, 2, 4, and 8 with paired end-to-end runs on B200.
Keep it only if the complete path improves by at least the predeclared
threshold and does not regress another required cell.

**Human verification:** inspect paired raw runs for cluster sizes 1, 2, 4,
and 8, including candidate traffic, Stage 2, TP exchange, and total latency.
The mean or median total improvement must exceed 3% with repeatability across
the declared shapes, unless the experiment was admitted to fix a named Gate 7
cell, in which case that cell must become a repeatable pointwise win without
creating a loss elsewhere.
An isolated microbenchmark improvement or candidate-byte reduction does not
pass this optional gate.

### Gate 7: complete B200 implementation and superiority gate

Do not defer plain-GEMM kernel discovery to this gate.
Gate 2c must already have selected B200 shape-specific schedules.
Tune only fused-path mechanisms that survived the earlier gates:

- Small adjustments around the Gate 2c winning GEMM and epilogue shapes.
- Pipeline stages only when the custom epilogue changes shared-memory
  carveout enough to invalidate the plain winner.
- Cluster, raster, swizzle, or scheduler settings already validated by the
  winning plain family.
- H bucket dispatch.
- B200-specific dispatch simplification that remains within the declared
  pointwise performance requirement.

Run the full benchmark matrix, memory-traffic profiles, and end-to-end vLLM
TPOT experiments.
The vLLM integration must invoke the CUTLASS implementation for the complete
sampling path under test.
A hidden fallback to Triton, `torch.mm`, or another provider fails the gate.
Record provider selection and kernel names in the server log for every
end-to-end run.

Before launching the final packet, materialize the complete B200 comparison
matrix in a checked-in or packet-local manifest.
It must include every configuration supported by the corresponding Triton
FlashSampling provider:

- Both primary `(V,D)` model-shape families.
- Every H value in `{1,2,4,8,16,32,64,128,256}`.
- TP1, TP2, TP4, and TP8 wherever the Triton provider supports that world
  size.
- Greedy, unrestricted Gumbel-Max, production fixed-K top-k, and every
  supported top-p-on-survivors mode.
- Every sample-count, temperature, and other benchmark dimension exposed by
  the canonical runner for those providers.
- Every declared vLLM model and concurrency cell in the B200 end-to-end
  benchmark.

The manifest is the definition of “every dimension and configuration.”
Missing or skipped cells fail the gate unless the same cell is unsupported by
the Triton provider and the exclusion is recorded before measurement.

For every kernel-level cell, compare paired raw repetitions from CUTLASS and
Triton FlashSampling on the same host and run.
For every end-to-end cell, compare matched vLLM runs with the same model,
request distribution, concurrency, TP degree, server configuration, and host
class.

The CUTLASS median latency or TPOT must be strictly lower than Triton's in
every cell in each of two independent confirmation runs.
Do not average across H, shapes, TP degrees, features, models, or concurrency.
Do not use pointwise minima across runs.
Report ratios as `Triton / CUTLASS`, so every required performance ratio must
be greater than 1.0.

Memory traffic and temporary allocation must not regress enough to invalidate
the fused design.
Any allowed metric tolerance must be declared before the final runs and may
not weaken the strict pointwise latency and TPOT requirement.

**Exit:** all B200 correctness packets remain green, feature coverage matches
the declared Triton FlashSampling surface, and CUTLASS is strictly faster in
every kernel and end-to-end performance cell in both confirmation runs.
Only this exit unlocks Gate 8.

**Human verification:** require raw repetitions, host and toolchain metadata,
memory-traffic reports, and end-to-end summaries for every declared cell.
Check missing cells, dispersion, warmup policy, autotune state, and whether
kernel-level changes agree with end-to-end bounds.
This gate supports production selection but does not generalize beyond the
tested B200 models, shapes, and host classes.

### Gate 8: Hopper port after B200 completion

**Status:** blocked on Gate 7.

Port the complete, approved B200 feature surface to H100 first and H200 after
H100 is stable.
Reuse architecture-independent contracts for API shape, deterministic ties,
RNG coordinates, Stage 2, TP behavior, and top-k semantics.
Do not reuse Blackwell-specific GEMM schedules, TMEM ownership, 2-SM
assumptions, cluster constraints, or accumulator visitation formulas.

Repeat the official plain-GEMM discovery for sm_90 with Hopper
non-persistent, persistent ping-pong, and persistent cooperative schedules,
legal clusters, raster order, swizzle, stages, and heuristic-emitted split-K.
Rerun Gate 1a and every dependent correctness gate using the winning Hopper
schedule before enabling sampling.

Bring features up in the same order as B200: greedy TP1, Philox and
Gumbel-Max, TP2/4/8, top-k, then end-to-end validation.
Use `torch.mm` as the sole strong matmul baseline and Triton FlashSampling as
the pointwise fused-performance baseline.

**Exit:** the Hopper implementation reaches the same correctness, feature
coverage, and pointwise performance standard used by the B200 Gate 7 packet.

**Human verification:** require a fresh sm_90 kernel-discovery packet,
accumulator ownership evidence, all dependent correctness packets, and two
complete pointwise performance confirmations.

## Validation strategy

Reuse the provider-agnostic sampling and distributed tests after a provider
exists.
The toolchain, standalone M-reduction, and stateless Philox gates use small
dedicated tests.
All active validation in Gates 2c through 7 runs on B200.
Hopper validation resumes only in Gate 8.

### Correctness

- `make modal-verify-correctness-tp1` for V in {100, 256, 128k}.
- `make modal-verify-correctness-large-vocab VOCAB_SIZE=128000
  NUM_SAMPLES=10000000` for the chi-squared test.
- `make modal-pytest-distributed` for TP2/4/8.
- For top-k: extend the chi-squared test to sample with top-k=20 and verify
  the empirical distribution over the top-k survivors matches the
  renormalized softmax.

### Performance

- Gate 2c uses NVIDIA Matmul Heuristics and `cutlass_profiler` to discover
  plain kernels, followed by the matched project runner for confirmation
  against `torch.mm`.
- The existing `triton_benchmark` harness: same configs, swap the provider.
- **Key comparison**: CUTLASS FMMS vs Triton FlashSampling pointwise in every
  B200 cell.
  Gate 7 requires strict CUTLASS wins at H=1 through H=256, not only the
  compute-bound regime.
- `torch.mm` remains the strong plain-matmul and sampling baseline, but it
  does not replace the required pointwise comparison with Triton
  FlashSampling.
- HBM traffic: re-run `make modal-memory-traffic-all`. CUTLASS FMMS should
  preserve the fused algorithm's low-traffic behavior and explain any
  difference from Triton FlashSampling.

### Profiling

- NCU per-kernel report (existing `parse-memory-traffic` infrastructure).
- Nsys end-to-end trace in vLLM (existing `modal-nsys-profile`).
- For the EVT visitor: NCU source-view to confirm the RNG and argmax
  instructions are where expected (in the consumer warp, not spilled).

## Current implementation files

The original file plan has been realized and reorganized during Gates 0-2.
Use the current structure rather than creating a second extension:

1. `src/fused_mm_sampling/csrc/cutlass/greedy_provider.cu` contains the
   production-facing greedy provider, plain-GEMM diagnostic dispatch, and
   Stage 2 launch.
2. `src/fused_mm_sampling/csrc/cutlass/evt_candidates.cu` contains the fused
   candidate EVT and architecture-specific CUTLASS composition.
3. `src/fused_mm_sampling/csrc/cutlass/max_with_index.cuh` contains the
   shared deterministic comparator and reduction primitives.
4. `src/fused_mm_sampling/cutlass_impl.py` owns JIT compilation and Python
   bindings.
5. `src/fused_mm_sampling/modal_lib/cutlass/` contains the canonical Modal
   gate runners and shared image helpers.
6. `Makefile` maps every allowlisted CUTLASS gate to its runner and numbered
   result directory.

## Gate 2c files to add or modify

Keep Gate 2c under the existing `ordinary-gemm-tuning` gate rather than
creating a parallel tuning framework.

1. Add a checked-in exact-problem JSON generator or static JSON under the
   existing CUTLASS runner directory.
2. Add a reproducible Modal build step for `nvidia-matmul-heuristics` and
   `cutlass_profiler` using the pinned CUTLASS revision.
3. Extend the canonical runner to preserve heuristic test lists, generated
   manifests, profiler CSVs, rejection diagnostics, and matched `torch.mm`
   timings.
4. Add only promoted winner families to `greedy_provider.cu` after the
   profiler search passes.
5. Keep generated sources, binaries, and evidence ignored under the canonical
   Gate 2c result directory.
6. Do not hand-copy dozens of generated candidates into the torch extension.
   Use the profiler to select first, then retain only the minimal winning
   dispatch families needed by production.

## Honest assessment of risks

The first risk is selecting a plain CUTLASS kernel family competitive with the
`torch.mm` path on the unusual very-large-M, small-N, large-K shapes.
Gate 2c addresses this with NVIDIA's heuristic and profiler instead of a
hand-selected kernel list.

The second performance risk is whether the M-axis max-with-index visitor can
preserve the winning plain CUTLASS schedule's performance on B200.
Gate 1 proves the reduction mechanics, and Gate 2d measures the transplanted
Blackwell schedule before RNG, top-k, TP, or candidate-compression DSMEM add
complexity.
Hopper schedule preservation is a separate Gate 8 risk after B200 completion.

The next risk is Blackwell callback compatibility.
CUTLASS does not ship a corresponding `Sm100TopKSoftmaxColReduction`, so
B200 compile and correctness coverage begins at Gate 0 rather than being
deferred to tuning.

Another risk is RNG cost and register pressure.
The plan uses a standalone stateless Philox prototype, global coordinate
mapping, and a direct greedy-versus-Gumbel profile.
It does not rely on persistent cuRAND state.

Top-k is on the B200 completion critical path because Gate 7 requires the full
declared Triton FlashSampling feature surface.
The first path is a fixed-K warp-group merge.
The private `cub::detail::block_topk` API is only an optional experiment.

TP has a correctness-preserving fallback: local candidate output followed
by a separate P2P fan-out kernel.
Direct peer stores are an optimization.

Candidate-compression DSMEM is not required by feature semantics.
It is attempted only when measured candidate costs make a predeclared
end-to-end improvement plausible or when a named Gate 7 cell needs it to
become a pointwise win.

## Out of scope for this plan

- **Single-kernel full-vocab reduce** (CCE-style atomic-max-with-argmax
  across V tiles in one launch). The stage-2 merge is already a tiny
  fraction of total latency (~5-10us launch overhead). Not worth the
  complexity. Revisit only if end-to-end profiling shows stage-2 dominates.
- **Cluster DSMEM reduction** for inter-rank TP. Cluster DSMEM is
  intra-rank only; inter-rank still needs NVLink. The current symm-mem P2P
  path is the right mechanism.
- **Unbounded manual GEMM tuning after Gate 2c**. Native 2-SM MMA,
  heuristic-emitted split-K, persistent schedules, preferred/fallback
  clusters, raster order, and swizzle are in scope for the bounded official
  search. Expanding beyond the audited top-32 population requires a specific
  profiler result or unsupported-family gap, not a generic request to try more
  combinations.
- **A100 (sm_80) CUTLASS support**. The Triton kernel remains the A100
  fallback. CUTLASS 3.x SIMT GEMM path is possible but low priority.
- **FP8 / FP4 weights**. Out of scope for the initial implementation.
  BF16 only.

## References

### Official GEMM discovery and scheduling references

- [CUTLASS GEMM Heuristics](https://docs.nvidia.com/cutlass/4.5.1/media/docs/cpp/heuristics.html)
  documents the `nvidia-matmul-heuristics` integration, exact-problem JSON,
  top-N configuration generation, emitted profiler test lists, and Hopper and
  Blackwell coverage.
- [CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)
  documents exhaustive kernel generation, instantiation levels, procedural
  names, stages, cluster shapes, raster order, swizzle, fixed-duration
  profiling, flexible Blackwell clusters, and profiler test lists.
- [NVIDIA: Improving GEMM Kernel Auto-Tuning Efficiency with Heuristics and CUTLASS](https://developer.nvidia.com/blog/improving-gemm-kernel-auto-tuning-efficiency-on-nvidia-gpus-with-heuristics-and-cutlass-4-2/)
  explains why manual tuning over a few templates is insufficient and shows
  the intended heuristic-generation-profiling workflow on H100 and B200.
- [CUTLASS 3.x GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)
  defines the collective, kernel, and device composition layers and explains
  Hopper non-persistent, ping-pong, and cooperative schedules.
- [Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)
  is the authoritative table for 1-SM and 2-SM `tcgen05.mma` dispatch
  policies, legal MMA tiles, per-SM epilogue tiles, layouts, and clusters.
- [CUTLASS dispatch policies](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/dispatch_policy.hpp)
  is the source of the Hopper schedule tags and Blackwell
  `KernelTmaWarpSpecialized1SmSm100` and
  `KernelTmaWarpSpecialized2SmSm100` policies.
- [CUTLASS Operator API overview](https://docs.nvidia.com/cutlass/latest/media/docs/operators/overview.html)
  records NVIDIA's newer operator-discovery direction, including preferred
  and fallback clusters and static versus dynamic scheduling.
- [CUTLASS overview](https://docs.nvidia.com/cutlass/latest/index.html)
  states the library's goal of peak-performance GEMM on Hopper and Blackwell
  and distinguishes optimized templates from merely composing a legal
  kernel.

`torch.mm` remains the only comparison baseline in this project.
The discovery references above are used to search CUTLASS's own kernel space,
not to introduce another matmul baseline.

### CUTLASS source files

Use the documentation links above for current capabilities, but compile and
inspect source at the Gate 0 pinned CUTLASS 4.6.1 commit
[`e05f953a5b3d38adc240df2ff928e0421c2abba3`](https://github.com/NVIDIA/cutlass/tree/e05f953a5b3d38adc240df2ff928e0421c2abba3).
If current documentation describes a feature absent from that commit, either
exclude it with evidence or establish a new toolchain baseline before using a
newer revision.

- `examples/61_hopper_gemm_with_topk_and_softmax/` - EVT top-k reference.
- `include/cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp` - source
  of top-k merge helpers, not the FMMS M-axis visitor layout.
- `include/cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp`:
  base classes.
- `examples/71_blackwell_gemm_with_collective_builder/` - Blackwell EVT.
- `examples/73_blackwell_gemm_flexible_cluster/` - sm_100a preferred and
  fallback cluster configuration.
- `examples/77_blackwell_fmha/`, `examples/88_hopper_fmha/` - hand-written
  epilogue patterns (the alternative to EVT).
- `examples/91_fp4_gemv/` - memory-bound GEMV reference (cp.async + warp
  shuffle, no tensor cores). Validates that at H=1 the SIMT path is
  correct.
- `examples/93_blackwell_low_latency_gqa/` - DSMEM cluster reduction
  pattern. Useful reference for the *intra-rank* reduction, not inter-rank.
- `include/cutlass/gemm/kernel/gemv_blockscaled.h` - the GEMV
  implementation behind example 91.

### CCCL / CUB

- `cub/cub/block/block_topk.cuh` - the top-k primitive.
- `cub/cub/block/specializations/block_topk_air.cuh` - the AIR radix-select
  algorithm.
- `cub/cub/block/block_radix_sort.cuh` - fallback full sort.
- `cub/cub/block/block_reduce.cuh` - for `DataAndIndex` argmax.

### Reference implementations

- `apple/ml-cce` - Triton fused linear+LSE. Architectural template.
- `linkedin/Liger-Kernel` - Triton fused linear+CE. Less relevant (does
  not fuse the matmul).
- `flashinfer-ai/flashinfer` `include/sampling.cuh` - CUDA Gumbel-Max
  with `DataAndIndex` argmax. Direct template for the per-tile argmax.
- `Dao-AILab/quack` - CuTe-DSL memory-bound kernels. Reduction template
  pattern.

### Project findings (this repo)

- `findings/cutlass/03-dsmem-cluster-reduction.md` - the deep dive on DSMEM
  candidate compression.
  It defines the optional post-TP profiling gate.
- `findings/cutlass/16-ordinary-gemm-specialization.md` - the first matched
  `torch.mm` comparison and small-N specialization.
- `findings/cutlass/17-ordinary-gemm-tuning.md` - the six-control retained
  candidate sweep and its now-superseded manual tuning handoff.
- `findings/cutlass/18-ordinary-gemm-stage-no-go.md` - the explicit-stage
  no-go for the manual cluster-1 family and the correction that reopens Gate
  2c.
- `findings/cutlass/21-winning-schedule-evt.md` - the first correct 2-SM
  fused-EVT reduction and the per-CTA M-coordinate rule that prevents
  double-counting the cluster rank.
- `findings/cutlass/02-topk-softmax-epilogue.md` - detailed analysis of
  example 61, including the constraints that motivate this plan's
  two-stage architecture.
- `findings/fused-top-k-top-p-feasibility.md` - earlier analysis that
  reached the same two-stage conclusion from the Triton side.
- `findings/register-spilling-bsz256.md` - the 118 MB spill in the current
  persistent Triton kernel.
  It must not be projected onto a conventionally tiled CUTLASS kernel.
- `findings/arithmetic-intensity-decode-matmul.md` - why FMMS is
  memory-bound at H<128, the regime where the CUTLASS port matters least.
- `findings/cutlass/00-2cta-mma-operand-swap-regression.md` - why swapping V onto N
  to reuse example 61 is out, and why 2-CTA MMA is not the baseline path.
- `findings/tma-store-blackwell-singleton-dims.md` - why per-tile candidate
  writes use plain `st.global`, not TMA.
- `findings/tp2-collective-overhead.md` - why symm-mem P2P replaced NCCL.

### Hardware / programming guides

- [CUTLASS 3.x backwards compatibility](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html)
  is the canonical reference for the collective and kernel schedule layers.
- [NVIDIA Blackwell Architecture Technical Brief](https://resources.nvidia.com/en-us-blackwell-architecture)
  covers `tcgen05.mma`, TMEM, and cluster capabilities at the architecture
  level.
- Quack blog "Getting Memory-bound Kernels to Speed-of-Light" (2025-07-10)
  for the TV-layout + cluster reduction methodology.

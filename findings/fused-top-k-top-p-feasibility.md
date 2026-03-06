# Fused top-k and top-p sampling in the FMMS kernel

## Context

The FMMS kernel fuses matrix multiplication with categorical sampling via the
Gumbel-max trick: `sample = argmax_i(logits_i + Gumbel_i)`. Argmax is a
streaming reducible operation — each V-tile computes its local max, a cheap
reduction finds the global winner, and the full `[V, H]` logit matrix is never
materialized. This is the core insight that makes fusion possible.

However, the kernel currently only supports vanilla categorical sampling
(temperature-scaled). Many inference workloads require **top-k** and/or
**top-p** (nucleus) sampling. This document analyzes whether these can be
fused into the kernel.

### The matmul is memory-bound for decode workloads

The FMMS kernel targets LLM decode, where the batch dimension H (number of
hidden states) is small. The matmul `W[V,D] × H[D,H]` has arithmetic
intensity:

```
FLOPs / bytes ≈ (2·V·D·H) / (2·V·D) = H
```

At H=1 (single-request decode), arithmetic intensity is **1** — pure
memory-bound. At H=64, it's 64 — still well below the H100's
compute-to-bandwidth ratio (~330 FLOPs/byte for BF16 tensor cores). The matmul
only becomes compute-bound around H≈128+.

**Consequence:** for decode workloads, the kernel's throughput is limited by
how fast it reads the weight matrix from HBM. Every byte loaded should do as
much useful work as possible — matmul, top-k selection, top-p filtering, and
sampling — in a single pass. Any extra kernel launch that re-reads data from
HBM wastes bandwidth on what is already the bottleneck.

## Sequential top-k-then-top-p in practice

vLLM, FlashInfer, and Qrita all default to sequential top-k-then-top-p (top-k first, then top-p on survivors). Key code locations:

- **vLLM v0.16.0** PyTorch path: [`topk_topp_sampler.py#L264`](https://github.com/vllm-project/vllm/blob/v0.16.0/vllm/v1/sample/ops/topk_topp_sampler.py#L264) (sort once, mask by k, then softmax+cumsum for top-p)
- **vLLM v0.16.1rc0** Triton kernel: [`topk_topp_triton.py`](https://github.com/vllm-project/vllm/blob/v0.16.1rc0/vllm/v1/sample/ops/topk_topp_triton.py) (PR [#33538](https://github.com/vllm-project/vllm/pull/33538)); this is the code `qitra.py` in this repo is based on.
- **FlashInfer v0.6.3**: [`sampling.py#L1127`](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.3/flashinfer/sampling.py#L1127) (`"top_k_first"` branch, the default)
- **SGLang** uses FlashInfer's `"joint"` mode instead (both constraints checked simultaneously).

## FlashInfer's rejection sampling

FlashInfer ([blog post](https://flashinfer.ai/2025/03/10/sampling.html))
replaces sorting-based top-k/top-p with **rejection sampling**:

1. Sample a token via inverse transform sampling (CDF without sorting).
2. Check if it satisfies the top-k or top-p constraint.
3. If not, update a pivot threshold and resample.

A dual-pivot variant guarantees convergence in O(log(1/ε)) rounds. This avoids
full sorting and multiple kernel launches.

**Key limitation for fusion:** each rejection round scans the full probability
distribution. In a fused kernel that would mean re-running the matmul per
round — a non-starter. FlashInfer's approach operates on pre-computed logits.

### `filter_apply_order`: sequential vs joint

`top_k_top_p_sampling_from_logits` in
`.venv/.../flashinfer/sampling.py` (lines 1127–1154) supports two strategies
via the `filter_apply_order` parameter:

**`"top_k_first"` (default):** sequential — three kernel launches.

```python
masked_logits = top_k_mask_logits(logits, top_k)   # 1. mask all but top-k
probs = torch.softmax(masked_logits, dim=-1)        # 2. softmax on survivors
return top_p_sampling_from_probs(probs, top_p, ...)  # 3. rejection-sample with top-p
```

Top-p is evaluated on the **renormalized** distribution after top-k filtering.

**`"joint"`:** single fused CUDA kernel (`TopKTopPSamplingFromProbKernel` in
`sampling.cuh`, ~line 1199). Uses dual-pivot binary search rejection sampling
and accepts a candidate only when **both** constraints are satisfied
simultaneously on the **original** distribution:

```cpp
if (aggregate_gt_pivot_0.count < k && aggregate_gt_pivot_0.value < p) {
    break;  // accept: both top-k and top-p satisfied jointly
}
```

### Sequential vs joint: when results differ

The two strategies can produce different candidate sets because renormalization
after top-k inflates surviving tokens' probabilities.

**Example:** 5 tokens, top_k=3, top_p=0.35

| Token | Original prob |
|-------|---------------|
| A     | 0.30          |
| B     | 0.25          |
| C     | 0.24          |
| D     | 0.11          |
| E     | 0.10          |

**Sequential** (top-k first):

1. Top-k=3: keep {A=0.30, B=0.25, C=0.24}, discard D, E.
2. Renormalize: A=0.380, B=0.316, C=0.304.
3. Top-p=0.35: cumsum A=0.380 ≥ 0.35 → only **{A}** survives.
   Result: deterministic.

**Joint** (both at once on original probs):

- Top-p=0.35: cumsum A=0.30 < 0.35, A+B=0.55 ≥ 0.35 → **{A, B}**.
- Top-k=3: {A, B, C}.
- Intersection: **{A, B}**.
  Result: still stochastic.

Renormalization inflated A from 0.30 to 0.38, crossing the 0.35 threshold
alone. On the original distribution A doesn't reach 0.35, so B is also needed.
In general, **sequential can produce a smaller candidate set** than joint.

## Top-k: feasible via tile-local top-k + merge

Top-k restricts sampling to the k tokens with the highest logits. The key
observation: the tile-local max can be generalized to a **tile-local top-k**.

### Single-pass algorithm

1. **Stage 1 (fused with matmul):** Each V-tile computes logits as now, but
   outputs its **local top-k** values + indices instead of just the top-1.
   Finding top-k within a tile of `BLOCK_SIZE_V` (128–256) elements for small
   k is cheap — partial sort within registers, or `tl.sort` on the tile
   followed by a slice.

2. **Stage 2 (merge):** Merge all per-tile top-k lists into a global top-k.
   This reduces `num_tiles × k` candidates — still tiny compared to the full
   vocab.

3. **Stage 3 (sample):** Apply Gumbel noise to the k surviving candidates and
   take argmax. Trivially cheap.

### Cost analysis

- **Intermediate storage:** Grows from `num_tiles × H` to
  `num_tiles × k × H`. For V=128K, BLOCK_SIZE_V=256, k=50, H=1: 512 tiles ×
  50 = 25,600 elements. Negligible vs materializing 128K logits.
- **Matmul cost:** Unchanged (dominant cost).
- **Within-tile partial sort:** O(BLOCK_SIZE_V × k) comparisons in registers.
  Small relative to the matmul.

### Engineering challenges

- **No built-in top-k in Triton.** Options:
  - `tl.sort` on the tile logits, then take the first k values. `tl.sort`
    exists in Triton and works on tile-sized data (bitonic sort, O(n log² n)).
  - Manual insertion sort maintaining k running maxima. Better for small k
    but awkward to express in Triton.
- **Stage 2 merge.** The current stage 2 is a single `max(axis=0)` in PyTorch.
  For top-k, it becomes a merge of sorted lists — still simple but requires
  a custom reduction (sort `num_tiles × k` candidates, take top k, then
  Gumbel-sample).

## Top-p: not directly fusible

Top-p (nucleus sampling) keeps the smallest set of tokens whose cumulative
probability mass exceeds p. This requires three global operations:

1. **Global softmax** — needs the max logit across ALL tiles for numerical
   stability. Requires a full pass before any probabilities can be computed.
2. **Global sorting** — must rank tokens by probability across the full vocab.
3. **Cumulative sum** — accumulate mass from highest to lowest until exceeding
   p.

None of these decompose into independent tile-local work. Fusion would require
multiple passes over the weight matrix (or materializing logits), defeating the
purpose.

## Min-p: partially feasible but needs global max

Min-p filters tokens whose probability is below `min_p × max_probability`.
This requires knowing the global max probability, which in turn requires:

1. A full pass to find the max logit (for softmax stability).
2. Computing `softmax(max_logit)` to get the reference probability.
3. A second pass to filter tokens below the threshold.

The need for two passes makes direct fusion impractical, though the approach is
simpler than top-p since no sorting or cumulative sum is needed.

## Hybrid approach: fused top-k + post-kernel top-p

The most practical path for supporting both:

1. **Fuse top-k into the kernel** (as described above) with a conservatively
   large k (e.g. k=256).
2. **Apply top-p on the k survivors** outside the kernel. On 256 elements,
   softmax + sort + cumsum + resample is trivially fast.

This mirrors how vLLM combines the two filters. vLLM's FlashInfer path
(`vllm/v1/sample/ops/topk_topp_sampler.py`, line 377) calls
`flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, k, p)` without
passing `filter_apply_order=`, so it uses the default `"top_k_first"`
(sequential). vLLM's native PyTorch path (`apply_top_k_top_p()`, lines
243–283) also applies top-k first: sort once, mask by k-th value, then
softmax + cumsum for top-p.

The fusion benefit is that the full logit matrix is still never materialized;
only `num_tiles × k × H` intermediate values are stored.

## Implementation language: Triton vs CUDA C++

### What CUDA C++ enables

The top-k/top-p parts of the kernel need primitives that Triton doesn't
expose:

- **`BlockRadixSort`** (CUB/CCCL) — partial or full radix sort within a
  thread block. O(n) for fixed-width keys. This is what FlashInfer uses
  internally.
- **`BlockScan`** — prefix sum (cumulative sum) for top-p thresholding.
- **`BlockReduce`** — reductions with custom operators.
- **Shared memory control** — maintain a min-heap of size k across tile
  iterations. In Triton, shared memory is implicit and can't persist
  structured data across loop iterations the same way.
- **Warp-level primitives** — `__shfl_sync` for small-k merge steps.

In Triton, `tl.sort` is bitonic sort (full sort, O(n log²n)). There is no
partial sort, no selection, and no built-in scan/prefix-sum.

### CUTLASS epilogue approach

The architecturally cleanest CUDA path uses CUTLASS for the matmul with a
**custom epilogue** that performs top-k selection on each output tile while
it's still in registers:

```
┌─────────────────────────────────────┐
│  CUTLASS GEMM: W[V_tile, D] × H^T  │
│           ↓ (tile in registers)     │
│  Epilogue: scale by 1/temperature   │
│           ↓                         │
│  Epilogue: BlockRadixSort, keep     │
│            top-k values + indices   │
│           ↓                         │
│  Write k candidates to global mem   │
└─────────────────────────────────────┘
         ↓ (num_tiles × k × H)
┌─────────────────────────────────────┐
│  Kernel 2: merge top-k lists        │
│  → softmax on k candidates          │
│  → BlockScan for cumsum (top-p)     │
│  → sample from survivors            │
└─────────────────────────────────────┘
```

- Kernel 1: standard CUTLASS GEMM + custom epilogue. The matmul stays at peak
  efficiency; the epilogue does local top-k on tile data already in registers.
- Kernel 2: operates on only `num_tiles × k` elements per sequence — trivially
  fast. CUB's `BlockScan` handles the cumsum for top-p natively.
- Full logit matrix is never materialized.

### Tradeoffs

| Aspect             | Triton                       | CUDA C++ (CUTLASS)              |
|--------------------|------------------------------|---------------------------------|
| Matmul quality     | Good (auto tensor cores)     | Excellent (CUTLASS)             |
| Top-k in tile      | `tl.sort` full bitonic sort  | `BlockRadixSort` partial sort   |
| Top-p cumsum       | No primitive, manual         | `BlockScan`                     |
| Development time   | Days                         | Weeks to months                 |
| Debugging          | Python-level, print-friendly | NSight, printf, painful         |
| Portability        | AMD + NVIDIA                 | NVIDIA only (CUTLASS)           |
| Maintainability    | ~100 lines                   | ~500+ lines, templates          |
| Autotuning         | `@triton.autotune` built-in  | Manual or CuTe tuning           |

### Pragmatic middle ground

Keep the Triton kernel for the matmul, extend it to output local top-k per
tile (using `tl.sort` + slice — not optimal but functional), and write a small
CUDA C++ kernel only for the merge + top-p + sample step where CUB primitives
shine. This gets ~90% of the benefit without rewriting the matmul.

## Quack / CuTe-DSL: hierarchical reductions for memory-bound decode

Reference: [Getting Memory-bound Kernels to Speed-of-Light](https://github.com/Dao-AILab/quack/blob/main/media/2025-07-10-membound-sol.md)
(Wentao Guo, Ted Zadouri, Tri Dao)

### Why this matters for FMMS

The FMMS matmul is **memory-bound** at decode batch sizes. Since the kernel
spends most of its time reading weights from HBM, every byte loaded should do
as much work as possible in a single pass — matmul, top-k, top-p, sampling.
Any extra kernel launch that re-reads data wastes bandwidth on the bottleneck.

Quack demonstrates achieving 90% of H100 peak HBM bandwidth (3.01 out of
3.35 TB/s) by reducing at every level of the memory hierarchy before touching
the next slower level:

```
Thread registers  (>100 TB/s, ~few ns)
  → Warp shuffle  (~10s ns)
    → Block SMEM  (20-30 TB/s, ~10-20 ns)
      → Cluster DSMEM  (5-10 TB/s, ~150-200 ns)  ← Hopper sm_90+
        → Grid HBM  (3.35 TB/s, ~400 ns)         ← last resort
```

Their softmax kernel loads data **once** from HBM and does all reductions
through this hierarchy. Torch.compile's Triton kernel loads **twice** (once for
max, once for softmax) and achieves only ~2 TB/s — a 50% throughput gap.

### How this applies: single-pass fused kernel

The current FMMS stage 2 writes per-tile maxes to HBM and reads them back in a
separate PyTorch reduction — an HBM round-trip. For the top-k extension, this
gets worse: each tile writes k values instead of 1.

With Hopper thread block clusters, the entire pipeline can stay on-chip:

```
┌─────────────────────────────────────────────────────────┐
│  Per V-tile (in registers):                             │
│    1. Load weight tile from HBM (only HBM read)        │
│    2. Matmul: W_tile × H^T → logits_tile               │
│    3. Temperature scale                                 │
│    4. Local top-k selection (sort/select in registers)  │
│                                                         │
│  Warp reduction:                                        │
│    5. Shuffle-merge local top-k across warp lanes       │
│                                                         │
│  Block reduction (SMEM):                                │
│    6. Merge warp top-k lists into block top-k           │
│                                                         │
│  Cluster reduction (DSMEM, Hopper only):                │
│    7. Merge block top-k lists across cluster via DSMEM  │
│    8. Softmax on merged top-k candidates (small)        │
│    9. BlockScan cumsum for top-p threshold              │
│   10. Sample from survivors                             │
│                                                         │
│  Write to HBM: one token index per sequence             │
└─────────────────────────────────────────────────────────┘
```

A cluster of 8 blocks (8 V-tiles) merges via DSMEM at 5-10 TB/s instead of
HBM at 3.35 TB/s. The final cross-cluster merge involves only
`num_clusters × k` elements — tiny. For V=128K with BLOCK_SIZE_V=256 and
cluster_size=8, there are 512/8 = 64 clusters; with k=50, the final merge is
64 × 50 = 3,200 elements.

### Top-p becomes feasible within a cluster

Top-p requires a global softmax normalization constant. With cluster DSMEM,
this can happen without HBM:

1. Each tile computes local top-k candidates + local max logit during matmul.
2. **Cluster DSMEM sync:** share max logits across the cluster, compute
   cluster-wide max.
3. Each tile computes `exp(logit - max)` and local sum for its candidates.
4. **Cluster DSMEM sync:** merge sums, compute cumsum for top-p.
5. Apply top-p threshold and sample.

Both "passes" happen in SMEM/DSMEM — HBM is touched only once (weight loads).

### CuTe-DSL as implementation path

Quack uses **CuTe-DSL** (Python), not raw CUDA C++ or CUTLASS templates. This
is more accessible than the CUTLASS epilogue approach — you get explicit
control over the memory hierarchy (vectorized 128-bit loads, shuffle
reductions, DSMEM barriers) while writing Python. The TV-layout system handles
memory coalescing automatically.

This could be a better development path than either Triton (which cannot
express clusters or DSMEM) or CUTLASS C++ (template-heavy, hard to debug).

### Limitations

- **Hopper only.** Thread block clusters and DSMEM require sm_90+ (H100, H200).
  The current codebase targets RTX 3090 (sm_86). A Triton fallback would still
  be needed for pre-Hopper GPUs.
- **CuTe-DSL maturity.** CuTe-DSL is newer than Triton and CUTLASS. Tooling
  (profiling, debugging) is less established.
- **Matmul in CuTe.** Writing an efficient thin-matmul (GEMV) in CuTe-DSL is
  more manual than Triton's `tl.dot`, though CuTe provides the building blocks
  (MMA atoms, tiled copies).

## Summary

| Strategy       | Fusible?  | Difficulty | Notes                                             |
|----------------|-----------|------------|---------------------------------------------------|
| **Top-k**      | Yes       | Medium     | Local top-k per tile + merge. `tl.sort` on tiles. |
| **Top-p**      | No        | —          | Requires global softmax + sorted cumsum.          |
| **Top-k+top-p**| Partially | Medium     | Fuse top-k, apply top-p on k survivors post-kernel.|
| **Min-p**      | No        | —          | Needs global max logit as threshold reference.    |

| Language         | Best for                               | Limitation                      |
|------------------|----------------------------------------|---------------------------------|
| **Triton**       | Matmul + local top-k (tile sort)       | No partial sort, no scan, no clusters |
| **CUDA/CUTLASS** | Full pipeline (matmul + top-k + top-p) | High dev cost, NVIDIA only      |
| **CuTe-DSL**     | Full pipeline with cluster reductions  | Hopper only, less mature        |
| **Hybrid**       | Triton matmul + CUDA top-p kernel      | Best effort/benefit ratio       |

The FMMS matmul is **memory-bound** for LLM decode workloads (H ≤ ~128). The
dominant cost is reading the weight matrix from HBM. Fusing top-k/top-p into
the kernel avoids extra HBM traffic for intermediate logits, and hierarchical
reductions (especially Hopper DSMEM clusters) can keep the merge/sampling
stages entirely on-chip.

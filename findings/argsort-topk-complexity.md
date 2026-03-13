# Argsort vs top-k complexity in the fused top-k kernel

## Why we use a custom argsort

The fused top-k kernel (`tl_fused_mm_topk.py`) needs both sorted values and their
original vocab indices to know which token each logit belongs to.
Triton's `tl.sort` returns sorted values only, no indices.
There is no `tl.argsort` in the standard library.

The custom `tl_argsort.py` is adapted from a community implementation posted in
[triton-lang/triton#3698](https://github.com/triton-lang/triton/issues/3698)
(by hgl71964).
It uses the same bitonic sort / hypercube-reshape approach as `tl.sort`, but
carries an `ids` tensor alongside the values through every compare-and-swap step.
A subtle correctness fix prevents id duplication on tied values: ids are only
swapped when values actually differ (`swap_ids = cond & (x != y)`).

### Could we use `tl.topk` + gather instead?

`tl.topk` exists (added ~Feb 2026) but returns values only, no indices.
There is no way to recover the value-to-index mapping after the sort without
carrying indices through the sort network.
The Triton maintainers' stance is that argsort/topk-with-indices are user-space
concerns, not planned for the standard library:

- [#3698](https://github.com/triton-lang/triton/issues/3698) (open): argsort request.
  Maintainer: "tl.sort is part of the standard library, argsort can be implemented on top."
- [#4012](https://github.com/triton-lang/triton/issues/4012) (open): topk request.
  A PR ([#5706](https://github.com/triton-lang/triton/pull/5706)) that returned
  values+indices was closed. Maintainer suggested "`tl.sort` + `tl.gather`" instead.
- [#9404](https://github.com/triton-lang/triton/issues/9404) (closed): argsort
  primitive request. Closed immediately with the same rationale.

## Full sort vs top-k: complexity analysis

Both `tl.sort` and `tl.topk` use bitonic sort on a hypercube of shape `[2]*n_dims`.

**Full sort** runs `log_n` stages. Stage `i` has `i` compare-and-swap (CAS) rounds
on all `n` elements.

**Top-k** (from [Massively Parallel Top-K](https://www.doc.ic.ac.uk/~hlgr/pdfs/MassivelyParallelTopK.pdf))
has two phases:

1. **Phase 1**: `log_k` bitonic sort stages on all `n` elements (same as sort, fewer stages).
2. **Phase 2**: `log_n - log_k` iterations, each doing one `max` reduction
   (halves the tensor) followed by `log_k` CAS rounds on the smaller tensor.

**Important**: `k` must be a power of 2. `_log2(k)` truncates, so `k=20` yields
`log_k=4`, returning only 16 elements. To cover top-20, you need `k=32`.

### Sequential depth (latency bound)

All comparisons within a round are parallel, so the number of sequential rounds
determines latency.

| | Rounds formula |
|---|---|
| Full sort | `log_n * (log_n + 1) / 2` |
| Top-k phase 1 | `log_k * (log_k + 1) / 2` |
| Top-k phase 2 | `(log_n - log_k) * (1 + log_k)` |

### Total work (comparisons)

Phase 2 operates on geometrically shrinking tensors.
Each iteration `j` does `n/2^(j+1)` max comparisons plus `log_k * n/2^(j+2)` CAS
comparisons.
The geometric sum gives phase 2 total work of approximately `n * (2 + log_k) / 2`.

### Concrete numbers for our use case

**BLOCK_SIZE_V=128** (`log_n=7`), full sort = **28 rounds**, **14n comparisons**.
User-facing `top_k=20` requires effective `k=32` (`log_k=5`):

| k (effective) | log_k | Rounds (depth) | % of sort | Work | % of sort |
|---|---|---|---|---|---|
| 4 | 2 | 18 | 64% | 3.1n | 22% |
| 8 | 3 | 22 | 79% | 4.9n | 35% |
| 16 | 4 | 25 | 89% | 7.6n | 54% |
| **32 (for top-20)** | **5** | **27** | **96%** | **10.1n** | **72%** |
| 64 | 6 | 28 | 100% | 13.0n | 93% |

**BLOCK_SIZE_V=256** (`log_n=8`), full sort = **36 rounds**, **18n comparisons**:

| k (effective) | log_k | Rounds (depth) | % of sort | Work | % of sort |
|---|---|---|---|---|---|
| 4 | 2 | 21 | 58% | 3.5n | 19% |
| 16 | 4 | 30 | 83% | 7.8n | 43% |
| **32 (for top-20)** | **5** | **33** | **92%** | **10.6n** | **59%** |
| 64 | 6 | 35 | 97% | 13.5n | 75% |

### Takeaway

At `k=32` (needed for `top_k=20`) and `BLOCK_SIZE_V=128`, top-k saves **one
sequential round** compared to full sort, a 4% latency reduction.
Total work drops by 28%, but within a single thread block the sequential depth
(latency) dominates, not total comparisons.

The ratio `k/n = 32/128 = 25%` is too large for the bitonic top-k selection to
provide meaningful speedup. The algorithm shines when k is tiny relative to n
(e.g. top-4 out of 1024).

Building an `argtopk` from `tl.topk`'s internals would be a code cleanliness
improvement (using upstream's algorithm rather than a vendored argsort), but it
would not fix the 14ms regression at bsz=256 seen in `preliminary-topk-top-benchs.md`.
That bottleneck is the sort itself being O(n log^2 n) work at large batch sizes,
regardless of whether it is a full sort or top-k.

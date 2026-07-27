# DSMEM cluster reduction for FMMS

Research note on whether Hopper and Blackwell thread block clusters with
Distributed Shared Memory (DSMEM) can improve a CUTLASS FMMS kernel.

The conclusion changed after checking the proposed GEMM mapping against the
CUTLASS visitor contracts and CTA tiling model.

## Verdict

DSMEM is a plausible optional optimization for **candidate compression
across adjacent vocabulary tiles**.
It is not an architectural fix for the current Triton kernel's H=256
register spill.

The baseline CUTLASS kernel should use:

- GEMM `W[V,D] @ hidden_states.T[D,H]`, so V is M and H is N.
- A custom M-axis max-with-index visitor derived from `Sm90RowReduction`.
- One `(value,index)` candidate per M tile and N column.
- The existing GPU Stage 2 to merge candidates across M tiles.

Only after that baseline is correct and profiled should a cluster along M be
tested.
The cluster can merge `cluster_m` candidates into one before HBM and can
potentially multicast the shared hidden-state N tile.

No speedup is assumed.

## Why the earlier register-spill argument was wrong

The current persistent Triton kernel uses a runtime grid over M and N but
each program iterates over multiple logical tiles.
At H=256 it uses `BLOCK_SIZE_H=64`, and three `[128,64]` FP32 tensors are
live simultaneously.
That kernel spills 118 MB in the measured configuration.

A conventional CUTLASS GEMM already tiles the output into fixed
`[tile_M,tile_N]` CTA tiles.
With `tile_N=64`, increasing global H from 64 to 256 creates more N tiles.
It does not make an individual CTA hold 256 columns.

Clustering CTAs does not change this:

- A cluster along M gives each CTA a different vocabulary tile and the same
  hidden-state N tile.
  Those CTAs can merge candidates, but each still owns a full fixed-size
  accumulator tile.
- A cluster along N gives each CTA different hidden-state columns.
  Those outputs correspond to different samples and cannot be reduced
  together.

Therefore a cluster cannot turn a CUTLASS `[128,64]` accumulator into
`[128,8]` merely because `cluster_m=8`.
Quack's published cluster recovery applies when clustering divides a large
reduction domain that one CTA would otherwise retain.
That mechanism is not present in this CUTLASS tiling.

The current Triton spill remains relevant as motivation to measure CUTLASS
register use, but it cannot be projected onto the new kernel.

## Correct reduction axis

The output is `[V,H]`.
In CUTLASS GEMM coordinates:

```text
M = V
N = H
K = D
```

Sampling reduces vocabulary, so it reduces M and emits one result per N.

CUTLASS example 61 uses `Sm90TopKSoftmaxColReduction`.
Its source states that fusion is over N and enforces:

```cpp
return N <= tile_N && N <= epi_N && N >= TopK;
```

It cannot be used as the FMMS reduction skeleton with the performant GEMM
orientation.
Putting V on N would make the constraint fail for V=128K-152K and repeats
the operand-swap direction already shown to regress in
`2cta-mma-operand-swap-regression.md`.

CUTLASS also ships `Sm90RowReduction`.
Despite the name, its implementation reduces M and emits results along N.
It supports:

- Fragment-local register reduction.
- Shuffle reduction across M lanes.
- Shared-memory reduction across multiple M warps.
- A non-final per-CTA output.
- An optional workspace and tile-counter path for cross-CTA final reduction.

The FMMS visitor should derive its tensor layouts and reduction choreography
from `Sm90RowReduction`.
It must replace the scalar state with `(noisy_logit, global_vocab_index)` and
use max-with-index as its reduction operation.

Example 61 remains useful only for its sorted-array insertion and K=2/K=4
PTX merge helpers.

## What DSMEM can do

### Candidate compression

Use a cluster shape:

```text
(cluster_m, 1, 1)
```

All CTAs in the cluster cover adjacent M tiles and the same N tile.
Each CTA first computes one argmax or top-k candidate per N column.
The CTAs exchange those small candidates through DSMEM.
One CTA writes the cluster result to HBM.

For argmax, this changes the candidate shape from approximately:

```text
[ceil(V / tile_M), H]
```

to:

```text
[ceil(V / (tile_M * cluster_m)), H]
```

The global Stage 2 is still required because the portable cluster size is at
most eight CTAs and a full vocabulary contains many more M tiles.

For top-k, the cluster must merge `cluster_m * k` value-index pairs into k
pairs.
A bitonic merge is a reasonable small-k prototype.
The FlashInfer DSMEM histogram approach is a separate option for larger k.

### Hidden-state multicast

CTAs clustered along M read different weight rows but share the same
hidden-state N tile.
TMA multicast may reduce repeated hidden-state loads.

This is expected to be secondary because the weight matrix dominates HBM
traffic.
It needs direct traffic and latency measurements.

### Tensor-parallel candidate traffic

Candidate compression inside each GPU reduces the number of candidate pairs
written to peer symmetric-memory buffers by `cluster_m`.
The inter-rank mechanism remains NVLink plus symmetric memory.
DSMEM is strictly intra-GPU and cannot replace it.

The reduced payload may or may not matter because candidate traffic is
already small.
Measure the complete Stage 1, P2P, barrier, and Stage 2 path.

## What DSMEM cannot do

- It cannot reduce weight-matrix HBM traffic.
- It cannot reduce the fixed CUTLASS CTA accumulator shape.
- It cannot be assumed to remove the current Triton kernel's spill.
- It cannot merge CTAs along N because those columns represent different
  hidden states.
- It cannot replace inter-GPU NVLink communication.
- It cannot eliminate global Stage 2 unless one cluster covers the entire
  vocabulary.

## Prototype sequence

1. Complete the greedy, Gumbel-Max, and TP gates without DSMEM.
2. Record candidate writes, registers per thread, local-memory traffic,
   occupancy, Stage-1 latency, Stage-2 latency, TP exchange, and total
   latency.
3. Proceed only if a predeclared 3% total improvement is plausible.
4. Build a standalone DSMEM max-with-index reduction for cluster sizes
   2, 4, and 8.
5. Integrate it after the CTA-local M reduction.
6. Compare cluster sizes 1, 2, 4, and 8 with paired end-to-end runs.
7. Keep clustering only if total latency improves by the predeclared
   threshold.
8. Test hidden-state multicast separately so its effect is not conflated
   with candidate compression.

Correctness validation must include:

- Exact greedy max-with-index results.
- Chi-squared sampling tests at large vocabulary.
- Boundary vocabulary sizes with partially out-of-bounds M tiles.
- Independent Philox sequences for every global output element.
- TP2/4/8 distributed correctness after symmetric-memory integration.

## Risks

1. No shipped CUTLASS EVT visitor performs this DSMEM exchange.
2. The cluster barrier must compose correctly with the warp-specialized GEMM
   and epilogue pipelines.
3. A cluster can reduce occupancy and scheduling flexibility.
4. Candidate traffic may be too small for compression to repay the cluster
   synchronization cost.
5. Top-k DSMEM exchange grows with k and may require a different algorithm
   from argmax.
6. RNG indexing must be based on global `(sample, vocab)` coordinates, not
   launch order, tile scheduler order, or cluster rank alone.

## Primary sources

- NVIDIA CUTLASS
  `include/cutlass/epilogue/fusion/sm90_visitor_topk_softmax.hpp`.
- NVIDIA CUTLASS
  `include/cutlass/epilogue/fusion/sm90_visitor_store_tma_warpspecialized.hpp`.
- NVIDIA CUTLASS example 61, Hopper GEMM with top-k and softmax.
- NVIDIA CUDA Programming Guide, thread block clusters and Distributed
  Shared Memory.
- Dao-AILab Quack `quack/reduce.py`.
- FlashInfer `fast_topk_clusters_exact`.

## Related project findings

- `cutlass-fmms-kernel-plan.md`.
- `cutlass-61-topk-softmax-epilogue.md`.
- `register-spilling-bsz256.md`.
- `2cta-mma-operand-swap-regression.md`.
- `tp2-collective-overhead.md`.

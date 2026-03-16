# GEMV kernel for bsz=1

Investigated whether a GEMV-specialized FMMS kernel (matrix-vector product instead of matrix-matrix) improves performance at batch size 1.

## Motivation

The standard FMMS kernel uses `tl.dot` with `BLOCK_SIZE_H=16` (the minimum for tensor cores). At bsz=1, 15 out of 16 output columns are wasted. The GEMV kernel eliminates this by using element-wise multiply + reduce (`tl.sum(w_blk * h_blk[None, :], axis=1)`) instead of `tl.dot`, producing a 1D logits vector `[BLOCK_SIZE_V]` instead of a 2D matrix `[BLOCK_SIZE_V, 16]`.

## Implementation

Two versions were implemented and tested:

1. **Approach 1 (bsz=1 only)**: 1D grid over V tiles. Each program loads one hidden state vector and computes a GEMV per tile. Second stage reduction is the same as the standard FMMS.

2. **Approach 2 (general bsz)**: 2D grid `(V_tiles, H)`. Each program handles one (V-tile, hidden-state) pair, with `tl.swizzle2d` for L2 cache reuse. Each program independently loads the W tile and its assigned hidden state vector.

Both versions pass the chi-squared sampling distribution test at vocab sizes 100, 200, 256 and n_hidden_states 1, 2.

## Results (RTX 3090, large case: V=128,256, D=8,192)

### H=1: GEMV vs GEMM

| Kernel              | Best config           | Median (ms) |
| ------------------- | --------------------- | ----------- |
| GEMM (fused-triton) | V=128, D=64, warps=8  | 2.374       |
| GEMV                | V=128, D=128, warps=4 | 2.311       |

~2.6% improvement. The GEMV kernel picks `BLOCK_SIZE_D=128` because the freed SHMEM (no `[BLOCK_SIZE_V, 16]` logits matrix, no `[16, BLOCK_SIZE_D]` hidden states tile) allows larger D tiles. The GEMM kernel is stuck on `BLOCK_SIZE_D=64` (3 of 4 autotune configs OOM on shared memory at D=8192).

### H > 1: GEMV scales linearly, GEMM stays flat

| H   | GEMM (ms) | GEMV (ms) | GEMV / GEMM  |
| --- | --------- | --------- | ------------ |
| 1   | 2.374     | 2.380     | 1.00x        |
| 2   | 2.375     | 4.656     | 1.96x slower |
| 4   | 2.376     | 9.263     | 3.90x slower |

Each GEMV program independently loads the full W tile for its hidden state. At H=2, the entire weight matrix is loaded twice from GMEM. The GEMM kernel handles H=1..16 in a single padded `tl.dot` with no extra W loads.

### Larger tile sizes don't help

Expanded the autotune search to `BLOCK_SIZE_V` in {128, 256, 512, 1024} and `BLOCK_SIZE_D` in {64, 128, 256}. All configs compiled (none OOM'd on SHMEM), but the autotuner still picked V=128, D=128. Larger V tiles increase register pressure without improving bandwidth utilization. The kernel is purely memory-bandwidth bound on W loads at bsz=1.

## Why the improvement is small

Both kernels load the same data from GMEM. The GEMM kernel's masked hidden state loads (`mask_h` is True for 1 of 16 rows) don't generate extra memory traffic: NVIDIA predicated loads suppress requests for masked-off rows.

The difference is in resource waste, not bandwidth:
- GEMM allocates `[BLOCK_SIZE_V, 16]` for logits (2048 float32 values) vs GEMV's `[BLOCK_SIZE_V]` (128 values)
- GEMM computes 15 dead `tl.dot` output columns
- GEMM allocates `[1, BLOCK_SIZE_V, 16]` for Gumbel noise vs `[1, BLOCK_SIZE_V]`

This SHMEM bloat limits GEMM to smaller D tiles, but the performance impact is marginal because W loads (~2 GB for V=128K, D=8192) dominate the runtime.

## Conclusion

The GEMV specialization yields a ~2.6% improvement at bsz=1, which is not worth the code complexity. The standard FMMS kernel with tensor core padding handles bsz=1 nearly as well. The GEMV approach degrades linearly for bsz > 1, making it unsuitable as a general replacement.

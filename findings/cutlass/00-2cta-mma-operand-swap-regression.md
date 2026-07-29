# 2-CTA MMA: operand swap causes low-H regression

Branch: `worktree-2cta-mma`. Conclusion: **abandoned**. The operand transpose required to enable Blackwell's `tcgen05.mma cta_group::2` introduces a 6-23% regression at H≤32 that any plausible 2-CTA gain at H≥128 cannot recover.

## Motivation

Blackwell's `tcgen05.mma` supports `cta_group::2`, where two CTAs in a cluster collaborate on one MMA. The accumulator is split along M, A is split along M (each CTA owns its own M-slice), and B is held by the leader CTA only. This halves the per-FLOP DRAM bandwidth on B for the cluster.

In the FMMS kernel `weights[V_tile, D] @ hidden[H_tile, D].T`, the bandwidth bottleneck is `weights` (~2 GB at V=152k, D=8k bf16). With the original `tl.dot(weights_tile, hidden_tile.T)`, weights is the A operand, so 2-CTA MMA would only halve the small `hidden_tile` traffic - useless.

Swapping operands to `tl.dot(hidden_tile, weights_tile.T)` (so weights is B and gets multicast) was the only way to put the bandwidth saving on the right operand.

## Direct `num_ctas=2` was a non-starter

Adding `num_ctas=2` to the autotune config without reshaping the kernel aborts at compile time:

```
ReduceOpToLLVM.cpp:31: Assertion `helper.isReduceWithinCTA() && "Unexpected srcLayout in ReduceOpConversion"' failed.
```

The compiler picks `CTAsPerCGA = [2, 1]`, splitting the V (M) axis across the cluster. Our `tl.argmax` over V then crosses CTAs, and Triton's reduce-op lowering only supports within-CTA reductions. Crash kills the whole autotune job (uncaught `RuntimeError`), not just that one config. See `benchmarking/modal-results.bak.../b200-2ctas/tp1/logs.txt` for the original repro.

This is what motivated the operand swap: with hidden as A, the V dimension becomes N (within one CTA), and the reduce stays local.

## What I changed

In the persistent inner loop:

- `logits_blk` shape `[V, H]` → `[H, V]`
- `tl.dot(weights_blk, hidden_blk.T)` → `tl.dot(hidden_blk, weights_blk.T)`
- `tl.where(mask_v[:, None], ...)` → `tl.where(mask_v[None, :], ...)`
- `tl.max(..., axis=0, ...)` → `tl.max(..., axis=1, ...)`
- `noise_offsets` reshape and `RETURN_LOGITS` pointer broadcast updated to match
- Output buffer layout (`max_out_ptr`, `max_out_idx_ptr`, `logits_out_ptr`) unchanged - the transpose is internal to the kernel only

`num_ctas=2` was **not** added to the autotune sweep yet; the regression below was measured with the operand swap alone.

## Correctness

All TP1 tests pass on RTX 3090 (sampling distribution chi-squared, RETURN_LOGITS, greedy, bsz_h: 25 tests total). All TP2 tests pass on B200 via Modal (fused-triton + naive-pt + naive-compiled + flashinfer + greedy across V∈{100,200,256} × H∈{1,2}: 30 cases). The transpose is mathematically `(W H^T)^T = H W^T`, correctness preserved.

## Performance regression (B200 TP1, fi-cupti, FMMS Triton, ms)

3 baseline runs (untransposed, main HEAD) × 3 transposed runs. Δσ = `(transposed_mean − baseline_mean) / baseline_std`.

**Large case (V=128k, D=8k):**

| H   | baseline mean | transposed mean | Δ%      | Δσ    |
|-----|---------------|-----------------|---------|-------|
| 1   | 0.3292        | 0.3380          | +2.7%   | 2.3   |
| 2   | 0.3329        | 0.3431          | +3.1%   | 2.4   |
| 4   | 0.3339        | 0.3434          | +2.9%   | 2.7   |
| 8   | 0.3345        | 0.3434          | +2.7%   | 2.6   |
| 16  | 0.3351        | 0.3443          | +2.7%   | 2.8   |
| **32**  | **0.3367**    | **0.3802**          | **+12.9%**  | **11.5**  |
| 64  | 0.3450        | 0.3477          | +0.8%   | 0.6   |
| 128 | 0.4347        | 0.4556          | +4.8%   | 1.5   |
| 256 | 0.7662        | 0.7989          | +4.3%   | 0.7   |

**Small case (V=152k, D=4k):**

| H   | baseline mean | transposed mean | Δ%      | Δσ    |
|-----|---------------|-----------------|---------|-------|
| 1   | 0.2104        | 0.2247          | +6.8%   | 10.2  |
| 2   | 0.2163        | 0.2307          | +6.7%   | 9.1   |
| 4   | 0.2164        | 0.2304          | +6.5%   | 7.9   |
| 8   | 0.2167        | 0.2305          | +6.4%   | 6.8   |
| 16  | 0.2170        | 0.2315          | +6.7%   | 8.6   |
| **32**  | **0.2205**    | **0.2702**          | **+22.6%**  | **31.6**  |
| 64  | 0.2279        | 0.2297          | +0.8%   | 1.0   |
| 128 | 0.2935        | 0.2980          | +1.5%   | 0.5   |
| 256 | 0.5069        | 0.5130          | +1.2%   | 0.3   |

The H=32 regression is large enough to be visible far outside any plausible measurement noise.

## Root cause

In the original kernel, MMA shape is `M = BLOCK_SIZE_V = 128` regardless of H (BLOCK_SIZE_V is the autotune-picked V tile, not data-dependent). After the transpose, `M = BLOCK_SIZE_H = bsz_h(H) ∈ {16, 32, 64}`. Per `bsz_h`:

| H bucket | BLOCK_SIZE_H | MMA M | Native sm_100 bf16 M? |
|----------|--------------|-------|-----------------------|
| 1..16    | 16           | 16    | No (likely padded)    |
| 17..32   | 32           | 32    | No (likely padded to 64) |
| 33..∞    | 64           | 64    | Yes (native)          |

The pattern matches:
- **H≥64 (M=64 native)**: regression collapses to noise (≤1% on small case).
- **H=17..32 (M=32)**: worst regression (~13% large, ~23% small). Likely M=32 gets padded to M=64, halving useful MMA throughput.
- **H≤16 (M=16)**: smaller regression (~3% large, ~7% small). Possibly because per-tile work is small enough that other overheads dominate, masking part of the MMA inefficiency.

Possible diagnostic to confirm: dump TTGIR/SASS at H=32 to see the emitted `tcgen05.mma` shape and whether M is padded. Not done; conclusion stands without it.

## Why this kills the experiment

The 2-CTA MMA gain we hoped for at H≥128 (the "real" payoff regime) was at best ~5-10% based on Modular's published numbers on a square 4096³ GEMM. Any gain there would have to *also* recover a 5-7% regression at low H to break even on the most common decode shapes. Net expected value is approximately zero or negative.

The operand-swap path is fundamentally incompatible with the heuristic that `BLOCK_SIZE_H` adapts to H: the MMA M-dim inherits H's variability and pays a non-native-shape penalty whenever H mod 64 ≠ 0.

## What stays on this branch

- The transposed kernel itself (preserved as record on `worktree-2cta-mma`).
- This findings doc.
- The `num_ctas=2` autotune attempt comment in `core.py` documenting the cross-CTA reduce constraint (already on `main` from the earlier `b200-2ctas` experiment).

Main branch is unchanged. No merge.

## What would unblock 2-CTA in the future

Either:

1. **Hybrid dispatcher**: keep the original orientation for H<128, dispatch to a transposed+2-CTA variant only at H≥128. Avoids the low-H regression. Cost: two kernels to maintain, plus the H≥128 gain still has to clear the operand-swap overhead at H=128 specifically.
2. **Triton compiler support for cross-CTA reductions**: would let us keep the original orientation and just enable `num_ctas=2`. Out of our hands.
3. **Force `BLOCK_SIZE_H ≥ 64` in the transposed kernel**: would fix the M=16/32 cases by padding to a native MMA shape. Costs 50-75% wasted MMA work at H=1, which is probably worse than the regression we see.

None of these are obviously worthwhile given the size of the upside.

## Repro pointers

- Worktree branch: `worktree-2cta-mma` at `.claude/worktrees/2cta-mma/`.
- Bench command: `make modal-triton-benchmark GPU=b200 POSTFIX=-transposed`.
- Result paths: `benchmarking/modal-results/triton-bench/fi-cupti/b200-transposed{,-r2,-r3}/tp1/` (transposed) and `b200-baseline-r{2,3}/tp1/` from the main repo (baseline reruns).
- Distributed test: `make modal-pytest-distributed GPU=b200`.

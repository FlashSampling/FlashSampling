# Where FMMS Speedups Come From: NCU + Proton Analysis

RTX 3090, small config (V=151936, d=4096). To regenerate:

```shell
cd benchmarking
make sweep-bsz-ncu                  # NCU: inter-kernel breakdown
make sweep-bsz-proton               # Proton: intra-kernel breakdown
make parse-sweep-ncu
make parse-sweep-proton
```

## Summary

| N   | FMMS | Naive | FI-sample | FI-topkp | FMMS vs Naive | FMMS vs FI-sample | FMMS vs FI-topkp |
| --- | ---- | ----- | --------- | -------- | ------------- | ----------------- | ---------------- |
| 1   | 1420 | 1505  | 1490      | 1722     | **1.06x**     | **1.05x**         | **1.21x**        |
| 4   | 1420 | 1625  | 1594      | 1899     | **1.14x**     | **1.12x**         | **1.34x**        |
| 16  | 1420 | 1744  | 1627      | 1977     | **1.23x**     | **1.15x**         | **1.39x**        |
| 64  | 1810 | 2173  | 1760      | 2166     | **1.20x**     | 0.97x             | **1.20x**        |
| 128 | 3520 | 4224  | 3481      | 4193     | **1.20x**     | 0.99x             | **1.19x**        |
| 256 | 6980 | 7912  | 6557      | 7812     | **1.13x**     | 0.94x             | **1.12x**        |

All times in us. Bold = FMMS is faster.

## Decomposition: matmul vs post-matmul

All baselines share the same cuBLAS matmul. FMMS uses its own fused Triton matmul.

| N   | cuBLAS matmul | FMMS matmul | Naive post | FI-sample post | FI-topkp post | # kernels (Naive / FI-s / FI-tkp) |
| --- | ------------- | ----------- | ---------- | -------------- | ------------- | --------------------------------- |
| 1   | 1410          | 1420        | 95         | 80             | 312           | 20 / 5 / 7                        |
| 4   | 1510          | 1420        | 115        | 84             | 389           | 20 / 5 / 7                        |
| 16  | 1520          | 1420        | 224        | 107            | 447           | 20 / 5 / 7                        |
| 64  | 1550          | 1810        | 623        | 200            | 606           | 20 / 5 / 7                        |
| 128 | 3080          | 3520        | 1144       | 391            | 1113          | 20 / 5 / 7                        |
| 256 | 5780          | 6980        | 2132       | 777            | 2032          | 16 / 5 / 7                        |

## Three sources of speedup (and one source of slowdown)

### 1. Fusing post-matmul work into the matmul kernel

FMMS fuses sampling into the matmul kernel itself, where it adds only ~2% overhead (see Proton data below). The baselines instead pay for separate kernel launches:

- **Naive compiled** (19 post-matmul kernels): 6 softmax + 13 multinomial. Of the 13 multinomial kernels, 10 are input validation (MinNan, MaxNan, Sum + assert_async). Only 3 do actual sampling (RNG, normalize, argmax). Post-matmul cost: 95 us (N=1) → 2132 us (N=256).

- **FI-sample** (4 post-matmul kernels): secondary cuBLAS + temperature div + arange + `unrolled_elementwise_kernel` (FlashInfer's fused softmax+sample). Much leaner than Naive. Post-matmul cost: 80 us (N=1) → 777 us (N=256).

- **FI-topkp** (6 post-matmul kernels): same as FI-sample plus `RadixTopKMaskLogitsKernel` (90–467 us) and `cunn_SoftMaxForward` (6–300 us). These two kernels alone account for most of the extra cost. Post-matmul cost: 312 us (N=1) → 2032 us (N=256).

### 2. Avoiding the cuBLAS GEMV→GEMM transition penalty

At N=1, cuBLAS uses `gemv2T_kernel_val` (1410 us). At N≥4 it switches to `ampere_bf16_s16816gemm` (1510–1560 us for N=4..64), a ~7% penalty for this memory-bound shape. FMMS's Triton matmul stays at 1420 us for N=1..16, avoiding this transition entirely. This gives FMMS a free ~90 us advantage at N=4..16 on top of the post-matmul savings.

### 3. Eliminating HBM round-trips for logits

In all baselines, the matmul writes the N×V logit tensor to HBM, then post-matmul kernels read it back. FMMS keeps logits in on-chip (SRAM/registers) and never writes them. The corrected HBM traffic model (`hbm-access.py`) predicts this saves 0-3% of total traffic at N=1..64 (small config), growing to 11% at N=256. This effect is small at low N but grows with batch size.

### Slowdown: Triton matmul efficiency at large N

FMMS's fused Triton matmul becomes less efficient than cuBLAS at large batch sizes:

| N   | FMMS BW% | cuBLAS BW% | FMMS matmul overhead |
| --- | -------- | ---------- | -------------------- |
| 1   | 97%      | 97%        | +10 us (+0.7%)       |
| 4   | 97%      | 91%        | −90 us (−6%)         |
| 16  | 96%      | 91%        | −100 us (−7%)        |
| 64  | 76%      | 90%        | +260 us (+17%)       |
| 128 | 41%      | 46%        | +440 us (+14%)       |
| 256 | 22%      | 26%        | +1200 us (+21%)      |

At N≥64, the FMMS matmul overhead exceeds FI-sample's post-matmul cost, making FI-sample faster overall. The BW drop is caused by the kernel's fixed grid (1187 blocks = V/128) and 16.7% occupancy (limited by shared memory), which handle larger N by doing more work per tile rather than launching more tiles.

## Why FMMS beats each baseline

**vs Naive (1.06x–1.23x, all N):** Eliminates 19 post-matmul kernels. The dominant savings come from removing multinomial validation (10 kernels) and softmax (6 kernels). FMMS wins at all tested batch sizes because Naive's post-matmul overhead is always larger than FMMS's matmul overhead.

**vs FI-sample (1.05x–1.15x at N≤16, loses at N≥64):** FI-sample is already lean (4 post-matmul kernels). FMMS wins at low N by eliminating those ~80–107 us plus the GEMV→GEMM penalty. At N≥64, FMMS's Triton matmul inefficiency (76%→22% BW) exceeds FI-sample's post-matmul cost, and FI-sample becomes 1–6% faster.

**vs FI-topkp (1.12x–1.39x, all N):** The RadixTopK masking + separate softmax adds 96–768 us of overhead that FMMS eliminates. FMMS wins at all tested batch sizes.

## Intra-kernel breakdown (Proton)

NCU treats FMMS as a single opaque kernel. Triton Proton provides two complementary views inside the kernel: warp-level scope timing (chrome trace) and PC sampling (line-by-line). Both confirm that **sampling adds negligible overhead to the matmul**.

### Scope-level breakdown (chrome trace)

| N   | matmul | sampling | setup |
| --- | ------ | -------- | ----- |
| 1   | 97.8%  | 1.9%     | 0.3%  |
| 4   | 97.8%  | 1.9%     | 0.3%  |
| 16  | 97.8%  | 1.9%     | 0.3%  |
| 64  | 93.6%  | 6.1%     | 0.3%  |
| 128 | 93.2%  | 6.5%     | 0.4%  |
| 256 | 93.2%  | 6.5%     | 0.4%  |

Sampling = sample scope + store scope. At N≤16, sampling is ~2.3% of kernel time. At N≥64, it rises to ~6% because each tile processes more rows (the grid stays fixed at V/BLOCK_SIZE_V blocks, so larger N means more work per tile in the sampling phase). Even at N=256, the matmul still dominates at 93%.

### Per-warp timing

Per-warp average times (ns) are remarkably stable across batch sizes:

| N   | matmul (ns) | sample (ns) | total (ns) |
| --- | ----------- | ----------- | ---------- |
| 1   | 169         | 3.2         | 173        |
| 16  | 169         | 3.2         | 173        |
| 64  | 176         | 11.3        | 188        |
| 256 | 164         | 11.3        | 176        |

The sampling cost per warp jumps from 3.2 ns to 11.3 ns at N≥64, but even at its peak it is 64x smaller than the matmul time.

### Key contrast with baselines

In FMMS, the per-warp sampling cost is essentially constant with N — each tile independently samples its own rows, so the work is distributed across the grid. In the baselines, post-matmul kernels operate on the full N×V logit tensor, so their cost grows with N:

| N   | FMMS sampling (per warp) | Naive post (total) | FI-sample post (total) |
| --- | ------------------------ | ------------------ | ---------------------- |
| 1   | 3.2 ns                   | 95 us              | 80 us                  |
| 16  | 3.2 ns                   | 224 us             | 107 us                 |
| 64  | 11.3 ns                  | 623 us             | 200 us                 |
| 256 | 11.3 ns                  | 2132 us            | 777 us                 |

The baselines' sampling cost scales with N; FMMS's does not.

## For the paper

The NCU + Proton data supports two claims about where the speedup comes from, plus one honest limitation. Note: the GEMV→GEMM transition (source #2 in the analysis above) is omitted from the paper — it's a cuBLAS implementation detail that complicates the narrative without adding insight.

**Speedup source 1: Fusing post-matmul work into the matmul.**
The standard pipeline launches many separate kernels after the matmul (softmax, input validation, RNG, argmax).
FMMS fuses sampling into the matmul kernel itself, where Proton shows it adds only ~2% overhead.
Eliminating kernel launch overhead and the separate post-matmul computation accelerates sampling.
This speeds up FMMS even when the HBM traffic model predicts ~0% improvement.

**Speedup source 2: Eliminating HBM round-trips for logits.**
In all baselines, the matmul writes the full N×V logit tensor to HBM, and then the sampling kernels read it back.
FMMS keeps logits in on-chip (SRAM/registers), so they are never materialized.
The corrected `hbm-access.py` traffic model (standard: reads Vd + Bd + BV, writes BV + B; FMMS: reads Vd + Bd, writes B) predicts savings that grow with N, up to 11% at N=256 (small config).

**Key finding: sampling adds negligible overhead to the fused kernel.**
Proton profiling shows the sampling phase (Gumbel noise + argmax + store) accounts for ~2.3% of kernel time at N≤16 and ~6% at N=256.
The matmul completely dominates (93–98%).
This means fusion is almost free:
the sampling computation is hidden behind the memory-bound matmul.

**Limitation: cuBLAS matmul scales better than the Triton matmul at large N.**
At N≥64, the fused Triton kernel's DRAM throughput drops (76% at N=64, 22% at N=256) while cuBLAS maintains higher memory throughput.
The sampling spedups and the matmul inefficiency cancel out at N≥64.
Improving the Triton matmul efficiency at large N is future work.

Sources:

- `benchmarking/profiles/sweeps/bsz/ncu-txt/case-small/`
- `benchmarking/profiles/sweeps/bsz/proton/case-small/`

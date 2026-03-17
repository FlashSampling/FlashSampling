# Arithmetic intensity of the decode matmul

## The computation

The FMMS kernel computes `W[V, D] × H[D, H]^T` where H is the batch dimension
(number of hidden states / sequences being decoded in parallel).

```
FLOPs  = 2 · V · D · H        (one multiply + one add per output element)
Bytes  = 2 · V · D            (weight matrix, BF16 = 2 bytes per element)
         + 2 · D · H          (hidden states, negligible when H ≪ V)
       ≈ 2 · V · D            (dominated by weight reads)

Arithmetic intensity = FLOPs / Bytes ≈ H
```

At H=1, every weight element is loaded from HBM to compute a single
multiply-add — pure memory-bound. At H=64, each weight element contributes to
64 multiply-adds — still memory-bound on modern GPUs.

## Compute-to-bandwidth ratio (ops:byte)

A matmul becomes compute-bound when its arithmetic intensity exceeds the GPU's
**compute-to-bandwidth ratio**:

```
ops:byte ratio = peak compute (FLOPs/s) ÷ peak memory bandwidth (bytes/s)
```

### H100 SXM

Data from the [NVIDIA H100 datasheet][1] and the [H100 whitepaper (Table 8)][2]:

- Peak HBM3 bandwidth: **3.35 TB/s**

| Precision              | Peak TFLOP/s (dense) | Peak TFLOP/s (sparse) |
|------------------------|---------------------:|----------------------:|
| FP16 / BF16 tensor core|                  989 |                 1,979 |
| FP8 tensor core        |                1,979 |                 3,958 |
| FP32 tensor core       |                  495 |                   989 |

Resulting ops:byte ratios (dense, which is the relevant case since LLM weights
are not structured-sparse):

| Precision              | ops:byte ratio | Matmul becomes compute-bound at H ≈ |
|------------------------|---------------:|-------------------------------------:|
| BF16 tensor core       |           ~295 |                                  295 |
| FP8 tensor core        |           ~591 |                                  591 |

Example for BF16: `989 × 10¹² / 3.35 × 10¹² ≈ 295 FLOPs/byte`.

### RTX 3090 (Ampere, sm_86)

Data from the [RTX 3090 specifications][3]:

- Peak HBM bandwidth: **936 GB/s**
- BF16 tensor core (dense): **142 TFLOP/s**

```
ops:byte = 142 × 10¹² / 936 × 10⁹ ≈ 152
```

The matmul becomes compute-bound around H ≈ 152 on RTX 3090.

## Implication for FMMS

For LLM decode workloads:
- **H=1** (single request): arithmetic intensity 1 — deeply memory-bound on
  all GPUs. The kernel is ~295× below the compute-bound threshold on H100.
- **H=32** (typical serving batch): arithmetic intensity 32 — still ~9×
  below threshold.
- **H=128**: approaching the crossover on RTX 3090 (~152), still well below
  on H100 (~295).

Since the kernel is memory-bound, **throughput is determined by how fast it
reads the weight matrix from HBM**. Fusing top-k/top-p/sampling into the
matmul kernel avoids additional HBM traffic for intermediate logits. Every
extra kernel launch that re-reads data wastes bandwidth on the bottleneck path.

## Empirical verification: return-logits ablation

The `RETURN_LOGITS` flag adds a single extra store of the raw logits tensor
`[V, H]` in float32 to the kernel, with no additional compute. This makes it a
clean controlled experiment to test the IO model.

### Predicted overhead

```
extra bytes    = 4 · V · H          (float32 logits store)
baseline bytes = 2 · V · D          (weight matrix read, BF16)
predicted overhead = 4·V·H / (2·V·D) = 2H / D
```

The overhead is independent of V and grows linearly with H.

### Results (B200, 5-run average)

**Large case (V=128,256, D=8,192):**

| H   | Predicted (2H/D) | Measured | FMMS (ms)   | +logits (ms)  |
|-----|------------------|----------|------------:|-------------:|
| 1   | 0.02%            | 0.5%     | 0.334±0.003 | 0.335±0.003  |
| 4   | 0.10%            | 0.7%     | 0.340±0.003 | 0.342±0.003  |
| 16  | 0.39%            | 1.6%     | 0.341±0.002 | 0.346±0.003  |
| 64  | 1.56%            | 3.6%     | 0.352±0.003 | 0.365±0.003  |
| 128 | 3.13%            | 5.4%     | 0.432±0.029 | 0.456±0.008  |
| 256 | 6.25%            | 8.1%     | 0.736±0.013 | 0.796±0.032  |

**Small case (V=151,936, D=4,096):**

| H   | Predicted (2H/D) | Measured | FMMS (ms)   | +logits (ms)  |
|-----|------------------|----------|------------:|-------------:|
| 1   | 0.05%            | 0.9%     | 0.214±0.001 | 0.216±0.001  |
| 4   | 0.20%            | 1.2%     | 0.219±0.002 | 0.222±0.002  |
| 16  | 0.78%            | 2.1%     | 0.221±0.001 | 0.225±0.001  |
| 64  | 3.13%            | 4.8%     | 0.235±0.002 | 0.246±0.001  |
| 128 | 6.25%            | 8.9%     | 0.291±0.007 | 0.317±0.006  |
| 256 | 12.50%           | 15.2%    | 0.512±0.017 | 0.589±0.015  |

### Analysis

The measured overhead consistently exceeds the prediction by ~1.5-2.7 percentage
points (a roughly constant offset). This gap likely comes from store instruction
overhead (address computation, mask evaluation, issue slots) that exists even
when bandwidth is not saturated. Despite this offset, the linear scaling with H
matches the prediction well. Halving D from 8,192 to 4,096 roughly doubles the
overhead at every batch size, as predicted.

Raw CSVs: `benchmarking/modal-results/triton-bench-b200-logits-ablation-run{1..5}/`

## References

[1]: https://resources.nvidia.com/en-us-tensor-core
[2]: https://resources.nvidia.com/en-us-data-center-overview/gtc22-whitepaper-hopper
[3]: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/

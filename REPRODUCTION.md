# Reproduction Guide

This document describes how to reproduce the results in the FlashSampling paper. Each section maps a paper artifact (table or figure) to the `make` target and the resulting output path.

The paper reports two classes of experiments:

1. **Kernel microbenchmarks** (Tables 1, 7; Figures 2, 3, 4, 5). FlashSampling vs. three baselines on H100, H200, B200, B300 at TP=1, 2, 4, 8.
2. **End-to-end vLLM** (Tables 4, 5; Figure 6). TPOT on Qwen3-1.7B and Qwen3-8B (TP1), Qwen3-32B and Llama-3.3-70B (TP2), all on B200.

Both pipelines run on Modal cloud GPUs. A local NVIDIA workstation can run a subset (NCU/Proton breakdown sweeps and the chi-squared correctness test).

## 1. Prerequisites

1. **Python `>=3.10`.** Install dependencies into the in-repo `.venv` and activate it:
   ```bash
   uv sync --all-extras
   source .venv/bin/activate    # or prepend .venv/bin to PATH
   ```
   This installs Modal, FlashInfer, Helion, and the rest of the benchmarking deps from `pyproject.toml` / `uv.lock`.
   The `make` targets below invoke `modal`, `python`, and `pytest` directly, so the venv must be active (or its `bin/` on `PATH`) for them to work.
2. **Modal account** with access to H100, H200, B200, and B300 GPUs. With the venv active, authenticate once:
   ```bash
   modal setup
   ```
   The Makefile uses a Modal volume named `fused-mm-sample`, which is created automatically on first run.
3. **HuggingFace token** exported as `HF_TOKEN` (needed for gated models such as Llama-3.3-70B-Instruct).
4. *(Optional)* A local CUDA toolkit for the NCU/Proton breakdown sweeps.

## 2. Software versions

These versions are baked into the Modal image used for all reported results:

| Component   | Version |
|-------------|---------|
| PyTorch     | 2.10.0  |
| CUDA        | 13.0    |
| Triton      | 3.6.0   |
| FlashInfer  | 0.6.9   |

The local `.venv` produced by `uv sync` is used only as a Modal driver and for plotting; its PyTorch and Triton versions are pinned by `uv.lock` and may differ from the table above. To verify the live Modal image versions, run `make modal-versions`.

Inputs and weights are BF16. Kernels are warmed up for 25 iterations before timing. Kernel timings are CUPTI medians over 100 iterations (`bench_fn=fi-cupti`).

## 3. Kernel microbenchmarks

### 3.1 Single-GPU sweep (Tables 1, 7; Figures 2, 5, 7)

Each call sweeps batch sizes B ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256} for both the small (D=4,096; V=151,936) and large (D=8,192; V=128,256) configurations.

```bash
# One GPU at a time:
make modal-triton-benchmark GPU=b200
make modal-triton-benchmark GPU=b300
make modal-triton-benchmark GPU=h200
make modal-triton-benchmark GPU=h100!

# Or all four in parallel (each Modal job claims its own GPU):
make modal-triton-benchmark-all-gpus
```

Outputs land in `benchmarking/modal-results/triton-bench/fi-cupti/<gpu>/tp1/`:

- `fused-mm-sample-batch-scaling-{small,large}.csv` — raw per-provider latencies.
- `relative-performance-vs-pytorch.csv`, `relative-performance-vs-flashinfer.csv` — speedup ratios populating Tables 1 and 7.
- `custom-plots/case-{small,large}/{batch-scaling,memory-throughput,roofline,relative-perf-vs-pytorch,relative-perf-vs-flashinfer}.{png,pdf}` — Figures 2, 5, 7.

Plotting is included in `modal-triton-benchmark`. To replot from existing CSVs, run `make plot-all`.

### 3.2 Multi-GPU TP=2/4/8 sweep (Figure 4)

```bash
# TP=2 on each GPU:
make modal-triton-benchmark GPU=b200 N_PROCS=2
make modal-triton-benchmark GPU=b300 N_PROCS=2
make modal-triton-benchmark GPU=h200 N_PROCS=2
make modal-triton-benchmark GPU=h100! N_PROCS=2

# TP=4 and TP=8 (B200 only in the paper):
make modal-triton-benchmark GPU=b200 N_PROCS=4
make modal-triton-benchmark GPU=b200 N_PROCS=8
```

Plot the scaling figure:

```bash
make plot-tp-scaling GPU=b200
```

Output: `imgs/tp-scaling/tp-scaling.pdf`.

### 3.3 Sampling-latency / matmul-latency breakdown (Table 3, Figure 3)

The sampling-vs-matmul split is profiled with NCU and Proton on a local RTX 3090.

```bash
cd benchmarking

# NCU sweep across batch sizes for all providers:
make sweep-bsz-ncu CASE=small

# Proton intra-kernel sweep (FlashSampling kernel breakdown):
make sweep-bsz-proton CASE=small

# Plot matmul-latency.pdf and sampling-latency.pdf:
make plot-bsz-sweep-runtime CASE=small
```

Outputs: `benchmarking/profiles/sweeps/bsz/ncu-txt/.../summary.txt` (Table 3) and `benchmarking/imgs/.../{matmul,sampling}-latency.pdf` (Figure 3).

### 3.4 Logits-store ablation (Table 8)

The provider `fused-triton-ret-logits` toggles `RETURN_LOGITS=True` inside the FMMS Triton kernel: it stores the computed `[B,V]` FP32 logits to HBM, isolating the round-trip cost predicted by the IO model (`2B/D`).

```bash
make modal-triton-benchmark GPU=b200 \
    NAME=fused-triton,fused-triton-ret-logits \
    POSTFIX=-logits-ablation
```

The ratio `fused-triton-ret-logits` / `fused-triton` runtimes gives Table 8's "Measured" column.

## 4. End-to-end vLLM

The vLLM experiments (Tables 4, 5; Figure 6) are produced from a private vLLM fork that integrates the FlashSampling sampler.
The fork's patch can be inspected at the anonymous mirror linked from the submission, but the Modal benchmark scripts and the fork itself are omitted from this supplemental to preserve double-blind anonymity.
Both will be released alongside the camera-ready version after acceptance.

The reported configuration was: concurrency levels {1, 2, 4, 8, 16, 32, 64} with 5 runs per level for both `baseline` and `fmms-triton` variants; sampling parameters `temperature=0.6, top_k=-1, top_p=1.0`; dataset `AI-MO/aimo-validation-aime` with `--hf-output-len 256` and `--max-model-len 1024`, prefix caching disabled.

## 5. Empirical correctness

### 5.1 Kernel chi-squared test

```bash
.venv/bin/pytest tests/test_core.py::test_sampling_distribution -v
```

This parametrizes over all providers, vocab sizes (100, 256), and `n_hidden_states ∈ {1, 2}`, drawing 5,000 samples each and running a chi-squared goodness-of-fit test against the theoretical softmax distribution.

For TP=2 distributed:

```bash
make pytest-distributed                  # local, requires >=2 NVLink GPUs
make modal-pytest-distributed GPU=b200   # via Modal
```

### 5.2 GSM8K end-to-end accuracy

The 89.4% vs. 89.6% comparison on Qwen3-1.7B / GSM8K (1,319 questions) uses the same private vLLM fork as Section 4 to drive baseline and FlashSampling decoding, then judges answers with a separate LLM. The eval scripts depend on the fork and are omitted from this supplemental for the same anonymity reason; they will be released alongside the camera-ready.

## 6. Hardware specs (Table 6)

Table 6 (HBM, peak BF16 dense TFLOP/s, ops:byte ratio) is populated from the per-GPU constants in `benchmarking/plot-triton-bench.py` (`GPU_PEAK_BW_GBS`, `GPU_PEAK_COMPUTE_TFLOPS`). No experiment is needed; the values are vendor specs.

## 7. Mapping paper artifacts to outputs

| Paper artifact                            | Command                                          | Output path |
|-------------------------------------------|--------------------------------------------------|-------------|
| Table 1 (small speedup, 4 GPUs)           | `modal-triton-benchmark-all-gpus`                | `triton-bench/fi-cupti/<gpu>/tp1/relative-performance-*.csv` |
| Table 7 (large speedup, 4 GPUs)           | `modal-triton-benchmark-all-gpus`                | same dir, `*-large.csv` rows |
| Table 3 (sampling % of kernel time)       | `make sweep-bsz-ncu` + `parse_ncu_sweep`         | `profiles/sweeps/bsz/ncu-txt/.../summary.txt` |
| Table 4 (TPOT % speedup)                  | `modal-vllm-benchmark-full-*`                    | `vllm-bench-b200-tp{1,2}/<model>/results.txt` |
| Table 5 (absolute TPOT)                   | same                                             | same |
| Table 8 (logits ablation)                 | `POSTFIX=-logits-ablation` runs                  | manual ratio of CSVs |
| Figure 2 (relative perf, B200)            | `modal-triton-benchmark GPU=b200`                | `triton-bench/.../custom-plots/case-small/relative-perf-vs-{pytorch,flashinfer}.pdf` |
| Figure 3 (sampling/matmul latency)        | `make sweep-bsz-ncu` + `plot-bsz-sweep-runtime`  | `benchmarking/imgs/.../{sampling,matmul}-latency.pdf` |
| Figure 4 (TP scaling)                     | TP={1,2,4,8} sweeps + `plot-tp-scaling`          | `imgs/tp-scaling/tp-scaling.pdf` |
| Figure 5 + Figure 7 (roofline, B200/H100) | `modal-triton-benchmark` per GPU                 | `triton-bench/.../custom-plots/case-small/{roofline,memory-throughput}.pdf` |
| Figure 6 (TPOT vs concurrency)            | `make plot-vllm-bench{,-tp2}`                    | `vllm-bench-b200-tp{1,2}/tpot-<model>.pdf` |

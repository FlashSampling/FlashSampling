# Modal benchmarking

## Modal profiles

Two Modal workspaces are configured. Switch with `modal profile activate <name>`.

- **`tomasruizt`** (personal): Used for vllm-bench runs on gpt-oss-120b and Qwen3-1.7B. The `fused-mm-sample` volume here holds these results.
- **`lmu-css`** (default): Used for triton-bench runs and vllm-bench runs on Qwen3-8B. The `fused-mm-sample` volume here holds these results.

Check the active profile with `modal profile list` (the `•` marker shows the active one). When downloading results with `make modal-get-results-*`, ensure the correct profile is active or the volume lookup will fail silently (no matching directory).

## Modal volume management

The `fused-mm-sample` volume stores benchmark results, model caches, and torch.compile caches. Useful commands:

```bash
modal volume ls fused-mm-sample                     # list root
modal volume ls fused-mm-sample triton-bench-b200   # list subdirectory
modal volume rm fused-mm-sample <path> -r           # delete recursively
modal volume get fused-mm-sample <path> <local_dir> # download to local
```

**Paths with special characters** (e.g. `triton-bench-h100!`): use double quotes around the path argument to prevent shell expansion.

## Triton-bench (kernel microbenchmarks)

Kernel microbenchmarks run on Modal cloud GPUs. The root `Makefile` has a three-step pipeline:

```bash
# Full pipeline: run bench → download results → plot
make modal-triton-benchmark GPU=h100!

# Or run steps individually:
make modal-create-results-triton-bench GPU=h100!   # runs on Modal, saves logs
make modal-get-results-triton-bench GPU=h100!      # downloads from Modal volume
make modal-plot-triton-bench GPU=h100!             # generates plots from CSVs
```

**GPU options**: `h100!`, `h100`, `a100-80gb`, `b200`, `h200` (the `!` suffix means dedicated/reserved GPU on Modal). Default is `b200`.

**Benchmark cases**: Controlled by `CASE` env var (default `"all"` → runs `["large", "small"]`). Available cases in `src/fused_mm_sampling/bench/triton_benchmark.py`:
- `large`: V=128,256, d=8,192 (Llama 3 70B)
- `small`: V=128,256, d=4,096 (Llama 3 8B)
- `qwen3-1.7b`: V=151,936, d=2,048
- `gpt-oss-120b`: V=201,088, d=2,880

**POSTFIX**: Use `POSTFIX=-foo` to create separate result directories for A/B comparisons without overwriting previous runs: `make modal-triton-benchmark GPU=h100! POSTFIX=-experiment1`.

**Key files**:
- `src/fused_mm_sampling/modal_lib/modal_triton_benchmark.py` — Modal app definition
- `src/fused_mm_sampling/modal_lib/utils.py` — image (PyTorch 2.10.0 + CUDA 13.0), volume config
- `src/fused_mm_sampling/bench/triton_benchmark.py` — benchmark runner, `Args` dataclass, `BENCHMARK_CASES`
- `benchmarking/plot-triton-bench.py` — plotting script, also contains `GPU_PEAK_BW_GBS` and `GPU_PEAK_COMPUTE_TFLOPS` dicts with per-GPU specs (HBM bandwidth, peak BF16 TFLOP/s)

**Results location**: `benchmarking/modal-results/triton-bench-{GPU}{POSTFIX}/` containing CSVs, plots in `custom-plots/`, and `logs.txt`.

## Triton benchmark CSV format

Triton's `perf_report` appends ` (Time (ms))` to column names based on `ylabel`. The plotting code strips this suffix via `read_triton_bench_csv()` in `benchmarking/plot-triton-bench.py`.

## vLLM-bench (end-to-end)

End-to-end vLLM benchmarks on Modal cloud GPUs. The root `Makefile` has per-model convenience targets and a composable pipeline:

```bash
# Per-model full benchmarks (all concurrency levels, 5 runs):
make modal-vllm-benchmark-full-gpt-oss-120b GPU=b200
make modal-vllm-benchmark-full-qwen3-1.7b GPU=b200
make modal-vllm-benchmark-full-qwen3-8b GPU=b200

# Composable pipeline (any model, any sweep):
make modal-vllm-benchmark GPU=b200 VLLM_MODEL=openai/gpt-oss-120b VLLM_SWEEP=all

# Run a single variant (e.g. rerun just baseline):
make modal-vllm-benchmark GPU=b200 VLLM_MODEL=openai/gpt-oss-120b VLLM_SWEEP=all VLLM_VARIANTS=baseline

# Steps can be run individually:
make modal-create-results-vllm-bench GPU=b200 VLLM_MODEL=...  # runs on Modal
make modal-get-results-vllm-bench GPU=b200                     # downloads from volume
make modal-collect-results-vllm-bench GPU=b200 VLLM_MODEL=...  # runs collect_results.py locally
```

**Key files**:
- `src/fused_mm_sampling/modal_lib/modal_vllm_benchmark.py` — Modal app that runs `vllm bench sweep serve` for each variant
- `benchmarking/vllm/bench-params.json` / `quick-bench-params.json` — single source of truth for sweep parameters (shared between local and Modal benchmarks)
- `benchmarking/vllm/collect_results.py` — result collection, run locally after downloading
- `benchmarking/vllm/parse_engine_stats.py` — works with both `sweep.log` and Modal log files (engine stats lines are the same format)

**Results location**: `benchmarking/modal-results/vllm-bench-{GPU}{POSTFIX}/` with per-model subdirectories containing `baseline/`, `fmms-triton/`, `logs/`, and `results.txt`.

**Makefile variables**:
- `GPU` — Modal GPU type (default: `b200`)
- `VLLM_MODEL` — HuggingFace model ID (default: `openai/gpt-oss-120b`)
- `VLLM_SWEEP` — `quick` (1 concurrency, 1 run, `--enforce-eager`) or `all` (full sweep, 5 runs)
- `VLLM_VARIANTS` — comma-separated variant filter, e.g. `baseline` or `fmms-triton`. Empty = all variants.
- `POSTFIX` — suffix for result directory (for A/B comparisons)

**Logs**: Timestamped per-model in `<model_slug>/logs/<YYYYMMDD_HHMMSS>.txt`. Multiple parallel runs won't collide.

### Modal vLLM image build

The image uses `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel` as base. Key pitfall: vLLM's `VLLM_USE_PRECOMPILED=1` installs precompiled `.so` files built for torch 2.10.0, but vLLM's metadata pins `torch==2.9.1`. The image build works around this with a two-step install:

1. `cd /opt/vllm && VLLM_USE_PRECOMPILED=1 uv pip install --system -e '.[bench]'` — installs vLLM with torch 2.9.1
2. `uv pip install --system 'torch==2.10.0' 'torchvision>=0.25' 'torchaudio>=2.10'` — force-upgrades torch to match the precompiled `.so`

Without step 2, you get an ABI mismatch: `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`.

Other image build lessons:
- `.pip_install("uv")` fails on Ubuntu 24.04 (PEP 668). Use `.run_commands("pip install --break-system-packages uv")`.
- `add_local_dir()` / `add_local_file()` require `copy=True` when subsequent build steps need the files.
- B200 GPU requires CUDA 13.0 / sm_100. PyTorch 2.9.1 only supports up to sm_90, so the 2.10.0 base image is necessary.
- `HF_TOKEN` is passed via `modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})`. The code intentionally fails if `HF_TOKEN` is not set locally.

### torch.compile startup overhead

On gpt-oss-120b (B200), `torch.compile` graph compilation takes **~8 minutes** on the first server start (495s for graph compilation + kernel downloads). The `--server-ready-timeout` is set to **1200s (20 min)** in the Modal app to accommodate cold-start compilation.

The second variant (fmms-triton) benefits from the compilation cache warmed by the baseline, so it starts faster (~2-3 min).

### Caching on Modal volumes

Ephemeral container caches (torch.compile graphs, flashinfer cubins) are lost between runs, causing expensive re-compilation. **Fix: set `XDG_CACHE_HOME` to the Modal volume path.** This is the standard Linux env var for cache directories — both vLLM (`~/.cache/vllm/`) and flashinfer (`~/.cache/flashinfer/`) respect it automatically. Prefer env vars over symlinks for redirecting caches.

The Modal function sets three cache-related env vars:
- `HF_HOME` → `{volume_path}/hf-cache` (model weights)
- `XDG_CACHE_HOME` → `{volume_path}/cache` (torch.compile, flashinfer cubins, etc.)

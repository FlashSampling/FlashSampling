# Modal benchmarking

## Modal profiles

Two Modal workspaces are configured. Switch with `modal profile activate <name>`.

- **`tomasruizt`** (personal): Used in the past, but not anymore.
- **`lmu-css`** (default): Used for triton-bench runs and vllm-bench runs on Qwen3 models, including Qwen3-1.7B and Qwen3-8B. The `fused-mm-sample` volume here holds these results.

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
- `VLLM_SWEEP` — `quick` (1 concurrency, 1 run, `--enforce-eager`) or `all` (batch sizes 1–64, 5 runs)
- `VLLM_VARIANTS` — comma-separated variant filter, e.g. `baseline` or `fmms-triton`. Empty = all variants.
- `VLLM_RESUME_EXPERIMENT` — if set to a previous experiment dir name (e.g. `20260409_101524`), passes `--resume --experiment-name <name>` to `vllm bench sweep serve` so it picks up where the previous run left off, skipping any `(concurrency, run)` combos that already have a `run=N.json` on the modal volume. See "Resuming a partial sweep" below.
- `POSTFIX` — suffix for result directory (for A/B comparisons)

**Logs**: Timestamped per-model in `<model_slug>/logs/<YYYYMMDD_HHMMSS>.txt`. Multiple parallel runs won't collide.

### Resuming a partial sweep

`vllm bench sweep serve` writes one `run=N.json` per (concurrency, run) combo and a final `summary.csv` after all combos complete. If the sweep is interrupted (e.g. by a transient HF Hub 5xx error or a kernel crash mid-sweep), the partial state on the modal volume contains the JSONs for the completed combos but no `summary.csv`. Re-running the same `make modal-vllm-benchmark` from scratch would start over with a fresh experiment-name and re-run everything.

To pick up where it left off, pass `VLLM_RESUME_EXPERIMENT=<experiment-name>`:

```bash
make modal-vllm-benchmark \
    GPU=b200 VLLM_MODEL=Qwen/Qwen3-1.7B VLLM_SWEEP=all \
    VLLM_VARIANTS=fmms-triton \
    VLLM_RESUME_EXPERIMENT=20260409_101524
```

The sweep tool will print `Found existing results.` for each already-complete combo and only execute the missing ones. After completion, it writes the canonical `summary.csv`. The downloaded local results then look identical to a fresh complete run.

The experiment-name is the timestamp directory under `<model>/<variant>/` on the modal volume (e.g. `vllm-bench-b200/Qwen3-1.7B/fmms-triton/20260409_101524/`). It is **not** the same as the local log file timestamp under `<model_slug>/logs/`, which is set independently when `make` runs.

### Modal vLLM image build

The image uses `pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel`, matching the fork's `torch==2.11.0` requirement and the ABI of its upstream precompiled wheel.
The fork is installed non-editably with `VLLM_USE_PRECOMPILED=1`, `--no-build-isolation`, and an explicit `VLLM_PRECOMPILED_WHEEL_COMMIT`.

#### Determining `VLLM_PRECOMPILED_WHEEL_SHA`

vLLM publishes precompiled wheels for commits on upstream `main`, not for commits that exist only on the fork.
The wheel SHA must therefore be the upstream commit on which the fork's Python-only integration changes are based, rather than `VLLM_FORK_SHA` itself.

For `VLLM_FORK_SHA=7a74973e4dc727df979f2a5ec9fff64ac5319467`, the most recent merge of `main` into `feature/fmms-sampler` is `ed9910164b84f01fb363f0534907c2076c6c96c0`.
That merge has these parents:

```text
0027ebfc2372357c4a82d1a8f1f5d80baa8eeefe 1a2c17634eccc4e68d9e1ab654f702d55361c754
```

The first parent is the previous feature-branch state.
The second parent is the upstream `main` tip merged into the feature branch, so the selected wheel pin is:

```text
VLLM_PRECOMPILED_WHEEL_SHA=1a2c17634eccc4e68d9e1ab654f702d55361c754
```

The derivation can be reproduced in the vLLM checkout:

```bash
fork_sha=7a74973e4dc727df979f2a5ec9fff64ac5319467
merge_sha=$(git rev-list --first-parent --merges -n 1 "$fork_sha")
git show -s --format='%H%n%P%n%s' "$merge_sha"
```

Use the second hash printed on the parents line.
After changing or rebasing the fork, repeat this procedure and update both constants together.
This approach is valid only while the commits after the upstream merge do not modify vLLM's compiled C++ or CUDA sources.
If compiled sources change, build a wheel for the fork instead of reusing the upstream binary.

Leaving the wheel commit unpinned selected a newer nightly artifact whose installed package exposed `_C_stable_libtorch` but not the `_C.abi3.so` required by this fork.
Pinning the upstream merge parent produced the expected extension.
The Modal image build verifies this explicitly with:

```bash
test -f /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so
```

Other image build lessons:
- `.pip_install("uv")` fails on Ubuntu 24.04 (PEP 668). Use `.run_commands("pip install --break-system-packages uv")`.
- `add_local_dir()` / `add_local_file()` require `copy=True` when subsequent build steps need the files.
- B200 GPU requires CUDA 13.0 / sm_100, so the CUDA 13.0 base image is necessary.
- `HF_TOKEN` is passed via `modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]})`. The code intentionally fails if `HF_TOKEN` is not set locally.

### torch.compile startup overhead

On gpt-oss-120b (B200), `torch.compile` graph compilation takes **~8 minutes** on the first server start (495s for graph compilation + kernel downloads). The `--server-ready-timeout` is set to **1200s (20 min)** in the Modal app to accommodate cold-start compilation.

The second variant (fmms-triton) benefits from the compilation cache warmed by the baseline, so it starts faster (~2-3 min).

### Caching on Modal volumes

Ephemeral container caches (torch.compile graphs, flashinfer cubins) are lost between runs, causing expensive re-compilation. **Fix: set `XDG_CACHE_HOME` to the Modal volume path.** This is the standard Linux env var for cache directories — both vLLM (`~/.cache/vllm/`) and flashinfer (`~/.cache/flashinfer/`) respect it automatically. Prefer env vars over symlinks for redirecting caches.

The Modal function sets three cache-related env vars:
- `HF_HOME` → `{volume_path}/hf-cache` (model weights)
- `XDG_CACHE_HOME` → `{volume_path}/cache` (torch.compile, flashinfer cubins, etc.)

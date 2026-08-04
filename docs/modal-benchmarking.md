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

**GPU options**: `b300`, `b200`, `h200`, `h100!`, `h100`, and `a100-80gb`.
The `!` suffix means a dedicated or reserved GPU on Modal.
The default is `b200`.

**Benchmark cases**: Pass the `CASE` Make variable as a command argument.
The default `all` runs `large` and `small`.
Available cases are defined in `src/fused_mm_sampling/bench/triton_benchmark_lib.py`:
- `large`: V=128,256, d=8,192 (Llama 3 70B)
- `small`: V=151,936, d=4,096 (Qwen3 8B and Qwen3-235B MoE)
- `qwen3-1.7b`: V=151,936, d=2,048
- `gpt-oss-120b`: V=201,088, d=2,880
- `kimi-k2.5`: V=163,840, d=7,168

**POSTFIX**: Use `POSTFIX=-foo` to create separate result directories for A/B comparisons without overwriting previous runs: `make modal-triton-benchmark GPU=h100! POSTFIX=-experiment1`.

**Key files**:
- `src/fused_mm_sampling/modal_lib/modal_triton_benchmark.py`: Modal app definition.
- `src/fused_mm_sampling/modal_lib/utils.py`: PyTorch 2.11.0 and CUDA 13.0 image plus volume configuration.
- `src/fused_mm_sampling/bench/triton_benchmark_lib.py`: runner, `Args`, and `BENCHMARK_CASES`.
- `benchmarking/plot-triton-bench.py`: plotting and per-GPU HBM and BF16 peak specifications.

**Results location**: `benchmarking/modal-results/triton-bench/{BENCH_FN}/{GPU}{POSTFIX}/tp{N_PROCS}/` containing CSVs, plots under `custom-plots/`, and `logs.txt`.

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
- `src/fused_mm_sampling/modal_lib/modal_vllm_benchmark.py`: Modal app that runs `vllm bench sweep serve` for each variant.
- `benchmarking/vllm/bench-params.json` and `quick-bench-params.json`: shared sweep parameters for local and Modal runs.
- `benchmarking/vllm/collect_results.py`: local result collection after download.
- `benchmarking/vllm/parse_engine_stats.py`: engine-stat parsing for `sweep.log` and Modal logs.

**Results location**: `benchmarking/modal-results/vllm-bench-{GPU}-tp{N_PROCS}{POSTFIX}/` with per-model subdirectories containing `baseline/`, `fi2/`, `fmms-triton/`, `logs/`, and `results.txt`.

**Makefile variables**:
- `GPU`: Modal GPU type, defaulting to `b200`.
- `VLLM_MODEL`: Hugging Face model ID, defaulting to `openai/gpt-oss-120b`.
- `VLLM_SWEEP`: `quick` for one eager run or `all` for five runs at batch sizes 1 through 64.
- `VLLM_VARIANTS`: comma-separated variant filter such as `baseline` or `fmms-triton`; empty means every variant.
- `VLLM_RESUME_EXPERIMENT`: previous experiment directory to resume through `--resume --experiment-name`.
- `POSTFIX`: result-directory suffix for A/B comparisons.

**Logs**: Timestamped per-model in `<model_slug>/logs/<YYYYMMDD_HHMMSS>.txt`.
Parallel launches for different models do not collide.
Two launches for the same model within the same second can still select the same local path, so use provider-specific Modal app logs or the `sweep.log` inside each experiment directory when attribution is ambiguous.

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

Ephemeral container caches such as torch compile graphs and FlashInfer cubins are lost between runs, causing expensive recompilation.
Set `XDG_CACHE_HOME` to the Modal volume path.
Both vLLM (`~/.cache/vllm/`) and FlashInfer (`~/.cache/flashinfer/`) respect this standard cache variable.
Prefer environment variables over symlinks for redirecting caches.

The Modal function sets three cache-related env vars:
- `HF_HOME` → `{volume_path}/hf-cache` (model weights)
- `XDG_CACHE_HOME` → `{volume_path}/cache` (torch.compile, flashinfer cubins, etc.)

## Run-level interpretation

Modal can place independent runs on different host classes.
When a benchmark compares a baseline with candidates, measure them interleaved in the same remote function whenever possible so the ratio cancels host variance.
A same-process `torch.mm` baseline can still enter a slower state during a sweep, so use agreement across independent runs rather than one packet for gate decisions.

Independent Modal jobs may run concurrently because each receives separate resources, but every job must write to a distinct local log path.
With an empty Triton autotune cache, launch one warmup first when practical so later jobs reuse the selected configurations instead of autotuning independently.
Kill a crashed `modal run` before relaunching because a crash-looping app can continue writing to the old log.

See `findings/modal-vllm-run-anomalies.md` for retained vLLM results and examples of correlated or anomalous sweeps.

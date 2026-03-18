# Profiling

## Proton profiling

Documentation: https://github.com/triton-lang/triton/tree/main/third_party/proton

`speed_test.py --use_proton=True` enables Proton profiling with `mode="pcsampling"` (instruction sampling), which gives per-line runtime breakdowns for Triton kernels. Key API:

- `proton.start(name, hook="triton", backend="cupti", mode="pcsampling")` — initialize profiling. `mode="pcsampling"` enables PC sampling for per-line stats (~20x end-to-end overhead, but per-kernel overhead is negligible).
- `proton.scope(name)` — context manager to annotate regions (warmup, timing, etc.).
- `proton.finalize()` — flush and write profile data.
- `proton-viewer` CLI to render `.hatchet` files as trees.

**Known issue**: `mode="pcsampling"` segfaults when non-Triton CUDA kernels (e.g. `torch.gather`) are launched during profiling. This affects the Helion barrier kernel which calls `torch.gather` in stage 2. Workaround: use `--name fused-triton` to profile only the hand-written Triton kernel, or omit `mode="pcsampling"` (loses per-line granularity).

**CUDA 13+ CUPTI compatibility**: Triton 3.6.0 bundles CUPTI 2025.1.1 which segfaults in `cuptiPCSamplingEnable` → `NVPW_CUDA_LoadDriver` on CUDA 13+ drivers. Fix: set `TRITON_CUPTI_LIB_PATH` to a directory containing a driver-compatible `libcupti.so` (e.g. `/usr/local/cuda-13.1/targets/x86_64-linux/lib`). This env var tells Proton where to `dlopen` CUPTI from. Note: this is a **directory** path, not a file path. The Makefile's `proton-profile` target sets this automatically.

**`--bench_fn=own` required for Proton**: The default `fi-cupti` benchmark path (FlashInfer's `bench_gpu_time`) does not call `setup_proton()` / `proton.finalize()`, so no `.hatchet` file is produced. The `own` benchmark function is the one wired to Proton.

**Pitfall: Proton inflates per-launch overhead.** Proton adds fixed instrumentation cost per kernel launch. When comparing approaches with different numbers of launches (e.g. 1 vs 4), the wall-clock difference under Proton is misleading. For example, the barrier vs two-stage comparison showed a ~5ms gap under Proton that doesn't exist in uninstrumented runs (~0.01ms real overhead). Always cross-reference Proton wall-clock with `speed_test.py --use_proton=False`.

## Proton intra-kernel profiling (TTGIR override)

DSL-level scopes (`pl.enter_scope`/`pl.exit_scope`) are disabled inside persistent kernel loops by design (the compiler may hoist them out). The workaround is to inject `proton.record` statements at the TTGIR level after compilation. Confirmed by Triton engineer (Corbin Robeck).

### Workflow

```bash
make proton-profile N_HIDDEN_STATES=1    # dump → inject → run, prints breakdown table
make sweep-bsz-proton                    # runs proton-profile per bsz (1,4,16,64,128,256)
```

Three steps (individual targets: `proton-dump-ttgir`, `proton-inject`, `proton-run`):

1. `dump_ttgir.sh` runs the kernel with `TRITON_DUMP_DIR` to capture TTGIR.
2. `insert_proton_records.py` pattern-matches the TTGIR and injects `proton.record start/end` pairs for six scopes.
3. The kernel runs again with `TRITON_KERNEL_OVERRIDE=1` so the instrumented TTGIR is loaded.

### Scopes

The persistent kernel fuses the D-loop and tile loop into one `scf.for`. Every 128 iterations, an `scf.if` epilogue fires.

| Scope | What | Fires |
|-------|------|-------|
| kernel | Full kernel (tt.func → tt.return) | once |
| setup | Temperature load, grid dims, TMA descriptors, first prefetch | once |
| mask | V-masking and temperature scaling | per tile |
| tile-mgmt | Next-tile coordinate computation (swizzle grouping) | per tile |
| sample | Gumbel noise + argmax reduce | per tile |
| store | Global index offset, address computation, tt.store | per tile |

Matmul time = kernel - setup - mask - tile-mgmt - sample - store.

### Key constraints

- **No per-chunk matmul scope.** The D-loop is fused (not unrolled), so there's one `tt.dot` in the TTGIR. A scope around it fires 128x per tile, overflowing the buffer. Proton also rejects start/end spanning different loop iterations.
- **Shared buffer overflow at high bsz.** Each CTA generates 4 + 8/tile events. At bsz>=256 (~58 tiles/CTA → ~468 events), the 256-slot shared buffer overflows, dropping kernel/setup scopes. Use `BUFFER_TYPE.GLOBAL` (HBM) for those.
- **HBM vs SMEM buffer.** Comparison shows identical ratios (within noise), so HBM write overhead is negligible with few events per tile.
- **Warp sampling.** `SAMPLING_STRATEGY.SELECTIVE` with `sampling_options="0"` profiles only warp 0.
- **Inductor conflict.** `proton.start(backend="instrumentation")` is process-global and breaks `torch.compile`-generated kernels. `proton_profile.py` calls the Triton kernel directly, skipping `_local_reduce`.
- **Insertion ordering.** When multiple `proton.record` ops target the same line index, the sort key in `insert_proton_records.py` ensures `end` appears before `start` and `kernel` is the outermost scope.

### Output

`parse_proton_intrakernel.py` reads the chrome trace and prints a markdown breakdown table. It can also be used as a library (`parse_chrome_trace`, `trace_phase_pcts`) by the batch-size sweep plotting script.

## NCU (Nsight Compute) batch-size sweep

`benchmarking/parse_ncu_sweep.py` parses per-kernel GPU time from NCU CSV exports across batch sizes. It expects a directory layout like:

```
<dir>/bsz1/fused-triton.txt
<dir>/bsz1/naive-compiled.txt
<dir>/bsz4/fused-triton.txt
...
```

Each `.txt` file is NCU output with `--csv --page raw` containing `gpu__time_duration.sum`. Lines starting with `"` are CSV rows; others are NCU log messages.

**Method file names** (mapped to display labels in `METHOD_FILES`): `fused-triton.txt`, `naive-pt.txt`, `naive-compiled.txt`, `flashinfer:sampling_from_logits.txt` → `fi-sample`, `flashinfer:top_k_top_p_sampling_from_logits.txt` → `fi-topkp`.

**Data locations**:
- `benchmarking/profiles/sweeps/bsz/ncu-txt/tp1/case-small/` — tp1 data (RTX 3090, but files lack CSV metrics, only log output)
- `benchmarking/profiles/sweeps/bsz/ncu-txt/tp2/case-small/` — tp2 data (has valid CSV data)

**Usage**:
```bash
python benchmarking/parse_ncu_sweep.py --dir benchmarking/profiles/sweeps/bsz/ncu-txt/tp2/case-small
# Or via Makefile:
make parse-sweep-ncu N_PROCS=2 CASE=small
```

**Ruff pitfall**: `first_baseline` variable looks unused (F841) but is referenced via pandas `@first_baseline` in `.query()`. Suppressed with `# noqa: F841`.

## Nsight Systems (nsys) profiling

### Setup

The Brev image ships with nsys 2021.3.3 (CUDA 11.5 era) which is **too old for H100**. Install a modern version via apt:

```bash
sudo apt-get install -y nsight-systems-2025.5.2
```

Binary location: `/opt/nvidia/nsight-systems/2025.5.2/bin/nsys` (also symlinked to `/usr/local/bin/nsys` via alternatives).

### vLLM + FMMS profiling (`benchmarking/vllm/Makefile`)

Single-model targets: `nsys-baseline`, `nsys-fmms-triton`, `nsys-fmms-flashinfer`, `nsys-all`.
Comparison targets (baseline + fmms-triton at high concurrency): `nsys-compare-qwen3-8b`, `nsys-compare-qwen3-32b`.
Individual sub-targets (e.g. `nsys-compare-qwen3-32b-baseline`) also work standalone.

```bash
make -C benchmarking/vllm nsys-compare-qwen3-32b
```

**Architecture**: Uses `vllm bench sweep serve` to manage the full server lifecycle (start, health check with 1200s timeout, bench, shutdown). The serve-cmd wraps `vllm serve` under `nsys profile --capture-range=cudaProfilerApi`, and `--profiler-config.profiler=cuda` makes vLLM call `cudaProfilerStart()`/`cudaProfilerStop()` via the `/start_profile` and `/stop_profile` endpoints. nsys only records between these calls, skipping model load, torch.compile, CUDA graph capture, and warmup.

**Bench params**: `nsys-bench-params.json` defines two iterations per server session:
1. `{"_benchmark_name": "warmup"}` (empty config, no `--profile`, not recorded by nsys)
2. `{"_benchmark_name": "profiled", "profile": true}` (adds `--profile`, recorded by nsys)

The `--profile` flag on `vllm bench serve` automatically calls `/start_profile` before the benchmark and `/stop_profile` after. The warmup config is empty (not `"profile": false`) because `vllm bench serve` doesn't accept `--no-profile`.

`VLLM_NVTX_SCOPES_FOR_PROFILING=1` enables NVTX range annotations in the model runner (`preprocess`, `forward`, `postprocess`, `sample`), making it easy to locate the FMMS kernel on the timeline.

Output: `benchmarking/vllm/profiles/nsight/<GPU>/<model_slug>/`.

### Key pitfalls

- **gpt-oss-120b OOMs under nsys on H100.** The 120B model is tight on 81 GiB. nsys adds some memory overhead. At `--gpu-memory-utilization 0.90`, available KV cache was -0.63 GiB. Use a smaller model (e.g. `gpt-oss-20b`, which has 50+ GiB headroom) or reduce `gpu-memory-utilization` further for large models.
- **Do not use `sudo` with nsys in the serve-cmd.** `nsys profile --capture-range=cudaProfilerApi` works without root (only CPU profiling needs root). Using `sudo` creates a separate process tree that `vllm bench sweep serve` cannot kill, causing the server to linger and OOM the next run. Environment variables (like `VLLM_USE_FMMS_SAMPLER`) are passed via the `env` prefix in the serve-cmd without needing sudo.
- **`pkill -f` can kill the make process.** Never use `pkill -f "bin/vllm serve"` in a Makefile recipe. The recipe shell's cmdline (passed to `/bin/sh -c '...'`) contains the full recipe text, which matches the pattern. This kills the recipe shell and terminates make. Use `vllm bench sweep serve` instead, which manages server lifecycle internally.
- **`BENCH_FLAGS` vs `BENCH_DATASET_FLAGS`**: `BENCH_FLAGS` includes `--hf-output-len 256` for regular benchmarks. The nsys recipe uses `BENCH_DATASET_FLAGS` (without `--hf-output-len`) so that `NSYS_*_FLAGS` can set `--hf-output-len 10` without duplication. vLLM warns on duplicate keys but uses the last value.
- **`vllm bench sweep serve` handles readiness.** Use `--server-ready-timeout 1200` (20 min) to accommodate large models with torch.compile cold starts. Never hand-roll health check loops.
- **nsys `--capture-range-end=stop` orphans vllm.** nsys exits after the capture range ends to generate the `.nsys-rep` report, but vllm keeps running. The sweep's `stop()` checks `server_process.poll()`, sees nsys already exited, and skips `killpg`. Fix: after the sweep, kill any process still on port 8000 with `kill $(lsof -ti :8000)`.
- **`--after-bench-cmd` is required.** The serve-cmd starts with `env ...` so the sweep can't auto-detect the server type for cache resets. Pass `--after-bench-cmd` explicitly.

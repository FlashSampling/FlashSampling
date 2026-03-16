# AGENTS.md

- the speed test runner is in `benchmarking/speed_test.py`, and the triton benchmark runner is in `benchmarking/triton_benchmark.py`. There are useful commands to run benchs in `./benchmarking/Makefile`.
- equivalent commands for modal can be found in `./Makefile`.
- to plot all plots use `make plot-all`.
- to test the distributed code works use `make pytest-distributed`.
- when benchmarking many combinations, don't run the bench in parallel, since they will contend for the same resources. Instead launch them sequentially. On modal, you can launch benchmarks in parallel, since each job should get its own resources. When launching many parallel Modal benchmarks with an empty Triton autotune cache, consider running a single warmup job first to populate the cache on the volume. Otherwise every parallel job will autotune independently, wasting GPU time and risking inconsistent config selection.
- use the hugging face cli when appropriate.
- The blog post is in `~/code/tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample/index.qmd`.
- The paper is in `~/code/papers/flashsampling-paper/`.

Development notes and lessons learned while building this project.

**Meta-rule: Continuously update this file.** After every task, write new insights, patterns, and lessons learned into this file. Proactively review and update outdated information — if a timeout was changed, a cache strategy was revised, or a workaround is no longer needed, update the relevant section. This file is the project's living knowledge base.

## Code style

- **Top-down structure**: Define high-level functions first, helpers below. A reader should encounter the main logic before the details it delegates to. Helper functions go **after** the function that calls them, not before.
- **Never introduce GPU-CPU synchronizations.** Operations like `tensor.item()`, `float(tensor)`, `tensor.cpu()`, or `print(tensor)` on CUDA tensors force the CPU to wait for all pending GPU work to finish, destroying pipeline parallelism. Pass scalar values as 0-d CUDA tensors instead of extracting Python floats. Both the Triton kernel (`tl.load(temperature_ptr)`) and the Helion kernel (`temperature: torch.Tensor`) accept 0-d tensors directly.
- **Always save logs to the output folder.** When running servers, benchmarks, or evals, pipe stdout/stderr to a log file in the results directory so logs are always accessible after the run. Never discard or hide process output.
- **Pandas style**: Always use pandas (or equivalent DataFrame library) for data analysis. Never write nested loops with manual data joins when a pandas-based solution exists. Use `.query()` for row filtering, never boolean indexing (`df[df["col"] == val]`). Use `.merge()` for joins. Use `.groupby().agg()` instead of manual loops over unique values. Use `.pivot()` / `.melt()` for reshaping. Use `pd.concat()` to build DataFrames, not list-of-dicts loops.

## Writing style (README, blog post, docs)

- Single author project. Never use "we". Prefer "I" + active voice, but use passive voice when it sounds more natural.
- One sentence per line in prose sections, to make git diffs cleaner.
- Don't write `torch.compile`-d or `torch.compiled` — say "torch compiled".
- Avoid jargon like "unfused" or "lean" when simpler words work ("baseline", "Gumbel-max kernel").
- When stating speedup ranges, verify them against the actual table data. Use "generally outperforms" rather than "always" when there are exceptions.
- Never use em dashes (—). Use periods, commas, or parentheses instead.

## Blog post

The blog post lives at `~/code/tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample/index.qmd` (Quarto format).
It should be kept in sync with the README benchmark numbers.
The blog uses both "large" (V=128,256, d=8,192) and "small" (V=151,936, d=4,096) configs, presented as the outermost tabset in the kernel benchmarks section.

### Quarto conventions

- **Panel tabsets**: `::: {.panel-tabset group="name"}` with `# Tab Name` headers. The `group=` attribute synchronizes tab selection across multiple tabsets with the same group name.
- **Nested tabsets**: Use `::::` (4 colons) for the outer tabset and `:::` (3 colons) for the inner tabset. Outer tabs use `#` headers, inner tabs use `##` headers. Example:

  ```markdown
  :::: {.panel-tabset group="baseline"}
  # vs PyTorch Compiled
  ::: {.panel-tabset group="gpu"}
  ## B300
  ![](imgs/relative-perf-vs-pytorch-b300.png)
  ## H100
  ![](imgs/relative-perf-vs-pytorch-h100.png)
  :::
  # vs FlashInfer
  ...
  ::::
  ```

- **Plots before tables**: Show the plot first, then the data table beneath it. This gives the reader the visual takeaway before the numbers.
- **GPU ordering**: B300, B200, H200, H100, A100 (strongest to weakest) in all tabsets and table rows.
- **Table precision**: At most 2 decimal places for all numeric values.
- **Images**: Blog images are stored in `~/code/tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample/imgs/` and referenced as `![](imgs/filename.png)`. Copy from `benchmarking/modal-results/` when updating.
- **TODO section**: A commented-out HTML section (`<!-- ... -->`) near the top of the blog post tracks planned improvements. When a TODO is completed, remove it from the list entirely (don't strike it through).
- **Copying plots to the blog**: `make -C ~/code/tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample copy-imgs` copies all benchmark plots from `benchmarking/modal-results/` into the blog's `imgs/` directory. Run this after regenerating any plots.
- **Color palette**: FMMS is bold red (`#d62728`), baselines are gray/blue. Defined in `PROVIDER_COLORS` in `benchmarking/plot-triton-bench.py` and `VARIANT_COLORS` in `benchmarking/vllm/plot_tpot.py`. Both scripts use the same red for FMMS.

## Development environment

- Use the `.venv` in the repo root (not system Python). Run tests/scripts with `.venv/bin/python` or `.venv/bin/pytest`.
- **Save all learnings in this file (`CLAUDE.md`), not in `~/.claude/` MEMORY.md.** The `~/.claude/` directory is local to the server and will be lost when switching machines. This file is checked into git and travels with the code.

### NVIDIA Brev machine quirks

The Brev cloud GPU environment (shadeform) has several non-standard behaviors:

- **`$HOME` is unset** in non-login shells. Always pass `HOME=/home/shadeform` explicitly when running `make` or scripts that depend on `~` expansion. The Makefile's `$(HOME)` resolves to empty string otherwise.
- **Single global venv at `/home/shadeform/.venv/`**, not per-project. Both vLLM and fused-mm-sampling are installed there. The project's `.venv/` (referenced in the Makefile as `$(HOME)/code/fused-mm-sample/.venv/`) and vLLM's `venv/` (`$(HOME)/code/vllm/venv/`) do not exist.
- **vLLM binary**: `/home/shadeform/.venv/bin/vllm` (not `~/code/vllm/venv/bin/vllm`).
- **Python**: `/home/shadeform/.venv/bin/python` (Python 3.10.12).
- **GPU**: 1x NVIDIA H100 PCIe, 81,559 MiB VRAM, CUDA 13.0.
- **`datasets` / `pyarrow` conflict**: The pre-installed `datasets==2.14.4` is incompatible with `pyarrow==23.0.0` (`pa.PyExtensionType` was removed). Fix: `pip install --upgrade datasets` (upgrades to 4.5.0+).
- **HuggingFace**: Not logged in by default. Set `HF_TOKEN` env var for gated models.
- **Pip cache**: `/ephemeral/cache/pip` has wrong permissions; pip disables cache automatically (harmless warning).
- **Makefile portability**: Both `benchmarking/Makefile` and `benchmarking/vllm/Makefile` use `$(shell which python)` / `$(shell which vllm)` to discover binaries dynamically. No hard-coded paths — just activate the correct venv before running `make`. Example:
  ```bash
  HOME=/home/shadeform make -C benchmarking/vllm quick \
    MODEL=openai/gpt-oss-120b \
    HF_TOKEN=<token>
  ```

### CUDA toolkit installation on Brev (H100)

The Brev image ships with the NVIDIA driver (CUDA runtime 13.0) but **no nvcc** by default. Several components require nvcc for JIT compilation:

- **`fused-cuda` provider**: Uses `torch.utils.cpp_extension.load()` to JIT-compile a CUDA C++ extension. Needs nvcc + a compatible C++ compiler (g++-12).
- **flashinfer**: JIT-compiles CUDA kernels on first use. Requires nvcc with sm_90a support.
- **`tvm_ffi`**: The `_optional_torch_c_dlpack.py` module JIT-compiles a helper shared library. Non-fatal if it fails (just a warning).

**Installation steps:**

1. The CUDA 12.2 local repo deb is pre-cached at `/var/cuda-repo-ubuntu2204-12-2-local/`. Install from there:
   ```bash
   sudo dpkg -i /var/cuda-repo-ubuntu2204-12-2-local/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get install -y cuda-toolkit-12-2
   ```
2. Install a compatible C++ compiler:
   ```bash
   sudo apt-get install -y g++-12
   ```
3. Set environment variables:
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.2
   export PATH=$CUDA_HOME/bin:$PATH
   ```

**Key pitfalls:**
- The Brev image may have CUDA 11.5 nvcc pre-installed (`/usr/local/cuda-11.5/bin/nvcc`). This is **too old for H100** (doesn't support `compute_90a` / `sm_90a`). You must use CUDA 12.0+ for H100.
- `tests/conftest.py` auto-discovers CUDA_HOME by searching `/usr/local/cuda-*` (preferring highest version) and validates that nvcc supports the current GPU's compute capability. If validation fails, it raises a clear error message.
- After installing a new CUDA toolkit, delete stale JIT caches: `rm -rf /ephemeral/cache/torch_extensions/` to force recompilation with the new nvcc.

## Triton TMA (Tensor Memory Access) pitfalls

TMA uses `tl.make_tensor_descriptor` / `desc.load()` / `desc.store()` for hardware-accelerated memory access on H100. Three hard-won lessons:

### 1. Innermost dimension must be aligned to 16 bytes

TMA descriptors require the **innermost (stride-1) dimension** to be a multiple of 16 bytes. For bfloat16 (2 bytes/element), that means **multiples of 8 elements**. Non-aligned dimensions cause **silent data corruption** — no error, just wrong results.

```
K=304 (304 % 8 == 0) → PASS
K=300 (300 % 8 == 4) → FAIL, max_err=92.0
N=200 (200 % 8 == 0) → PASS
N=33  (33 % 8 == 1)  → FAIL, max_err=34.75
```

**Fix:** Pad tensors in the Python wrapper before passing to the kernel. Zero-padding doesn't affect matmul results. See `_tma_pad()` in `tl_matmul.py`. After the kernel, slice output back to the original dimensions.

### 2. `tl.dot(a, b.T)` does NOT work with TMA-loaded blocks

`.T` only swaps the logical view without rearranging shared memory layout. Tensor core MMA instructions depend on physical (row-major) layout, so the dot product produces wrong results. You must pre-transpose the matrix in the wrapper to make it physically contiguous in the layout the kernel expects.

### 3. Triton enforces `strides[-1] == 1`

You cannot describe a transpose via TMA strides — Triton's `semantic.py` checks that the last stride is 1 and raises `CompilationError` otherwise. The only option is to pre-transpose and make the matrix contiguous in the desired layout.

## Findings

The `findings/` directory contains detailed write-ups of bugs, workarounds, and design decisions discovered during development:

- `upcasting-before-softmax.md` — `torch.multinomial` produces wrong distributions with bfloat16. Fix: upcast to float32 before softmax.
- `helion-hl-rand-specialize-1-bug.md` — `hl.rand` crashes when a dimension is `hl.specialize(1)`. Includes root cause analysis, in-place fix, and minimal reproduction.
- `helion-barrier-single-kernel.md` — Merging stage 2 into the Helion kernel with `hl.barrier()`. Eliminates host-side reduction, reduces kernel launches from 3 to 1. Rigorous benchmarking shows barrier is ~3% slower at H=1 (host overhead is negligible). Barrier code is on the `barrier-kernel` branch.
- `rtx3090-barrier-comparison/` — Raw benchmark results (speed test, proton, NCU) for barrier vs two-stage on RTX 3090.
- `fused-top-k-top-p-feasibility.md` — Analysis of fusing top-k/top-p into the FMMS kernel. Top-k is feasible (tile-local top-k + merge); top-p is not directly fusible (needs global softmax + sorted cumsum). Practical path: fuse top-k, apply top-p on survivors post-kernel.
- `arithmetic-intensity-decode-matmul.md` — The decode matmul has arithmetic intensity ≈ H (batch size). Memory-bound up to H≈295 on H100 (BF16), H≈152 on RTX 3090. Includes ops:byte ratio derivation and data sources.
- `lm-head-configurations.md` — Survey of LM head shapes (vocab_size, hidden_size) across popular LLMs. Conclusion: vocab sizes cluster around 128K-152K; hidden_size is the real variable. Two benchmark groups: small (d=4,096) and large (d=8,192).
- `qwen3-8b-tpot-gap-at-high-concurrency.md` — Unexplained 29% TPOT improvement at concurrency 256 for Qwen3-8B on B200, despite FMMS being 18% slower in kernel microbenchmarks at that batch size. Hypotheses point to vLLM sampling code path overhead (GPU-CPU syncs, extra kernel launches, memory allocation). Proposed investigation: nsys profiling on Modal.
- `argsort-topk-complexity.md` — Why the fused top-k kernel uses a custom argsort (Triton has no `tl.argsort`; `tl.topk` returns values only). Complexity analysis shows that for our parameters (BLOCK_SIZE_V=128, top_k=20 → effective k=32), `tl.topk` saves only 1 sequential round vs full sort (4% latency reduction). Upstream Triton maintainers have declined to add argsort/topk-with-indices to the standard library.

## Architecture

- **Weights**: `[V, D]`, **hidden_states**: `[H, D]` everywhere in public APIs.
- The Helion kernel internally uses `hidden_states` as `[D, H]` (transposed) for matmul efficiency. The wrapper handles the transpose.
- All sampler variants are registered in `get_sampler()` in `core.py` via a match/case. New samplers only need a case there.
- The `Sampler` Protocol requires `prepare()` and `sample(**kwargs)`. Wrap simple callables with `SimpleSampler`.
- **Qitra** (`src/fused_mm_sampling/qitra.py`): Vendored from vLLM. Sort-free top-k/top-p Triton kernel based pivots (it does not sample tough). Used via the `pt-qitra` provider.

## Helion kernel pitfalls

Helion has an API reference: https://helionlang.com/api/index.html

and also reference examples: https://helionlang.com/examples/index.html

### `torch.argmax()` returns global indices

Inside a Helion kernel, `torch.argmax(tensor, dim=0)` returns **global** indices, not tile-local ones. The generated Triton code uses `triton_helpers.max_with_index` with the tile's global `indices_0` offset baked in. **Do NOT add `tile_v.begin`** — that double-counts the offset.

```python
# WRONG — double counts offset
new_max_idx_local = torch.argmax(summed, dim=0)
new_max_idx_global = tile_v.begin + new_max_idx_local  # BUG

# CORRECT — argmax already returns global indices
new_max_idx = torch.argmax(summed, dim=0)
```

### `for tile in hl.tile(N)` is parallel, not sequential

Each tile becomes a separate GPU program (thread block) that runs in parallel. You **cannot** do cross-tile communication via shared tensors — it's a race condition. This manifests as correct-looking results for some vocab sizes (e.g. 256, evenly divisible) but broken distributions for others (e.g. 100).

**Fix**: Use `hl.barrier()` to synchronize stages within a single kernel:
1. **Stage 1**: Each `(V, H)` tile writes its local max/argmax to `tile_maxs[tile_v.id, :]`.
2. `hl.barrier()` — grid-wide sync.
3. **Stage 2**: Reduce across tiles with `argmax` + `gather` inside the same kernel.

This eliminates Python-side tensor allocations and separate kernel launches for the reduction. However, rigorous benchmarking (25 warmup + 100 runs) shows the barrier version is ~3% **slower** at H=1 on RTX 3090 (2.38ms vs 2.32ms) due to cooperative launch constraints (`num_stages=1`, persistent scheduling, barrier sync stalls). The host-side overhead it eliminates is only ~0.01ms — negligible. The ~5ms gap seen under Proton profiling is an instrumentation artifact (Proton adds fixed overhead per kernel launch). See `findings/helion-barrier-single-kernel.md`.

### Tensor allocations inside kernels trigger warnings

`TensorOperationInWrapper` warning fires for tensor ops outside `hl.tile` loops. Allocate output buffers in the Python wrapper and pass them as kernel arguments instead.

### Advanced indexing does not work for gather

Helion interprets `tensor[idx_tensor, tile_var]` as a Cartesian product (producing a higher-rank result), not element-wise gather. Use `torch.gather` instead:

```python
# WRONG — Cartesian product, produces 2D
out[tile_h] = tile_max_idxs[best_tile, tile_h]  # RankMismatch error

# CORRECT — element-wise gather
out[tile_h] = torch.gather(
    tile_max_idxs[:, tile_h], 0, best_tile.unsqueeze(0)
).squeeze(0)
```

### Random number generation

Use `hl.rand([tile_v, n], seed=seed)` — not `torch.rand` or `torch.rand_like`. The `hl.rand` API uses Philox PRNG with proper per-tile offsets. Historical issues #1041 and #1309 are fixed in current Helion.

**Bug: `hl.rand` crashes when a dimension is `hl.specialize(1)`**. The `_rand_codegen` in `random_ops.py` tries to look up a block ID for every dimension, but a specialized size-1 dimension has no associated tile loop. `hl.zeros`/`hl.full` don't have this problem because they only need the shape, not index variables. Fix: in `_rand_codegen` (and `_randint_codegen`), when `size == 1`, use `tl.full([1], 0, tl.int32)` as the index var and `"1"` as the size name instead of trying to allocate a reduction dimension. We applied this fix in-place in `.venv/lib/python3.12/site-packages/helion/language/random_ops.py` — it will be lost on reinstall. Filed upstream: https://github.com/pytorch/helion/issues/1397

### Autotuning

- `autotune_effort`: `"none"` / `"quick"` / `"full"`. Controlled via `HELION_AUTOTUNE_EFFORT` env var (default: `"quick"`). Tests set it to `"none"` for speed.
- `LocalAutotuneCache` caches best config per GPU on disk. Cache dir set to `helion-cache/` in the repo root via `HELION_CACHE_DIR` env var (gitignored). Different GPUs autotune independently.
- First run with a new specialization key (e.g. new `n_hidden_states` value via `hl.specialize`) triggers autotuning (~3 min for `"full"`). Subsequent runs use the cache instantly.
- Set `HELION_AUTOTUNE_ACCURACY_CHECK=0` for stochastic kernels (the kernel output changes each run, so accuracy checks would always fail).
- To force re-tuning: delete the cache dir or set `HELION_SKIP_CACHE=1`.

### Performance: barrier kernel vs two-stage

Rigorous benchmarking (25 warmup + 100 runs, RTX 3090, V=128K, D=8192, H=1) shows the **two-stage version is ~3% faster** (2.32ms vs 2.38ms median). The host-side overhead eliminated by the barrier (tensor alloc + 3 auxiliary kernel launches) is only ~0.01ms. The barrier kernel pays for cooperative launch constraints: `num_stages=1` (no pipelining), 164 persistent blocks vs 1,002 one-shot blocks, and 52% of CPI spent on barrier sync stalls.

Initial Proton profiling showed a ~5ms wall-clock advantage for the barrier version — this is a **profiling artifact**. Proton adds fixed instrumentation overhead per kernel launch, making the 4-launch two-stage appear much slower than the 1-launch barrier. Always cross-reference Proton with uninstrumented speed tests.

See `findings/helion-barrier-single-kernel.md` for full NCU and Proton analysis, and `findings/rtx3090-barrier-comparison/` for raw data.

## `torch.multinomial` and bfloat16

`torch.multinomial` produces incorrect sampling distributions when given bfloat16 probabilities. Fix: upcast to float32 before softmax:

```python
probs = (logits.float() / temperature).softmax(dim=1)
```

See `findings/upcasting-before-softmax.md` for details.

## Testing

- `test_sampling_distribution` uses a chi-squared goodness-of-fit test comparing empirical samples against theoretical softmax probabilities.
- Parametrized over all providers, multiple vocab sizes (100, 256), and n_hidden_states (1, 2) to catch tile-boundary and dimension-edge-case bugs.
- Bins with expected count < 5 are excluded (chi-squared assumption). Expected counts are rescaled to match observed totals.
- `make_synthetic_inputs()` in `src/fused_mm_sampling/testing.py` constructs weights/hidden_states that produce known logit vectors (ascending and descending) via SVD + pseudoinverse.

## Naming conventions

The algorithm is called **FMMS** (Fused Matrix Multiplication & Sampling). Provider display names in benchmarks follow the pattern:
- `"FMMS (Triton)"` — hand-written Triton kernel
- `"FMMS (Helion)"` — Helion kernel
- `"FMMS (Triton NoNoise)"` — Triton kernel without Gumbel noise (for profiling)

These names are defined in `provider_names` in `src/fused_mm_sampling/bench/triton_benchmark.py` and used in plots, CSVs, and the README.

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

## Modal profiles

Two Modal workspaces are configured. Switch with `modal profile activate <name>`.

- **`tomasruizt`** (personal): Used for vllm-bench runs on gpt-oss-120b and Qwen3-1.7B. The `fused-mm-sample` volume here holds these results.
- **`lmu-css`** (default): Used for triton-bench runs and vllm-bench runs on Qwen3-8B. The `fused-mm-sample` volume here holds these results.

Check the active profile with `modal profile list` (the `•` marker shows the active one). When downloading results with `make modal-get-results-*`, ensure the correct profile is active or the volume lookup will fail silently (no matching directory).

### Modal volume management

The `fused-mm-sample` volume stores benchmark results, model caches, and torch.compile caches. Useful commands:

```bash
modal volume ls fused-mm-sample                     # list root
modal volume ls fused-mm-sample triton-bench-b200   # list subdirectory
modal volume rm fused-mm-sample <path> -r           # delete recursively
modal volume get fused-mm-sample <path> <local_dir> # download to local
```

**Paths with special characters** (e.g. `triton-bench-h100!`): use double quotes around the path argument to prevent shell expansion.

## Modal benchmarking (triton-bench)

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

## vLLM integration

The FMMS sampler is integrated into vLLM on the `feature/fmms-sampler` branch in `~/code/vllm`. Key files:

- `vllm/v1/sample/fmms_sampler.py` — thin wrapper adapting FMMS kernel to vLLM's `SamplerOutput`
- `vllm/envs.py` — `VLLM_USE_FMMS_SAMPLER` and `VLLM_FMMS_PROVIDER` env vars
- `vllm/v1/worker/gpu_model_runner.py` — calls `FMMSSampler` in `sample_tokens()` when enabled

### Benchmarking

End-to-end vLLM benchmarks live in `benchmarking/vllm/`. Key files:

- `Makefile` — `make all` (full sweep, 3 runs) and `make quick` (smoke test, 1 run, `--enforce-eager`). Supports `MODEL=` override for different models.
- `bench-params.json` / `quick-bench-params.json` — sweep parameters (concurrency levels, num_prompts, request_rate)
- `collect_results.py` — reads `summary.csv` from each variant's latest timestamped run, prints summary table (last run only) and per-run breakdown. Usage: `python collect_results.py <model_dir>`
- `parse_engine_stats.py` — extracts KV cache occupancy, running/waiting request counts from `sweep.log` files. Parses the periodic engine stats lines emitted every 10s by vLLM. Usage: `python parse_engine_stats.py <sweep.log> [--by-concurrency]`. The `--by-concurrency` flag aggregates across runs per concurrency level. Useful for diagnosing KV cache pressure at high batch sizes.
- `plot_tpot.py` — plots median TPOT vs concurrency for all models, using `sns.lineplot` with shading for run-to-run variance. Output: `tpot_vs_concurrency.png`.
- Results are organized as `<model_slug>/baseline/`, `<model_slug>/fmms-triton/`, `<model_slug>/fmms-flashinfer/`

### Baseline sampler is plain PyTorch, not flashinfer

vLLM's default (baseline) sampling path uses plain PyTorch ops (softmax + multinomial), **not** a flashinfer sampling kernel. In nsys traces, the baseline `sample` scope shows `compute_logits` (lm_head matmul) followed by PyTorch ops, not a fused flashinfer call.

**TODO**: Add an FMMS baseline variant that uses the `naive-compiled` provider (compiled PyTorch matmul + sampling, unfused). This gives a fairer apples-to-apples comparison for both nsys profiling and TPOT benchmarks — same code path, same overhead, only the fusion differs. Currently the baseline uses vLLM's native sampler which has a different code path entirely.

### `.item()` CPU-GPU synchronization bug

`temperature[0].item()` in `fmms_sampler.py` caused a CPU-GPU sync on every decode step. At concurrency 32, TPOT regressed from 9ms to 18ms. Fix: use `temperature[0]` (scalar tensor) instead. This applies broadly — never call `.item()`, `float()`, `.cpu()`, or `print()` on GPU tensors in the hot path.

### Triton autotuning at runtime

The Triton kernel's `@triton.autotune` originally had `n_hidden_states` in its `key=` parameter. Every unique batch size triggered autotuning (benchmarking all configs). In vLLM, high concurrency produces many unique batch sizes (33, 34, ..., 256), each causing an autotune run **during the benchmark**. This inflated TPOT by 2-10x at concurrency 32+.

**Fix applied**: Replaced `n_hidden_states` with `BLOCK_SIZE_H` in the autotune `key=`, and changed `n_hidden_states` from `tl.constexpr` to a regular runtime int in the kernel signature. `BLOCK_SIZE_H` has only 3 possible values (16, 32, 64), so autotuning runs at most 3 times per (V, D) combination instead of once per unique batch size. All three uses of `n_hidden_states` inside the kernel (`tl.cdiv`, comparison, arithmetic) work fine with runtime values.

## Modal benchmarking (vllm-bench)

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

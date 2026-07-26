# AGENTS.md

- the speed test runner is in `benchmarking/speed_test.py`, and the triton benchmark runner is in `benchmarking/triton_benchmark.py`. There are useful commands to run benchs in `./benchmarking/Makefile`.
- equivalent commands for modal can be found in `./Makefile`.
- The shared `Args` class for both speed_test and triton_benchmark lives in `triton_benchmark_lib.py`. speed_test's `CliArgs` overrides defaults (`case="small"`, `n_hidden_states=1`).
- to plot all plots use `make plot-all`.
- to test the distributed code works use `make modal-pytest-distributed` (requires >= 2 GPUs with NVLink for symmetric memory). `make pytest-distributed` also works but auto-skips on single-GPU machines.
- when benchmarking many combinations, don't run the bench in parallel, since they will contend for the same resources. Instead launch them sequentially. On modal, you can launch benchmarks in parallel, since each job should get its own resources. When launching many parallel Modal benchmarks with an empty Triton autotune cache, consider running a single warmup job first to populate the cache on the volume. Otherwise every parallel job will autotune independently, wasting GPU time and risking inconsistent config selection.
- available CLIs: hugging face, brev, github
- The blog post is in `~/code/tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample/index.qmd`.
- The paper is in `~/code/papers/flashsampling-paper/`.
- The FMMS Triton kernel is in `src/fused_mm_sampling/core.py`.
- Generally speaking, do imports at the top of the file, not inside functions.
- Do no speculate blindly about why code is slow. Causal statements need to be backed by empirical evidence. Choose appropriate language to hedge, e.g. "Possibly", "Potentially", and point out what data would let us clear the uncertainty and make confident claims.
- **Don't block the user with long sleeps.** When waiting for a task, never `sleep` for 60s+ in a single foreground call — that freezes the conversation and prevents the user from steering. For tasks that may run >1 min, launch with `run_in_background=true` and wait for the completion notification. For shorter polling, use 5-15s sleeps and check in.

Development notes and lessons learned while building this project.

**Meta-rule: Continuously update this file.** After every task, write new insights, patterns, and lessons learned into this file. Proactively review and update outdated information — if a timeout was changed, a cache strategy was revised, or a workaround is no longer needed, update the relevant section. This file is the project's living knowledge base.

## Code style

- **Top-down structure**: Define high-level functions first, helpers below. A reader should encounter the main logic before the details it delegates to. Helper functions go **after** the function that calls them, not before.
- **Never introduce GPU-CPU synchronizations.** Operations like `tensor.item()`, `float(tensor)`, `tensor.cpu()`, or `print(tensor)` on CUDA tensors force the CPU to wait for all pending GPU work to finish, destroying pipeline parallelism. Pass scalar values as 0-d CUDA tensors instead of extracting Python floats. Both the Triton kernel (`tl.load(temperature_ptr)`) and the Helion kernel (`temperature: torch.Tensor`) accept 0-d tensors directly.
- **Cache CUDA device properties on hot paths.** Use `num_sms_cached()` in `core.py` instead of repeatedly calling `torch.cuda.get_device_properties(...).multi_processor_count` inside decode-time wrappers.
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
- **Keep Modal submission modules CPU-importable.** Import GPU-only runtime dependencies such as Torch, Triton, FlashInfer, and benchmark modules inside the remote function. Pass primitive serializable arguments from the local entrypoint and construct runtime argument objects inside the Modal container.
- Use `make modal-verify-correctness-tp1` to run the correctness checks on one Modal GPU. `modal-pytest-distributed` accepts `N_PROCS` for other world sizes. Both launch through `torchrun`, including TP1, so the worker can use `TPInfo.from_world()` with an initialized process group.
- Use `make modal-verify-correctness-large-vocab VOCAB_SIZE=... NUM_SAMPLES=... SAMPLES_PER_CALL=...` for the TP1 large-vocabulary chi-squared check. It batches sampling, accumulates counts on GPU, reports test statistics and coverage, and passes parameters to Modal as CLI options.
- At V=128,000 and 10M samples on B200, bfloat16 per-tile maxima caused measurable bias. Changing `maxs` to float32 passed the test (reduced chi-squared 0.99844, p=0.6503, 99.84% probability mass). The RNG now also uses separate sample streams and unique tile-element offsets to avoid collisions.
- Use `make modal-memory-traffic-all CASE=large N_HIDDEN_STATES=64` to profile FlashSampling and the three paper baselines concurrently on Modal. Each provider has its own results subdirectory containing `report.ncu-rep`, `traffic.csv`, `memory.json`, and `log.txt`; `parse-memory-traffic` aggregates completed artifacts with pandas.
- On B200 with the large case at B=1/64/256, FlashSampling used 0.05/0.77/2.97 MiB peak temporary memory, a 98.48-99.53% reduction against the three baselines. Its HBM-read reduction grew from 0.17-0.33% at B=1 to 4.33-26.52% at B=256, while its HBM-write reduction grew from 37.98-43.30% to 95.73-97.94%.
- Brev machine quirks and CUDA toolkit setup: see [docs/brev-environment.md](docs/brev-environment.md).

## Triton TMA (Tensor Memory Access) pitfalls

See [docs/triton-tma-pitfalls.md](docs/triton-tma-pitfalls.md). Key points: innermost dim must be 16-byte aligned, `tl.dot(a, b.T)` fails with TMA blocks, Triton enforces `strides[-1] == 1`.
`set_torch_allocator_for_tma_descriptors_cached()` is called by `fused_mm_sample_triton()`, so clients that use the sampler/wrapper APIs do not need to call it directly. Keep direct calls only for raw TMA kernel launch paths that bypass the wrapper.

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
- `tma-cache-modifiers.md` — Analysis of using L2 cache modifiers (`evict_first`/`evict_last`) for FMMS Triton kernel loads. Hidden states should be kept warm (reused across V tiles), weights should stream through (no reuse at low batch sizes). Conclusion: TMA `desc.load()` does not support cache modifiers (the PTX `cp.async.bulk.tensor` instruction lacks those fields), and switching to regular `tl.load()` to get them would lose TMA's async prefetch pipeline. The tile swizzling (GROUP_SIZE_V=4) already provides the main L2 benefit, and hidden states are too small relative to L2 (0.03% at H=1) to be evicted.
- `torch-compile-overhead-tp2.md` — torch.compile adds 0.05-0.13ms overhead at TP2 that hurts all baselines at low batch sizes (up to 1.43x for multinomial, 1.23x for FlashInfer on small config). FMMS is the exception: its compiled `_local_reduce`/`_stack_and_select_winner` are small enough that compile helps. The effect is weaker for large config. At H>=64 compile wins for all providers.
- `register-spilling-bsz256.md` — At H=256 the kernel spills 118 MB due to three [128, 64] f32 tensors being live simultaneously (persistent loop iter_arg + scaled logits + Gumbel noise). Fix: raise `maxnreg` from 128 to 255 (1.74x speedup on RTX 3090). Fusing noise into the matmul accumulator eliminates spilling but breaks D-loop software pipelining (30% regression). Datacenter GPUs keep `maxnreg=128` because warp specialization adds warps that exceed the register file at 255.
- `proton-scopes-persistent-kernel.md` — DSL-level Proton scopes don't work inside persistent kernels (by design). Solution: TTGIR-level injection via `insert_proton_records.py`. Six scopes (kernel, setup, mask, tile-mgmt, sample, store); matmul derived by subtraction. Includes buffer overflow constraints, warp sampling, HBM vs SMEM comparison, and per-bsz results showing sampling grows from 1% (bsz=1) to 23% (bsz=256) due to BLOCK_SIZE_H increase and register spilling.
- `tp2-collective-overhead.md` — FMMS TP2 collective overhead was ~0.12-0.20ms from NCCL latency. Fixed by allocating kernel outputs in symmetric memory (direct NVLink writes, no NCCL). Reduced H=1 latency from 0.304ms to 0.246ms (large) and 0.329ms to 0.254ms (small) on B200 x2.
- `tp2-dispatch-asymmetry.md` — One rank dispatches kernels 300-700us slower per iteration than the other. Which rank is slow varies by run. Likely caused by OS CPU scheduling noise on shared cloud hardware. NUMA binding reduces but does not eliminate the asymmetry (median 364us unbound vs 277us bound). Affects all providers (fused-triton, naive-compiled, flashinfer). mp.spawn and torchrun both exhibit the asymmetry. CPU sampling not available on Modal (gVisor blocks perf_event_open). `modal-nsys-profile` uses per-rank nsys via `benchmarking/nsys_wrapper.py` to capture both devices.
- `tma-store-blackwell-singleton-dims.md` — On B200 (sm_100, Triton 3.6), `tl.make_tensor_descriptor(...).store(...)` with a 3D `block_shape=[1, 1, BLOCK_SIZE_H]` (two singleton dims) silently no-ops most stores in a persistent loop: only ~2/1187 V-tile slots written, the rest left uninitialized. The same pattern works on Hopper and RTX 3090. Caused vLLM to crash with OOB token ids on the first decode step. Fix: drop TMA descriptors for the per-tile output stores entirely (they're tiny and TMA gives no bandwidth win below ~32 KB), use plain `tl.store` with computed offsets instead. Keep TMA on the matmul *load* descriptors. Not just int64 — bf16 is affected too. Microbench was a false negative because it only checked the gathered final id, not all maxs_idx slots; uninit memory on freshly-allocated CUDA pages is mostly zeros, which `0 <= x < V` happily accepts.
- `tp2-fullgraph-via-functional-collective.md` — TP>1 used `torch.compile(...)` (no fullgraph) for `sample_compiled`/`greedy_sample_compiled` because dynamo could not trace through the `dist.all_gather` call inside `_allgather_logits` (decorated with `@torch.compiler.disable`). Switched to `torch.distributed._functional_collectives.all_gather_tensor` (compile-friendly, returns `AsyncCollectiveTensor`), dropped the decorator, and collapsed both wrappers to a single fullgraph path. Also pulled `torch.manual_seed(seed)` out of the compiled body (dynamo skips `manual_seed`). On B200 TP2 small config (3+3 runs each side): `Multinomial Sampling (Compiled)` is ~28% faster at H=1-8, ~19% at H=32, ~12% at H=64, but ~5-6% slower at H=128/256 (real, not noise — std is 0.001-0.003 ms on both sides). FMMS is unchanged (uses `kraken_post_kernel_reduce`, not `_allgather_logits`).
- `2cta-mma-operand-swap-regression.md` — Attempt to enable Blackwell `tcgen05.mma cta_group::2` for the FMMS kernel. Direct `num_ctas=2` aborts in Triton's `ReduceOpToLLVM` because our `tl.argmax` over V crosses CTAs. Swapping operands (`tl.dot(hidden, weights.T)`) makes V the N axis and lets the reduce stay within one CTA, but the MMA M-dim then inherits H's variability via `BLOCK_SIZE_H`. On B200 TP1 the swap alone (no `num_ctas=2`) regresses FMMS by 6-7% at H≤16 and 13-23% at H=32 (M=32, non-native sm_100 bf16 shape, padded to M=64). H≥64 is flat. Net expected value of 2-CTA after the swap is approximately zero. Abandoned; transposed kernel preserved on `worktree-2cta-mma` branch as record.
- `b200-warp-spec-single-tile-crash.md` — On B200 / Triton 3.6, warp specialization crashes the `NVWSInsertAref` MLIR pass (`PassManager::run failed`) for small correctness-test shapes with `num_samples=10_000`. The original failure had one tile; a later V=512 failure had 2--4 tiles, so tile count alone does not characterize the trigger. Workaround: enable `WARP_SPECIALIZE` only when `num_samples == 1` (the production decode path), while multi-sample distribution tests use the non-WS lowering. Verified on Modal B200 TP1 across 30 distribution and 6 greedy cases.
- `tp-scaling-fast-pod-b200.md` — Apples-to-apples TP=2/4/8 scaling on fast-pod b200. Low-H gains stay clean (TP4≈0.7×TP2 small; TP8≈0.7×TP4 large). At high H on small, TP=8 *regresses* vs TP=4 (1.13-1.47×) because the fan-out symm-mem write cost scales O(world_size) and dominates once the matmul shrinks. Fast-pod hit-rate at TP=8 was 1/5 (vs 7/11 at TP=2); HGX runs hit the 1200s Modal timeout 2/5 times, never the fast pod.

## Architecture

- **Weights**: `[V, D]`, **hidden_states**: `[H, D]` everywhere in public APIs.
- The Helion kernel internally uses `hidden_states` as `[D, H]` (transposed) for matmul efficiency. The wrapper handles the transpose.
- All sampler variants are registered in `get_sampler()` in `core.py` via a match/case. New samplers only need a case there.
- The `Sampler` Protocol requires `prepare()` and `sample(**kwargs)`. Wrap simple callables with `SimpleSampler`.
- **Qitra** (`src/fused_mm_sampling/qitra.py`): Vendored from vLLM. Sort-free top-k/top-p Triton kernel based pivots (it does not sample tough). Used via the `pt-qitra` provider.

## Helion kernel pitfalls

See [docs/helion-pitfalls.md](docs/helion-pitfalls.md). Covers: argmax global indices, parallel tiles, tensor allocations, gather indexing, RNG, autotuning, barrier vs two-stage performance.

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

## Profiling (Proton, NCU, nsys)

See [docs/profiling.md](docs/profiling.md).

### Proton intra-kernel profiling (TTGIR override)

DSL-level scopes (`pl.enter_scope`) don't work in persistent kernels (compiler hoists them out of loops).
Instead, `insert_proton_records.py` injects `proton.record` ops directly into the TTGIR after compilation.
See `findings/proton-scopes-persistent-kernel.md` for full details.

Key files:
- `benchmarking/proton_profile.py` — standalone profiling script (calls kernel directly, skips `_local_reduce` to avoid inductor conflict)
- `benchmarking/insert_proton_records.py` — injects six scopes into TTGIR: kernel, setup, mask, tile-mgmt, sample, store
- `benchmarking/parse_proton_intrakernel.py` — parses chrome traces, derives matmul = kernel - setup - mask - tile-mgmt - sample - store
- `benchmarking/dump_ttgir.sh` — dumps TTGIR via `TRITON_DUMP_DIR`

Makefile targets: `make proton-profile` (all-in-one), `make sweep-bsz-proton` (per-bsz sweep).

Constraints:
- The persistent kernel's D-loop is fused (not unrolled), so per-chunk matmul scopes are impossible without overflowing the 128-slot shared buffer.
- At high bsz (>=256), the buffer overflows even with the current 6 scopes. Use `BUFFER_TYPE.GLOBAL` (HBM) for those. HBM vs SMEM gives identical ratios.
- `SAMPLING_STRATEGY.SELECTIVE` with `sampling_options="0"` profiles only warp 0 to reduce event count.
- When multiple proton.record ops share a line index, the sort key in `insert_proton_records.py` ensures correct nesting (end before start, kernel outermost).

## Symmetric memory TP reduction

`src/fused_mm_sampling/tensor_parallel_reduce.py` replaces the NCCL all_gather in the TP>1 code path with symmetric memory. Used automatically when `tp.size > 1`.

Flow: the kernel output buffers (`maxs`, `maxs_idx`) are allocated in symmetric memory via `get_symm_mem_workspace`, so the kernel's existing TMA stores write directly to NVLink-mapped addresses. After the kernel completes, a host-side barrier ensures all ranks' writes are visible. Each rank then reads all ranks' per-tile outputs from symmetric memory, runs `_local_reduce` per rank, and picks the global winner via `_stack_and_select_winner`.

Requires: NVLink-connected GPUs, PyTorch >= 2.6, CUDA >= 12.4. See `findings/tp2-collective-overhead.md` for motivation and analysis.

## Distributed process launching (torchrun vs mp.spawn)

`run_maybe_distributed()` in `src/fused_mm_sampling/tp_info.py` supports two backends:
- **torchrun** (preferred for profiling): Detected automatically via `RANK`/`WORLD_SIZE` env vars. Uses `init_method="env://"`. No parent process overhead.
- **mp.spawn** (fallback): Used when torchrun env vars are absent. Manual NUMA binding, `tcp://` init method, parent process polls child sentinels.

`modal-nsys-profile` uses torchrun for TP>1 runs, with per-rank nsys instances via `benchmarking/nsys_wrapper.py`. Each rank gets its own `.nsys-rep` file. This is necessary because nsys cannot capture both devices when wrapping torchrun from outside (the `--capture-range=cudaProfilerApi` only captures the first child process's CUDA context). The dispatch asymmetry persists with torchrun (see `findings/tp2-dispatch-asymmetry.md`), confirming it is not an mp.spawn artifact. Other distributed callers (triton benchmark, pytest) still use mp.spawn.

**Speed test modes**: `speed_test.py` has two separate code paths controlled by `--nsys_profile=true`:
- `benchmark()`: timing with CUDA events, no profiler API. Used for speed measurements.
- `nsys_profile()`: `cudaProfilerStart/Stop`, `dist.barrier()` for rank sync, NVTX ranges. Used for nsys capture. No timing events.

The `--nsys_profile` flag is a pydantic-settings `bool` field. On the CLI, pass `--nsys_profile=true` (not just `--nsys_profile`, which fails with "expected one argument").

## vLLM integration

See [docs/vllm-integration.md](docs/vllm-integration.md). Covers: sampler wrapper, env vars, local benchmarking, `.item()` sync bug, autotuning fix.

## Benchmark timing functions

Shared timing primitives live in `triton_benchmark_lib.py`:

- **`bench_cupti(fn, ...)`**: FlashInfer's CUPTI-based `bench_gpu_time`. Uses hardware counters. Adaptive iteration count for TP1, fixed counts for distributed (to avoid collective mismatches).
- **`bench_cuda_events(fn, ...)`**: CUDA event timing with L2 cache flushing via `create_l2_cache()`/`clear_l2_cache()`. Fixed iteration counts always.
- **`synchronize(is_distributed)`**: `dist.barrier()` for distributed, `torch.cuda.synchronize()` for TP1.

Both return `list[float]` (per-iteration times in ms). The `bench_fn` parameter (`"fi-cupti"` or `"own"`) selects which one to use. Empirical comparison (b200, h200, h100!, TP1) shows the two methods produce equivalent results (mean diff 1.46%, within noise). At TP2, `own` reports systematically higher latencies than `fi-cupti` (mean +7.3% on h100!), so the methods are not interchangeable for distributed runs.

### fi-cupti + TP2 SIGSEGV (non-deterministic)

`bench_fn=fi-cupti` with TP2 causes non-deterministic SIGSEGV crashes in the NCCL watchdog thread (`cudaSetDevice` inside `c10d::ProcessGroupNCCL::Watchdog`). Observed on b200 and h200, not on h100!. The crash occurs in `triton_benchmark` (which calls `bench_cupti` repeatedly across 9 batch sizes in the `perf_report` loop) but not in `speed_test` (which calls it once per provider). However, it also crashed with a single provider and all batch sizes, and succeeded on retry, so the trigger is non-deterministic. Possibly CUPTI instrumentation corrupts NCCL internal state during repeated init/teardown cycles within one `mp.spawn` process. Workaround: use `bench_fn=own` for TP2 benchmarks.

## Modal benchmarking

See [docs/modal-benchmarking.md](docs/modal-benchmarking.md). Covers: Modal profiles, volume management, triton-bench pipeline, vllm-bench pipeline, image build, caching.

### Directory structure

Modal triton-bench results are organized as: `modal-results/triton-bench/{bench_fn}/{gpu}/tp{N}/`. Custom plots go into `custom-plots/case-{small,large}/` subdirectories within each tp directory. The `BENCH_FN` make variable (default: `fi-cupti`) controls which timing method and directory to use.

### Makefile variable passing

Makefile variables use `:=` assignment, so environment variables do NOT override them. Always pass overrides as make arguments (`make FOO=bar target`), not env vars (`FOO=bar make target`). The `NAME` variable defaults to `default` (all providers). `Args.providers()` treats both `None` and `"default"` as the sentinel for `DEFAULT_PROVIDERS`.

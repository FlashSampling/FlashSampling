# AGENTS.md

- the speed test runner is in `benchmarking/speed_test.py`, and the triton benchmark runner is in `benchmarking/triton_benchmark.py`. There are useful commands to run benchs in `./benchmarking/Makefile`.
- equivalent commands for modal can be found in `./Makefile`.
- to plot all plots use `make plot-all`.
- to test the distributed code works use `make pytest-distributed`.
- when benchmarking many combinations, don't run the bench in parallel, since they will contend for the same resources. Instead launch them sequentially. On modal, you can launch benchmarks in parallel, since each job should get its own resources. When launching many parallel Modal benchmarks with an empty Triton autotune cache, consider running a single warmup job first to populate the cache on the volume. Otherwise every parallel job will autotune independently, wasting GPU time and risking inconsistent config selection.
- use the hugging face cli when appropriate.
- The blog post is in `~/code/tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample/index.qmd`.
- The paper is in `~/code/papers/flashsampling-paper/`.
- The FMMS Triton kernel is in `src/fused_mm_sampling/core.py`.

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
- Brev machine quirks and CUDA toolkit setup: see [docs/brev-environment.md](docs/brev-environment.md).

## Triton TMA (Tensor Memory Access) pitfalls

See [docs/triton-tma-pitfalls.md](docs/triton-tma-pitfalls.md). Key points: innermost dim must be 16-byte aligned, `tl.dot(a, b.T)` fails with TMA blocks, Triton enforces `strides[-1] == 1`.

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
- `register-spilling-bsz256.md` — At H=256 the kernel spills 118 MB due to three [128, 64] f32 tensors being live simultaneously (persistent loop iter_arg + scaled logits + Gumbel noise). Fix: raise `maxnreg` from 128 to 255 (1.74x speedup on RTX 3090). Fusing noise into the matmul accumulator eliminates spilling but breaks D-loop software pipelining (30% regression). Datacenter GPUs keep `maxnreg=128` because warp specialization adds warps that exceed the register file at 255.

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

## vLLM integration

See [docs/vllm-integration.md](docs/vllm-integration.md). Covers: sampler wrapper, env vars, local benchmarking, `.item()` sync bug, autotuning fix.

## Modal benchmarking

See [docs/modal-benchmarking.md](docs/modal-benchmarking.md). Covers: Modal profiles, volume management, triton-bench pipeline, vllm-bench pipeline, image build, caching.

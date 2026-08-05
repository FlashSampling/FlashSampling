# AGENTS.md

## Required startup reading

Before planning or implementation, read [loop-feedback.md](loop-feedback.md) completely and apply its checklist throughout the task.
This file is intentionally a compact startup index.
Load only the task-specific documents linked below instead of reading every historical finding.

## Documentation policy

- Keep `AGENTS.md` limited to rules and routing information that apply to most tasks.
- Put stable workflows and operational instructions in a focused file under `docs/`.
- Put measured results, investigations, rejected approaches, and design decisions in a focused file under `findings/`.
- Put the active CUTLASS state in `findings/cutlass/current-handoff.md`, not here.
- Update an existing focused document when one already covers the topic.
- Add new documents to the relevant index so future agents can discover them.
- Do not append run-specific results, benchmark packets, or temporary handoff notes to this file.
- `CLAUDE.md` is a symlink to this file.
  Do not put durable project knowledge in `~/.claude/MEMORY.md` because it does not travel with the repository.

## Task-specific reading

| Task area | Read first |
| --- | --- |
| Planning or implementation | [loop-feedback.md](loop-feedback.md) |
| Local benchmarks and timing | [docs/benchmarking.md](docs/benchmarking.md) |
| Modal benchmarks and caching | [docs/modal-benchmarking.md](docs/modal-benchmarking.md) |
| Testing and distribution checks | [docs/testing.md](docs/testing.md) |
| Tensor parallel and process launch | [docs/distributed-development.md](docs/distributed-development.md) |
| Proton, NCU, or nsys profiling | [docs/profiling.md](docs/profiling.md) |
| vLLM integration | [docs/vllm-integration.md](docs/vllm-integration.md) |
| Blog updates | [docs/blog-maintenance.md](docs/blog-maintenance.md) |
| Triton TMA work | [docs/triton-tma-pitfalls.md](docs/triton-tma-pitfalls.md) |
| Helion work | [docs/helion-pitfalls.md](docs/helion-pitfalls.md) |
| Brev setup | [docs/brev-environment.md](docs/brev-environment.md) |
| Historical or empirical context | [findings/README.md](findings/README.md) |
| CUTLASS work | [findings/cutlass/README.md](findings/cutlass/README.md) and [findings/cutlass/current-handoff.md](findings/cutlass/current-handoff.md) |
| CUTLASS source or template changes | [docs/cutlass-compilation.md](docs/cutlass-compilation.md) |
| CUTLASS experiment execution | [docs/modal-benchmarking.md#cutlass-experiment-development-driver](docs/modal-benchmarking.md#cutlass-experiment-development-driver) |
| CUTLASS development infrastructure | [findings/cutlass/25-development-infrastructure.md](findings/cutlass/25-development-infrastructure.md) |

## Project map

- The FMMS Triton kernel is in `src/fused_mm_sampling/core.py`.
- The speed-test runner is `benchmarking/speed_test.py`.
- The Triton benchmark runner is `benchmarking/triton_benchmark.py`.
- Their shared `Args` class lives in `src/fused_mm_sampling/bench/triton_benchmark_lib.py`.
- Local benchmark commands live in `benchmarking/Makefile`.
- Equivalent Modal commands and every allowlisted CUTLASS gate live in the root `Makefile`.
- The blog post is `~/code/tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample/index.qmd`.
- The paper is `~/code/papers/flashsampling-paper/`.

## Development workflow

- Use the repository `.venv`.
  Run Python tools with `.venv/bin/python` and tests with `.venv/bin/pytest`.
- Put imports at module scope by default.
- Keep Modal submission modules CPU-importable.
  Import Torch, Triton, FlashInfer, benchmark runners, and other GPU-only dependencies inside the remote function, and pass primitive serializable values from the local entry point.
- Save stdout and stderr from servers, benchmarks, evaluations, and profilers into the corresponding results directory.
  Never discard or hide process output.
- Do not block the user with a foreground sleep of 60 seconds or more.
  Use background execution for long tasks and poll at short intervals when needed.

## Code and data style

- Define high-level functions before their helpers.
  A reader should encounter the main logic before the details it delegates to.
- Public APIs use weights shaped `[V, D]` and hidden states shaped `[H, D]`.
  The Helion kernel internally transposes hidden states to `[D, H]`.
- Register sampler variants in `get_sampler()` in `core.py`.
  The `Sampler` protocol requires `prepare()` and `sample(**kwargs)`, and simple callables should use `SimpleSampler`.
- Use pandas or an equivalent DataFrame library for data analysis.
  Use `.query()` for filtering, `.merge()` for joins, `.groupby().agg()` for aggregation, `.pivot()` or `.melt()` for reshaping, and `pd.concat()` to combine frames.
  Do not replace these operations with manual nested joins or loops over unique values.

## GPU correctness and performance invariants

- Never introduce a GPU-to-CPU synchronization on a hot path.
  Operations such as `tensor.item()`, `float(tensor)`, `tensor.cpu()`, or printing a CUDA tensor wait for pending GPU work.
  Pass scalar parameters as zero-dimensional CUDA tensors when the kernel accepts them.
- Use `num_sms_cached()` in `core.py` rather than repeatedly querying CUDA device properties on decode-time paths.
- Do not make causal performance claims without empirical evidence.
  Use qualified language for hypotheses and state what measurement would resolve the uncertainty.
- Do not infer a sampling distribution from bfloat16 multinomial probabilities.
  Upcast logits to float32 before softmax as documented in `findings/upcasting-before-softmax.md`.

## Benchmark discipline

- Do not run local GPU benchmarks concurrently because they contend for the same resources.
- Independent Modal benchmarks may run concurrently when each job receives separate resources and writes to a distinct local log.
- With an empty Triton autotune cache, prefer one warmup job before launching a large parallel Modal batch.
- Kill a crashed `modal run` before relaunching it because a crash loop can keep writing into the previous log.
- Compare a baseline and candidates interleaved in the same remote function when possible.
  Separate Modal functions can land on different B200 host classes.
- A same-process `torch.mm` baseline can still change performance state during a sweep.
  Use agreement across independent runs instead of one packet for gate decisions.
- Verify every reported speedup against the underlying table and account for exceptions.
- Use `make plot-all` to regenerate every plot.

## Testing entry points

- Use `make modal-verify-correctness-tp1` for one-GPU Modal correctness.
- Use `make modal-pytest-distributed N_PROCS=2` for distributed correctness on two NVLink-connected GPUs.
- Use `make pytest-distributed` locally.
  It skips automatically on a single-GPU machine.
- Set `N_PROCS` when validating another distributed world size.
- Every implementation plan and handoff must state the validation command, expected result, actual result, possible failure modes, and artifact location as required by `loop-feedback.md`.

## Writing style

- This is a single-author project.
  Never use “we”.
  Prefer “I” with active voice, but use passive voice when it reads more naturally.
- Put one sentence on each line in prose sections to keep diffs clean.
- Write “torch compiled”, not “torch.compile-d” or “torch.compiled”.
- Prefer plain terms such as “baseline” and “Gumbel-max kernel” over unnecessary jargon.
- Never use em dashes.
  Use a period, comma, or parentheses instead.
- In rebuttals, omit internal benchmark labels and report the relevant dimensions or measurements directly.
- Use “generally outperforms” instead of “always” when exceptions exist.
- Show plots before tables, use no more than two decimal places in tables, and follow the GPU order documented in `docs/blog-maintenance.md`.

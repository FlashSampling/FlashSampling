# Benchmarking

## Runners and configuration

- `benchmarking/speed_test.py` is the speed-test runner.
- `benchmarking/triton_benchmark.py` is the Triton benchmark runner.
- Useful local commands live in `benchmarking/Makefile`.
- Equivalent Modal commands live in the root `Makefile`.
- The shared `Args` class lives in `src/fused_mm_sampling/bench/triton_benchmark_lib.py`.
- The speed test's `CliArgs` overrides the shared defaults with `case="small"` and `n_hidden_states=1`.
- Run `make plot-all` to regenerate all plots.

## Timing functions

Shared timing primitives live in `src/fused_mm_sampling/bench/triton_benchmark_lib.py`.

- `bench_cupti(fn, ...)` wraps FlashInfer's CUPTI-based `bench_gpu_time`.
  It uses adaptive iteration counts for TP1 and fixed counts for distributed runs to avoid collective mismatches.
- `bench_cuda_events(fn, ...)` uses CUDA events and flushes L2 through `create_l2_cache()` and `clear_l2_cache()`.
  It always uses fixed iteration counts.
- `synchronize(is_distributed)` uses `dist.barrier()` for distributed runs and `torch.cuda.synchronize()` for TP1.

Both timing functions return per-iteration milliseconds as `list[float]`.
The `bench_fn` argument selects `fi-cupti` or `own`.
On B200, H200, and dedicated H100 at TP1, the two methods agreed within measurement noise, with a mean difference of 1.46%.
At TP2 on dedicated H100, `own` reported latencies 7.3% higher on average, so the methods are not interchangeable for distributed comparisons.

### `fi-cupti` with TP2

`bench_fn=fi-cupti` has caused nondeterministic SIGSEGV crashes in the NCCL watchdog thread on B200 and H200, but not on dedicated H100.
The crash was observed when the benchmark used `mp.spawn` and repeatedly called `bench_cupti`, but it also occurred with one provider and sometimes disappeared on retry.
It has not been revalidated after the move to `torchrun`.
Use `bench_fn=own` for distributed benchmarks until it is revalidated.

## Result layout and Make variables

Modal Triton results use `benchmarking/modal-results/triton-bench/{bench_fn}/{gpu}/tp{N}/`.
Custom plots use `custom-plots/case-{small,large}/` below each TP directory.
The `BENCH_FN` Make variable defaults to `fi-cupti` and selects both the timing method and directory.

Makefile variables use `:=`, so environment variables do not override them.
Pass overrides as Make arguments, for example `make FOO=bar target`, rather than `FOO=bar make target`.
`NAME=default` means all providers, and `Args.providers()` treats both `None` and `"default"` as the default-provider sentinel.

## Naming

The algorithm is FMMS (Fused Matrix Multiplication and Sampling).
Provider display names follow these patterns:

- `FMMS (Triton)` for the hand-written Triton kernel.
- `FMMS (Helion)` for the Helion kernel.
- `FMMS (Triton NoNoise)` for the Triton profiling ablation without Gumbel noise.

The canonical display names live in `provider_names` in `src/fused_mm_sampling/bench/triton_benchmark.py` and flow into plots, CSVs, and the README.

# vLLM integration

The FMMS sampler is integrated into vLLM on the `feature/fmms-sampler` branch in `~/code/vllm`. Key files:

- `vllm/v1/sample/fmms_sampler.py` — thin wrapper adapting FMMS kernel to vLLM's `SamplerOutput`
- `vllm/envs.py` — `VLLM_USE_FMMS_SAMPLER` and `VLLM_FMMS_PROVIDER` env vars
- `vllm/v1/worker/gpu_model_runner.py` — calls `FMMSSampler` in `sample_tokens()` when enabled

## Local benchmarking

End-to-end vLLM benchmarks live in `benchmarking/vllm/`. Key files:

- `Makefile` — `make all` (full sweep, 3 runs) and `make quick` (smoke test, 1 run, `--enforce-eager`). Supports `MODEL=` override for different models.
- `bench-params.json` / `quick-bench-params.json` — sweep parameters (concurrency levels, num_prompts, request_rate)
- `collect_results.py` — reads `summary.csv` from each variant's latest timestamped run, prints summary table (last run only) and per-run breakdown. Usage: `python collect_results.py <model_dir>`
- `parse_engine_stats.py` — extracts KV cache occupancy, running/waiting request counts from `sweep.log` files. Parses the periodic engine stats lines emitted every 10s by vLLM. Usage: `python parse_engine_stats.py <sweep.log> [--by-concurrency]`. The `--by-concurrency` flag aggregates across runs per concurrency level. Useful for diagnosing KV cache pressure at high batch sizes.
- `plot_tpot.py` — plots median TPOT vs concurrency for all models, using `sns.lineplot` with shading for run-to-run variance. Output: `tpot_vs_concurrency.png`.
- Results are organized as `<model_slug>/baseline/`, `<model_slug>/fmms-triton/`, `<model_slug>/fmms-flashinfer/`

## Baseline sampler is plain PyTorch, not flashinfer

vLLM's default (baseline) sampling path uses plain PyTorch ops (softmax + multinomial), **not** a flashinfer sampling kernel. In nsys traces, the baseline `sample` scope shows `compute_logits` (lm_head matmul) followed by PyTorch ops, not a fused flashinfer call.

**TODO**: Add an FMMS baseline variant that uses the `naive-compiled` provider (compiled PyTorch matmul + sampling, unfused). This gives a fairer apples-to-apples comparison for both nsys profiling and TPOT benchmarks — same code path, same overhead, only the fusion differs. Currently the baseline uses vLLM's native sampler which has a different code path entirely.

## `.item()` CPU-GPU synchronization bug

`temperature[0].item()` in `fmms_sampler.py` caused a CPU-GPU sync on every decode step. At concurrency 32, TPOT regressed from 9ms to 18ms. Fix: use `temperature[0]` (scalar tensor) instead. This applies broadly — never call `.item()`, `float()`, `.cpu()`, or `print()` on GPU tensors in the hot path.

## Triton autotuning at runtime

The Triton kernel's `@triton.autotune` originally had `n_hidden_states` in its `key=` parameter. Every unique batch size triggered autotuning (benchmarking all configs). In vLLM, high concurrency produces many unique batch sizes (33, 34, ..., 256), each causing an autotune run **during the benchmark**. This inflated TPOT by 2-10x at concurrency 32+.

**Fix applied**: Replaced `n_hidden_states` with `BLOCK_SIZE_H` in the autotune `key=`, and changed `n_hidden_states` from `tl.constexpr` to a regular runtime int in the kernel signature. `BLOCK_SIZE_H` has only 3 possible values (16, 32, 64), so autotuning runs at most 3 times per (V, D) combination instead of once per unique batch size. All three uses of `n_hidden_states` inside the kernel (`tl.cdiv`, comparison, arithmetic) work fine with runtime values.

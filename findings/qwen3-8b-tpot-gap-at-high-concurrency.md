# Qwen3-8B: unexplained TPOT gap at high concurrency

## Observation

vLLM end-to-end benchmarks for Qwen3-8B on B200 show FMMS Triton is ~29% faster than baseline at concurrency 256, but the kernel microbenchmark shows FMMS is ~18% *slower* at batch size 256.

### vLLM TPOT at concurrency 256 (median across 5 runs)

| Variant       | Median TPOT (ms) | Output tok/s | Duration (s) |
|---------------|-------------------|-------------|-------------|
| Baseline      | 27.73             | ~7,900      | ~33          |
| FMMS Triton   | 19.61             | ~11,000     | ~24          |

### Kernel microbenchmark (B200, V=128,256, d=4,096)

| Batch size | FMMS Triton (ms) | Naive PyTorch Compiled (ms) | FMMS slower by |
|------------|-------------------|-----------------------------|----------------|
| 128        | 0.447             | 0.464                       | -3.5% (faster) |
| 256        | 0.769             | 0.654                       | +17.5% (slower)|

The kernel saves ~0.1ms at bsz=128 and loses ~0.1ms at bsz=256.
Neither explains a 9ms TPOT difference (28ms vs 20ms) at concurrency 256.

## What we ruled out

- **KV cache pressure**: Baseline KV cache usage peaks at 8.5% at concurrency 256. Not memory-bound.
- **Request queuing**: Both variants hit `Peak concurrent requests: 512`. No significant waiting queue differences visible in baseline logs.
- **Vocab size mismatch**: The microbenchmark uses V=128,256 (Llama 3 8B "small" case), but Qwen3-8B has V=151,936. This is a ~18% larger vocab, but unlikely to flip the relative performance that dramatically.

## Missing data

- **No engine stats for fmms-triton at concurrency 256.** The first Modal run timed out (3600s) during the fmms-triton variant at concurrency 8. The successful second run was a separate Modal invocation whose logs were not saved locally. The benchmark result JSONs were saved to the volume, but the interleaved engine stats (batch sizes, waiting queue, KV cache) were not captured.

## Hypotheses

The gap must come from something in vLLM's code path outside the kernel itself:

1. **Hidden GPU-CPU synchronization in the baseline sampler.** vLLM's native sampler uses plain PyTorch ops (softmax + multinomial). If any of these ops trigger implicit `.item()` or CPU-GPU syncs (e.g., for shape computation, error checking), the cost compounds at 256 concurrent decode steps.

2. **Extra kernel launches in the baseline sampling path.** The baseline does: `compute_logits` (lm_head matmul) + softmax + multinomial = 3+ kernel launches per decode step. FMMS fuses matmul + sampling into 1 kernel. The kernel launch overhead itself is small (~5us each), but 256 concurrent requests means the scheduler has more work to interleave.

3. **Memory allocation patterns.** The baseline materializes the full logits tensor `[bsz, V]` = `[256, 128K]` = 64 MB, then reads it back for softmax + sampling. FMMS never materializes this intermediate. At high concurrency, this extra memory traffic could cause more L2 cache thrashing or DRAM bandwidth contention.

4. **torch.compile graph fragmentation.** Different batch sizes may cause different compiled graph paths. The FMMS sampler bypasses torch.compile for the sampling step entirely (it's a hand-written Triton kernel), while the baseline goes through compiled PyTorch ops that may behave differently under varying batch sizes.

## Proposed investigation: nsys profiling on Modal

The most direct way to identify the source of the gap is to nsys-profile both variants on Modal under the same conditions.

### Approach

Adapt the existing local nsys workflow (`benchmarking/vllm/Makefile` `run_nsys` target) to run inside the Modal container:

1. **Install nsys in the Modal image.** The `pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel` base image may already ship nsight-systems. If not, add `apt-get install -y nsight-systems-*` to the image build.

2. **Single-request profile (low-hanging fruit).** Start vLLM under `nsys profile --capture-range=cudaProfilerApi`, warmup, `/start_profile`, send one request with `max_tokens=10`, `/stop_profile`. Save the `.nsys-rep` to the Modal volume. This reveals the per-step kernel launch sequence and any CPU-GPU syncs in both baseline and FMMS, even at bsz=1.

3. **High-concurrency profile (ideal but harder).** To capture the actual bsz=256 behavior, fire 256 concurrent requests during the profile window. The profile will be larger but would show the exact GPU timeline under load. Use a short `max_tokens` (e.g., 5) to keep the trace manageable.

4. **Compare the `sample` NVTX scope.** With `VLLM_NVTX_SCOPES_FOR_PROFILING=1`, vLLM annotates the model runner with NVTX ranges (`preprocess`, `forward`, `postprocess`, `sample`). Compare the `sample` scope duration and contents between baseline and FMMS.

### What to look for in the traces

- **Kernel count inside the `sample` scope.** Baseline should show multiple kernels (softmax, multinomial, etc.). FMMS should show one fused kernel.
- **CPU-GPU sync points.** Look for `cudaStreamSynchronize` or `cudaDeviceSynchronize` calls inside the sampling path.
- **Memory operations.** Large `cudaMemcpy` or allocation calls in the baseline that FMMS avoids.
- **Gaps between kernels.** Dead time on the GPU between the `forward` and `sample` scopes could indicate Python overhead or sync points.

### Implementation notes

- No `sudo` needed on Modal (containers run as root).
- Save `.nsys-rep` files to the Modal volume (`/vol-fused-mm-sample/`) for download.
- The existing `NSYS_MAX_TOKENS=10` and warmup pattern from the local Makefile can be reused.
- For the high-concurrency variant, use `vllm bench serve` with a single concurrency level instead of a full sweep.

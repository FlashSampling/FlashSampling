# Integrating FMMS into vLLM

## Goal

Benchmark FMMS (Fused Matrix Multiplication & Sampling) in a real serving
engine.  vLLM is the target because it is the most widely used open-source LLM
serving framework.

## The sampling pipeline in vLLM v1

The path from hidden states to sampled token IDs consists of two clearly
separated phases that run in **different methods** to allow CUDA graph
optimization:

### Phase 1 — `execute_model()` (gpu_model_runner.py:3516-3596)

```
hidden_states = model.forward(input_ids, positions, ...)   # transformer
sample_hidden_states = hidden_states[logits_indices]        # [B, D]
logits = model.compute_logits(sample_hidden_states)         # [B, V]
```

`compute_logits` is defined on each model class (e.g. `LlamaForCausalLM`):

```python
def compute_logits(self, hidden_states):
    return self.logits_processor(self.lm_head, hidden_states)
```

The `LogitsProcessor` (model_executor/layers/logits_processor.py) does:

1. `lm_head.quant_method.apply(lm_head, hidden_states)` — the **matmul**
   `hidden_states @ lm_head.weight.T` → `[B, V]` logits
2. Gather across tensor-parallel ranks (if TP > 1)
3. Strip vocab padding
4. Optional soft-cap (Gemma 2) and scaling

### Phase 2 — `sample_tokens()` (gpu_model_runner.py:3598-3644)

```
logits = execute_model_state.logits
sampler_output = self._sample(logits, spec_decode_metadata)
```

`_sample` calls `self.sampler(logits, sampling_metadata)` which is
`vllm/v1/sample/sampler.py:Sampler.forward()`. The steps:

1. Compute raw logprobs (if requested)
2. Cast logits to float32
3. Apply logits processors (allowed tokens, bad words, min tokens, logit bias)
4. Apply penalties (repetition, frequency, presence)
5. **Sample**:
   a. Greedy: `argmax`
   b. Temperature scaling: `logits /= temperature`
   c. Argmax-invariant processors (min_p)
   d. Top-k / top-p filtering + random sampling → `TopKTopPSampler`

### TopKTopPSampler (v1/sample/ops/topk_topp_sampler.py)

Multiple backends, selected at init:

| Backend | Condition | Notes |
|---------|-----------|-------|
| **FlashInfer** | `VLLM_USE_FLASHINFER_SAMPLER=1`, CUDA, no per-request generators | Fuses softmax + top-k + top-p + sampling. Uses rejection sampling internally to avoid full vocab sort. Requires CPU-GPU sync. |
| **Native PyTorch** | Default on CUDA | `apply_top_k_top_p` (sort-based) → `softmax` → Gumbel-max trick (`probs / q.exponential_()`). No CPU-GPU sync. |
| **torch.compile CPU** | CPU platforms | Same as native but compiled |

The native path uses the **Gumbel-max trick** (same as FMMS), but as a
*separate* step after the lm_head matmul has already materialized the full
`[B, V]` logits tensor.

## Where FMMS would plug in

FMMS fuses the lm_head matmul with categorical sampling via the Gumbel-max
trick, **never materializing the full [B, V] logits tensor**.  This is exactly
the boundary between Phase 1 and Phase 2 above.

### Injection point

In `gpu_model_runner.py`, replace:

```python
# Current code (lines 3551-3552):
sample_hidden_states = hidden_states[logits_indices]
logits = self.model.compute_logits(sample_hidden_states)
```

With an FMMS path that computes `sampled_token_ids` directly from
`sample_hidden_states` and `lm_head.weight`, skipping the full logits
materialization and the entire `Sampler` forward pass.

### What must be bypassed

When FMMS is active, these steps from the normal pipeline are **skipped**:

| Step | Can FMMS support it? |
|------|---------------------|
| Logits materialization `[B, V]` | **Skipped** — this is the whole point |
| Logprobs computation | **No** — needs full logits. Could do a second pass but defeats the purpose |
| Allowed token IDs mask | **No** — would require masking inside the kernel |
| Bad words exclusion | **No** — same as above |
| Logit bias processor | **No** — same |
| Min tokens processor | **No** — same |
| Repetition/frequency/presence penalties | **No** — needs full logits + token history |
| Temperature scaling | **Yes** — trivially fused (divide logits by T before Gumbel noise) |
| min_p filtering | **No** — needs full softmax distribution |
| Top-k filtering | **No** — needs partial sort |
| Top-p (nucleus) filtering | **No** — needs cumulative softmax |
| Greedy sampling | **Yes** — FMMS with temperature→0 (just argmax of matmul) |
| Gumbel-max random sampling | **Yes** — this is what FMMS does |

### Constrained sampling parameters

FMMS can only serve requests where:

- `temperature > 0` (random) or `temperature = 0` (greedy)
- `top_k = -1` (disabled) or `top_k = vocab_size`
- `top_p = 1.0` (disabled)
- `min_p = 0.0` (disabled)
- No repetition / frequency / presence penalties
- No logit bias
- No allowed/banned token lists
- No logprobs requested
- No structured output / grammar constraints

This is actually the **default configuration** for many use cases (e.g.
`temperature=0.6` with no other constraints).

## Fair comparison: FMMS vs FlashInfer sampling

### What FlashInfer's sampler does

FlashInfer's `top_k_top_p_sampling_from_logits` fuses:
- Softmax
- Top-k filtering
- Top-p (nucleus) filtering
- Categorical sampling

It does **not** fuse the lm_head matmul. It takes `[B, V]` logits as input —
the full logits tensor is already materialized.

### What FMMS does differently

FMMS fuses:
- The lm_head matmul (`hidden_states @ weights.T`)
- Gumbel noise addition
- Argmax (tile-parallel reduction)

It **never materializes** the `[B, V]` logits tensor, which for large
vocabularies (128K tokens) saves significant memory bandwidth.

### Apples-to-apples comparison

For a fair benchmark, we should compare:

| Scenario | Pipeline |
|----------|----------|
| **Baseline (FlashInfer)** | `compute_logits` → `flashinfer.top_k_top_p_sampling_from_logits` |
| **Baseline (Native)** | `compute_logits` → `apply_top_k_top_p` → `softmax` → `Gumbel-max` |
| **Baseline (temperature-only)** | `compute_logits` → `softmax` → `Gumbel-max` (no top-k/top-p) |
| **FMMS** | `fmms_sample(hidden_states, weights, temperature)` — single fused op |

The fairest comparison is **Baseline (temperature-only)** vs **FMMS**, since
both do temperature + pure categorical sampling (Gumbel-max) without top-k/top-p.
This isolates the benefit of **fusing the matmul with sampling**.

Comparing against FlashInfer with top-k/top-p would mix two effects:
1. Fusion benefit (FMMS advantage)
2. Missing top-k/top-p filtering (FMMS disadvantage in quality)

## Framing: FMMS as a performance-quality tradeoff

FMMS can be positioned as a **sampling method that trades sampling parameter
flexibility for performance**:

- **When to use FMMS**: High-throughput serving where requests use simple
  sampling (temperature only), e.g. chat completions with `temperature=0.6`.
  The vocabulary matmul + sampling is fused into a single kernel, saving one
  full read/write of the `[B, V]` tensor.

- **When NOT to use FMMS**: Requests requiring top-k, top-p, min_p, penalties,
  logprobs, structured output, or other constraints that need access to the full
  logit distribution.

- **Mixed batches**: vLLM serves batches where different requests can have
  different sampling parameters. A hybrid approach could route
  temperature-only requests through FMMS and the rest through the normal path.
  However, this adds complexity and the lm_head matmul would still need to
  run for the non-FMMS requests.

## Implementation plan

We integrate FMMS into a private vLLM branch using if-else branching controlled
by the `VLLM_USE_FMMS_SAMPLER` environment variable. Both the Triton and Helion
kernels are supported, selectable via `VLLM_FMMS_PROVIDER`.

### Key idea

In `execute_model()`, when FMMS is active we **skip** `compute_logits()` (set
`logits = None`) and let `sample_hidden_states` flow through to `sample_tokens()`.
In `_sample()`, an FMMS branch calls the fused kernel directly on
`sample_hidden_states` + `lm_head.weight`, producing token IDs without ever
materializing the `[B, V]` logits tensor.

### Data flow

```
                    Default path                          FMMS path
                    ────────────                          ─────────
execute_model():
  hidden_states = model.forward(...)                     (same)
  sample_hidden_states = hidden_states[logits_indices]   (same)
  logits = model.compute_logits(sample_hidden_states)    logits = None  ← skip matmul

sample_tokens():
  _sample(logits, ..., sample_hidden_states):
    sampler(logits, sampling_metadata)                   fmms_sampler(lm_head_weight,
                                                             sample_hidden_states,
                                                             temperatures)
    → SamplerOutput(sampled_token_ids, logprobs)         → SamplerOutput(sampled_token_ids, None)
```

### Files to modify

#### 1. `vllm/envs.py` — add env vars

```python
VLLM_USE_FMMS_SAMPLER: bool = False
VLLM_FMMS_PROVIDER: str = "fused-triton"   # or "helion"
```

Follow the same pattern as `VLLM_USE_FLASHINFER_SAMPLER`.

#### 2. `vllm/v1/worker/gpu_model_runner.py` — main changes

**`ExecuteModelState` (line 314):** Change `logits` field to `torch.Tensor | None`.

**`__init__` (around line 418):** Read the flag and store it.

```python
self.use_fmms_sampler = envs.VLLM_USE_FMMS_SAMPLER
```

**`load_model` (after line 4127):** Capture lm_head weight reference.

```python
if self.use_fmms_sampler:
    self.fmms_lm_head_weight = self.model.lm_head.weight  # [V, D]
```

**`execute_model()` (lines 3551–3552):** Skip `compute_logits` when FMMS.

```python
sample_hidden_states = hidden_states[logits_indices]
if self.use_fmms_sampler:
    logits = None
else:
    logits = self.model.compute_logits(sample_hidden_states)
```

**`sample_tokens()` (line 3644):** Pass `sample_hidden_states` to `_sample`.

```python
sampler_output = self._sample(logits, spec_decode_metadata, sample_hidden_states)
```

**`_sample()` (line 2843):** Add FMMS branch.

```python
def _sample(self, logits, spec_decode_metadata, sample_hidden_states=None):
    sampling_metadata = self.input_batch.sampling_metadata
    self.input_batch.update_async_output_token_ids()

    if self.use_fmms_sampler and logits is None:
        return self.fmms_sampler(
            lm_head_weight=self.fmms_lm_head_weight,
            hidden_states=sample_hidden_states,
            sampling_metadata=sampling_metadata,
        )

    # ... existing code unchanged ...
```

#### 3. `vllm/v1/sample/fmms_sampler.py` — new file

Thin wrapper that adapts the FMMS kernel interface to vLLM's `SamplerOutput`:

```python
import torch
from vllm.v1.outputs import SamplerOutput

class FMMSSampler:
    def __init__(self, provider: str = "fused-triton"):
        from fused_mm_sampling.core import get_sampler
        self.provider = provider
        # get_sampler needs a weights tensor — we defer init to first call
        self._sampler = None

    def _ensure_sampler(self, lm_head_weight):
        if self._sampler is None:
            from fused_mm_sampling.core import get_sampler
            self._sampler = get_sampler(self.provider, weights=lm_head_weight)
            self._sampler.prepare()

    def __call__(self, lm_head_weight, hidden_states, sampling_metadata):
        self._ensure_sampler(lm_head_weight)

        temperature = sampling_metadata.temperature  # [B] tensor or None
        # FMMS kernel takes scalar temperature — use the batch value
        # (for benchmarking, all requests use the same temperature)
        if temperature is not None:
            temp_scalar = temperature[0].item()
        else:
            temp_scalar = 1.0

        sampled = self._sampler.sample(
            weights=lm_head_weight,
            hidden_states=hidden_states,
            num_samples=1,
            temperature=temp_scalar,
        )  # [B, 1] torch.long

        return SamplerOutput(
            sampled_token_ids=sampled.to(torch.int32),
            logprobs_tensors=None,
        )
```

### Temperature handling

The FMMS kernel takes a **scalar** temperature. vLLM supports per-request
temperatures. For benchmarking this is fine (all requests use the same value).
For production, the kernel would need to be extended to accept a per-row
temperature vector — this is a straightforward change (index into a `[B]`
tensor inside the Triton kernel's inner loop).

### Install `fused-mm-sampling` in the vLLM venv

```bash
cd ~/code/vllm && pip install -e ~/code/fused-mm-sample
```

Both kernels become available since they're registered in `get_sampler()`.

### Key technical details

- **Weight access**: `self.model.lm_head.weight` gives the `[V, D]` tensor.
  With tensor parallelism it is sharded — for initial benchmarking use TP=1.

- **CUDA graphs**: vLLM captures the forward pass in CUDA graphs. The FMMS
  kernel runs in `sample_tokens()` which is **outside** the graph, so no
  conflict.

- **Two-phase execution**: We preserve the two-phase split. `execute_model()`
  stores `logits=None` + `sample_hidden_states` in `ExecuteModelState`.
  `sample_tokens()` reads both and calls the FMMS kernel. No scheduling
  assumptions are violated.

- **Dtype**: vLLM models run in bfloat16/float16. The FMMS Triton kernel
  accumulates in float32 internally, matching vLLM's own `logits.to(float32)`
  cast.

### Verification

1. Run vLLM **without** the env var — behavior is identical (no code path changes).
2. Run with `VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton`:
   - Send requests with `temperature=0.6` (no top-k/top-p)
   - Verify tokens are returned and text is coherent
3. Run with `VLLM_FMMS_PROVIDER=helion` and repeat.
4. Benchmark: compare throughput (tokens/sec) and latency (TTFT, TPOT)
   with and without FMMS using `vllm bench serve`.

## Implementation status

The integration is complete and benchmarked on the `feature/fmms-sampler` branch in `~/code/vllm`.

### Files modified in vLLM

| File | Change |
|------|--------|
| `vllm/envs.py` | Added `VLLM_USE_FMMS_SAMPLER` and `VLLM_FMMS_PROVIDER` env vars |
| `vllm/v1/sample/fmms_sampler.py` | New file — thin wrapper adapting FMMS kernel to `SamplerOutput` |
| `vllm/v1/worker/gpu_model_runner.py` | 5 edits: `ExecuteModelState.logits` → `Optional`, init sampler, capture `lm_head.weight`, skip `compute_logits`, add FMMS branch in `_sample()` |

### Install fused-mm-sampling in the vLLM venv

```bash
cd ~/code/vllm && uv pip install -e ~/code/fused-mm-sample --python venv/bin/python
```

## Benchmark results

Benchmarked with `vllm bench sweep serve` on Qwen/Qwen3-1.7B, RTX 3090.
See `findings/vllm-bench-results/README.md` for full details and reproduction via `make`.

Three variants:
- **Baseline**: vLLM default (cuBLAS `compute_logits` + FlashInfer sampler)
- **FMMS Triton**: Fused matmul+sampling kernel (`fused-triton` provider)
- **FMMS FlashInfer**: Control — unfused matmul + FlashInfer sampling through FMMS integration path

### Median TPOT (ms)

| Concurrency | Baseline | FMMS Triton | FMMS FlashInfer |
|---|---|---|---|
| 1 | 5.24 | 5.11 | 5.30 |
| 32 | 8.95 | 8.79 | 8.93 |

All three variants perform equivalently. FMMS Triton matches baseline within noise.

### Key finding: `.item()` CPU-GPU sync

An earlier version used `temperature[0].item()` to extract a scalar from the per-request temperature tensor. This caused a CPU-GPU synchronization on every decode step, which compounded at high concurrency (TPOT was 18.66ms vs 8.98ms baseline at concurrency 32). Replacing with `temperature[0]` (keeping the value as a scalar tensor) eliminated the regression.

## Commands

### Run benchmarks

```bash
make -C findings/vllm-bench-results all        # all three variants
make -C findings/vllm-bench-results baseline    # just baseline
make -C findings/vllm-bench-results fmms-triton # just FMMS Triton
```

### Smoke test

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-1.7B",
    "messages": [{"role": "user", "content": "What is 2+2? Answer in one word."}],
    "temperature": 0.6,
    "max_tokens": 32
  }' | python3 -m json.tool
```

## Summary

| Aspect | Detail |
|--------|--------|
| **Injection point** | `execute_model` skips `compute_logits`; `_sample` calls FMMS kernel |
| **Control** | `VLLM_USE_FMMS_SAMPLER=1`, `VLLM_FMMS_PROVIDER=fused-triton\|helion` |
| **Fair baseline** | Temperature-only sampling (no top-k/top-p) via native Gumbel-max |
| **FMMS advantage** | Avoids materializing `[B, V]` logits tensor (128K × B × 4 bytes) |
| **FMMS limitation** | No top-k, top-p, min_p, penalties, logprobs, or structured output |
| **Positioning** | Performance optimization for simple sampling configurations |
| **Files changed** | `envs.py`, `gpu_model_runner.py`, new `fmms_sampler.py` |
| **Performance** | Matches baseline TPOT at all concurrency levels (Qwen3-1.7B, RTX 3090) |
| **Pitfall** | Avoid `.item()` on GPU tensors — causes CPU-GPU sync that kills throughput at high concurrency |

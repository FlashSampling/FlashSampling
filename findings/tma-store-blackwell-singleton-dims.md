# TMA store descriptors with singleton dims silently no-op on Blackwell

## TL;DR

On B200 (sm_100, Triton 3.6, torch 2.11.0+cu130), the FMMS Triton kernel was using `tl.make_tensor_descriptor(...).store(...)` for both its `bf16` and `int64` per-tile output buffers with `block_shape=[1, 1, BLOCK_SIZE_H]` (two singleton dims) over a 3D output buffer of shape `[num_samples, max_grid_size_v, n_hidden_states]`. The store **silently dropped most writes**: only the first ~2 of 1187 V-tile slots were ever written. Downstream code gathered uninitialized memory and returned out-of-vocab token ids, crashing vLLM at the next embedding lookup.

The same descriptor pattern is **correct** on Hopper (sm_90) and on RTX 3090 (sm_86). The bug is specific to Blackwell.

## Symptom

In vLLM on B200 with Qwen3-1.7B (V=151936, D=2048):

```
RuntimeError: FMMS produced OOB token ids: 1/1 out of [0, 151936).
min=-4617947733907095797, max=-4617947733907095797.
H=1, V_local=151936, tp.size=1, num_samples=1
```

`-4617947733907095797 = 0xBFE0000000000000` is the IEEE-754 binary64 representation of `-0.5`. It's not the kernel writing fp64 — it's `torch.empty_like` returning uninitialized memory whose bytes happened to look that way.

Without `--enforce-eager`, the failure surfaces a step later as:

```
/tmp/torchinductor_root/.../py:41: Assertion `index out of bounds: 0 <= tmp5 < 151936' failed
```

(`tmp5` is a sampled token id used to index `embed_tokens.weight`.)

With `--enforce-eager`, the failure surfaces as a sticky `CUBLAS_STATUS_EXECUTION_FAILED` on the next decode step's qkv_proj GEMM.

## Diagnosis

1. **Pre-fill `maxs_idx` with a sentinel** (`torch.full(..., -99999)`) to distinguish "kernel writes garbage" from "kernel doesn't write at all". Result: only **2 of 1187** V-tile slots overwrite the sentinel; the rest still equal `-99999`.

2. **Same kernel + same dumped inputs** on RTX 3090: writes **1187/1187** correctly.

3. **Standalone replay on B200** (no vLLM, no torch.compile, no cudagraphs): also only **2/1187**. The failing path is the kernel, not the surrounding environment.

4. **Forced WARP_SPECIALIZE=False on B200**: still 2/1187. Warp specialization is not the cause.

5. **Forced the dev-machine autotune config** (`BLOCK_SIZE_D=32, num_warps=4, num_stages=2, maxnreg=255`) on B200: still 2/1187. No config in the search space writes the slots.

6. **Both the `bf16` `max_desc.store` and the `int64` `max_idx_desc.store` are affected.** The bf16 store sometimes returns plausible numbers (for the few slots it does write) and sometimes ridiculous garbage like `2.12e+38` (near bf16 max), suggesting the store completes for the first ~2 V-tiles per program then drops out.

7. **The `maxs` and `maxs_idx` outputs are byte-for-byte deterministic** between iterations on RTX 3090; on B200 they vary across iterations because most of the buffer is whatever happened to be on those memory pages.

## Root cause

Comparison against the canonical Triton tutorial `09-persistent-matmul.py` (which is documented to work on Blackwell):

| | Tutorial (works) | Our broken pattern |
|---|---|---|
| **Descriptor rank** | 2D `[M, N]` | 3D `[num_samples, V_tiles, H]` |
| **`block_shape`** | `[BLOCK_SIZE_M, BLOCK_SIZE_N]` | `[1, 1, BLOCK_SIZE_H]` |
| **Singleton dims** | none | two |
| **Element dtype** | bf16 / fp16 | bf16 + int64 |
| **Store size** | `BLOCK_M * BLOCK_N * 2` bytes (KBs) | `BLOCK_SIZE_H * 2 (or 8)` bytes (~16-128 bytes) |
| **Per-iteration store coverage** | dense block | sparse: writes a single tile in a `[1, 1, *]` slot |

The Blackwell TMA hardware appears to silently drop stores when the descriptor pattern uses a 3D shape with two singleton block dimensions (effectively dressing up a 1D scatter as 3D TMA). This isn't TMA's intended use — TMA is built for large contiguous bulk transfers.

## Fix

**Drop TMA descriptors for the tiny per-tile output stores entirely.** The output buffers are ~10 KB total and only one store per tile per program — TMA gives no bandwidth win below the byte sizes where its descriptor setup overhead dominates. Replace `desc.store(...)` with plain `tl.store(ptr + offset, val, mask=...)`.

Keep the TMA *load* descriptors for `weights` and `hidden_states` (the matmul inputs) — those are large dense reads where TMA pays off.

```python
# Before (broken on B200):
max_desc = tl.make_tensor_descriptor(
    max_out_ptr,
    shape=[num_samples, max_grid_size_v, n_hidden_states],
    strides=[max_grid_size_v * n_hidden_states, n_hidden_states, 1],
    block_shape=[1, 1, BLOCK_SIZE_H],
)
...
max_desc.store([sample_idx, pid_v_c, h_start_c], gumbel_max[None, None, :])

# After (fixed):
offsets_h_out = h_start_c + tl.arange(0, BLOCK_SIZE_H)
mask_h_out = offsets_h_out < n_hidden_states
base_offset = (
    sample_idx * max_grid_size_v * n_hidden_states
    + pid_v_c * n_hidden_states
    + offsets_h_out
)
tl.store(max_out_ptr + base_offset, gumbel_max, mask=mask_h_out)
```

See commit `b5f2bea` and `core.py:fused_mm_sample_triton_kernel`.

## Verification

After the fix:

- **Standalone B200 replay** writes 1187/1187 `maxs_idx` slots, deterministic across iterations, all in `[0, V)`. Identical first8/last8 values to the RTX 3090 reference.
- **Full vLLM Qwen3-1.7B sweep on B200** runs cleanly through every concurrency point (`c ∈ {1..256}`, 5 runs each). 0 failed requests (was crashing on the first request before).
- **B200 microbench** shows the new no-TMA-store kernel is at least as fast as the prior (broken) kernel. 1-7% faster at higher batch sizes — the TMA descriptor setup/teardown overhead was apparently dominating the per-tile output store cost. Larger savings at higher batch sizes.
- **No regression on H100 / H200**: those GPUs were never broken (TMA store descriptors with singleton dims are only buggy on Blackwell), and the new code path is functionally identical to TMA stores at this scale.

## Why we didn't catch this with the existing chi-squared test

The microbench `modal_fmms_correctness.py` was a *false negative* on B200. It checked `0 <= sample < V` after the host-side `_local_reduce` gather, not the raw `maxs_idx` values. With:

- random `randn` inputs → random global argmax → uniformly random V-tile selected by gather
- only 2/1187 valid slots in `maxs_idx`, the rest being uninitialized memory that on a freshly-allocated CUDA page is mostly zeros
- `0` is in `[0, V)` and passes the assertion

…the microbench reported "all 1800 calls OK" while the kernel was actually returning zeros for the gathered token id 99.8% of the time. The chi-squared sampling distribution test would have caught this on B200 (since the empirical distribution would be heavily biased toward token id 0), but we hadn't run it on B200 before — only locally on RTX 3090 (where the kernel is correct).

**Lesson:** correctness microbenches that go through a final reduction (gather, argmax, etc.) hide kernel bugs that only show up in masked-out output regions. To catch this class of bug in the future, the microbench should pre-fill output buffers with a sentinel and assert ALL slots are overwritten with valid values, not just the gathered final result.

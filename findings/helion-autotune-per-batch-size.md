# Helion kernel re-autotuning for every unique batch size

## Problem

The Helion kernel autotuning runs separately for every unique `n_hidden_states` (H) value. In the benchmark sweep (H = 1, 2, 4, 8, 16, 32, 64, 128, 256), this triggers **9 separate autotuning runs**, each taking 6-35 seconds. Total autotuning time: ~2 minutes.

This is the Helion equivalent of the Triton autotune key issue (see `findings/triton-autotune-batch-size-key.md`). However, the mechanism is different: Triton re-autotunes because `n_hidden_states` was in the `key=` parameter; Helion re-autotunes because it specializes on tensor shapes.

## Root cause

Helion's specialization key includes exact tensor shapes by default (`static_shapes` is not set to `False` in the kernel decorator). The key is computed in `_tensor_key()` in `helion/runtime/kernel.py` (line 938):

```python
# static_shapes=True (default):
return (
    obj.dtype,
    obj.device.type,
    (*obj.size(),),       # exact size tuple
    (*obj.stride(),),
    static_indices,
)
```

Since `hidden_states` is `[D, H]`, every unique H produces a different specialization key, triggering a new autotuning run and kernel compilation.

## Evidence from benchmark logs

The `block_sizes[0]` in each Helion autotuned config matches the H value exactly:

| H | Autotune time | Best config `block_sizes` | Best `block_sizes[0]` (H block) |
|---|---|---|---|
| 1 | 8.7s | `[1, 128]` | 1 |
| 2 | 10.7s | `[2, 256]` | 2 |
| 4 | 11.4s | `[4, 256]` | 4 |
| 8 | 9.9s | `[8, 256]` | 8 |
| 16 | 6.5s | `[16, 64]` | 16 |
| 32 | 21.0s | `[32, 128]` | 32 |
| 64 | 35.3s | `[64, 64]` | 64 |
| 128 | (in progress) | ... | ... |

This shows two things: (a) each H triggers a separate autotune, and (b) the optimal H block size genuinely varies — the autotuner picks H_block = H for small H, and starts capping around 32-64 for larger H.

## `static_shapes=False` is not a fix

Helion has a `static_shapes=False` mode that buckets sizes into `{0, 1, 2+}` via `min(s, 2)`. This would reduce autotuning to just 2 runs (H=1, and H>=2). However, the config autotuned for H=2 (`block_sizes[0]=2`) would be reused for H=256, producing 128 tiles in the H dimension — clearly suboptimal compared to the `block_sizes[0]=32` or `64` that the autotuner would pick for large H.

## Scope of impact

**Benchmarking only.** The Helion kernel is not used in vLLM (only the Triton kernel is integrated). The autotuning cost is a one-time expense per (GPU, V, D, H) combination, cached to disk in `helion-cache/`. Subsequent benchmark runs reuse cached configs instantly.

For the Triton kernel in vLLM, the analogous fix (replacing `n_hidden_states` with `BLOCK_SIZE_H` in the autotune key) works because `BLOCK_SIZE_H` has only 3 values. Helion's autotuner doesn't have an equivalent mechanism — its specialization is shape-driven, not user-controlled.

## Possible mitigations

1. **Accept the cost.** Autotuning results are cached to disk. After the first benchmark run, subsequent runs are instant. This is the current approach.

2. **Use `hl.specialize` on H.** Could bucket H into a few groups (e.g., 1, <=16, <=64, <=256). But `hl.specialize` has a known bug with `hl.rand` when a dimension is specialized to 1 (patched locally, not upstream).

3. **Pad H to power-of-2 before calling the kernel.** Reduces unique H values but wastes computation on padding.

4. **Set a single fixed Helion config.** Use `@helion.kernel(config=helion.Config(...))` to skip autotuning entirely. Sacrifices per-H optimization.

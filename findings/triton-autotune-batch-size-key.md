# Triton autotune: replacing `n_hidden_states` with `BLOCK_SIZE_H` in key

## Problem

The Triton kernel's `@triton.autotune` included `n_hidden_states` (batch size) in its `key=` parameter. Every unique batch size triggered a full autotuning run, benchmarking all config combinations.

In vLLM, high concurrency produces many unique batch sizes (H=33, 34, ..., 256). Each one caused autotuning **during inference**, inflating TPOT by 2-10x at concurrency 32+. Evidence: autotune cache timestamps showed H=33 through H=256 were all first tuned during a single benchmark run. Run 0 at concurrency 32 took 18.3ms (autotuning); Runs 1-2 took 8.6ms (cached).

## Root cause

`n_hidden_states` was declared as `tl.constexpr` in the kernel signature and listed in the autotune `key=`. Triton treats each unique value of a `key=` parameter as a separate specialization point, triggering a fresh autotune search.

But the optimal `BLOCK_SIZE_V` / `BLOCK_SIZE_D` config doesn't actually depend on the exact batch size — it depends on `BLOCK_SIZE_H`, which the heuristic already computes from batch size and which only has 3 possible values (16, 32, 64).

## Fix

Two changes in `src/fused_mm_sampling/core.py`:

1. **Autotune key**: replaced `"n_hidden_states"` with `"BLOCK_SIZE_H"`.

```python
# Before
key=["vocab_size", "hidden_size", "n_hidden_states", "num_samples", "GUMBEL"],

# After
key=["vocab_size", "hidden_size", "BLOCK_SIZE_H", "num_samples", "GUMBEL"],
```

2. **Kernel parameter**: changed `n_hidden_states` from `tl.constexpr` to a regular runtime int.

```python
# Before
n_hidden_states: tl.constexpr,

# After
n_hidden_states,
```

The three uses inside the kernel all work with runtime values:
- `tl.cdiv(n_hidden_states, BLOCK_SIZE_H)` — `tl.cdiv` accepts runtime args
- `offsets_h < n_hidden_states` — comparison works at runtime
- `pid_v * n_hidden_states * num_samples` — arithmetic works at runtime

## Result

Autotuning runs at most 3 times per (V, D) combination (once per `BLOCK_SIZE_H` value) instead of once per unique batch size. In vLLM this eliminates ~250 redundant autotune runs during serving.

## How vLLM kernels avoid this

They never include batch-size-like dimensions in the autotune `key=`. Examples from `ssd_bmm.py`, `chunk_o.py`, `chunk_delta_h.py` all use `key=["H", "K", "V", "BT"]` where H=num_heads (fixed per model). Batch size only affects the grid shape.

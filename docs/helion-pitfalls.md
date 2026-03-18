# Helion kernel pitfalls

Helion has an API reference: https://helionlang.com/api/index.html

and also reference examples: https://helionlang.com/examples/index.html

## `torch.argmax()` returns global indices

Inside a Helion kernel, `torch.argmax(tensor, dim=0)` returns **global** indices, not tile-local ones. The generated Triton code uses `triton_helpers.max_with_index` with the tile's global `indices_0` offset baked in. **Do NOT add `tile_v.begin`** — that double-counts the offset.

```python
# WRONG — double counts offset
new_max_idx_local = torch.argmax(summed, dim=0)
new_max_idx_global = tile_v.begin + new_max_idx_local  # BUG

# CORRECT — argmax already returns global indices
new_max_idx = torch.argmax(summed, dim=0)
```

## `for tile in hl.tile(N)` is parallel, not sequential

Each tile becomes a separate GPU program (thread block) that runs in parallel. You **cannot** do cross-tile communication via shared tensors — it's a race condition. This manifests as correct-looking results for some vocab sizes (e.g. 256, evenly divisible) but broken distributions for others (e.g. 100).

**Fix**: Use `hl.barrier()` to synchronize stages within a single kernel:
1. **Stage 1**: Each `(V, H)` tile writes its local max/argmax to `tile_maxs[tile_v.id, :]`.
2. `hl.barrier()` — grid-wide sync.
3. **Stage 2**: Reduce across tiles with `argmax` + `gather` inside the same kernel.

This eliminates Python-side tensor allocations and separate kernel launches for the reduction. However, rigorous benchmarking (25 warmup + 100 runs) shows the barrier version is ~3% **slower** at H=1 on RTX 3090 (2.38ms vs 2.32ms) due to cooperative launch constraints (`num_stages=1`, persistent scheduling, barrier sync stalls). The host-side overhead it eliminates is only ~0.01ms — negligible. The ~5ms gap seen under Proton profiling is an instrumentation artifact (Proton adds fixed overhead per kernel launch). See `findings/helion-barrier-single-kernel.md`.

## Tensor allocations inside kernels trigger warnings

`TensorOperationInWrapper` warning fires for tensor ops outside `hl.tile` loops. Allocate output buffers in the Python wrapper and pass them as kernel arguments instead.

## Advanced indexing does not work for gather

Helion interprets `tensor[idx_tensor, tile_var]` as a Cartesian product (producing a higher-rank result), not element-wise gather. Use `torch.gather` instead:

```python
# WRONG — Cartesian product, produces 2D
out[tile_h] = tile_max_idxs[best_tile, tile_h]  # RankMismatch error

# CORRECT — element-wise gather
out[tile_h] = torch.gather(
    tile_max_idxs[:, tile_h], 0, best_tile.unsqueeze(0)
).squeeze(0)
```

## Random number generation

Use `hl.rand([tile_v, n], seed=seed)` — not `torch.rand` or `torch.rand_like`. The `hl.rand` API uses Philox PRNG with proper per-tile offsets. Historical issues #1041 and #1309 are fixed in current Helion.

**Bug: `hl.rand` crashes when a dimension is `hl.specialize(1)`**. The `_rand_codegen` in `random_ops.py` tries to look up a block ID for every dimension, but a specialized size-1 dimension has no associated tile loop. `hl.zeros`/`hl.full` don't have this problem because they only need the shape, not index variables. Fix: in `_rand_codegen` (and `_randint_codegen`), when `size == 1`, use `tl.full([1], 0, tl.int32)` as the index var and `"1"` as the size name instead of trying to allocate a reduction dimension. We applied this fix in-place in `.venv/lib/python3.12/site-packages/helion/language/random_ops.py` — it will be lost on reinstall. Filed upstream: https://github.com/pytorch/helion/issues/1397

## Autotuning

- `autotune_effort`: `"none"` / `"quick"` / `"full"`. Controlled via `HELION_AUTOTUNE_EFFORT` env var (default: `"quick"`). Tests set it to `"none"` for speed.
- `LocalAutotuneCache` caches best config per GPU on disk. Cache dir set to `helion-cache/` in the repo root via `HELION_CACHE_DIR` env var (gitignored). Different GPUs autotune independently.
- First run with a new specialization key (e.g. new `n_hidden_states` value via `hl.specialize`) triggers autotuning (~3 min for `"full"`). Subsequent runs use the cache instantly.
- Set `HELION_AUTOTUNE_ACCURACY_CHECK=0` for stochastic kernels (the kernel output changes each run, so accuracy checks would always fail).
- To force re-tuning: delete the cache dir or set `HELION_SKIP_CACHE=1`.

## Performance: barrier kernel vs two-stage

Rigorous benchmarking (25 warmup + 100 runs, RTX 3090, V=128K, D=8192, H=1) shows the **two-stage version is ~3% faster** (2.32ms vs 2.38ms median). The host-side overhead eliminated by the barrier (tensor alloc + 3 auxiliary kernel launches) is only ~0.01ms. The barrier kernel pays for cooperative launch constraints: `num_stages=1` (no pipelining), 164 persistent blocks vs 1,002 one-shot blocks, and 52% of CPI spent on barrier sync stalls.

Initial Proton profiling showed a ~5ms wall-clock advantage for the barrier version — this is a **profiling artifact**. Proton adds fixed instrumentation overhead per kernel launch, making the 4-launch two-stage appear much slower than the 1-launch barrier. Always cross-reference Proton with uninstrumented speed tests.

See `findings/helion-barrier-single-kernel.md` for full NCU and Proton analysis, and `findings/rtx3090-barrier-comparison/` for raw data.

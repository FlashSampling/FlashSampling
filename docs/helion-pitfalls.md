# Helion kernel pitfalls

Helion has an API reference: https://helionlang.com/api/index.html

and also reference examples: https://helionlang.com/examples/index.html

## `torch.argmax()` returns global indices

Inside a Helion kernel, `torch.argmax(tensor, dim=0)` returns global indices, not tile-local ones.
The generated Triton code uses `triton_helpers.max_with_index` with the tile's global `indices_0` offset baked in.
Do not add `tile_v.begin` because that double-counts the offset.

```python
# WRONG: double counts offset
new_max_idx_local = torch.argmax(summed, dim=0)
new_max_idx_global = tile_v.begin + new_max_idx_local  # BUG

# CORRECT: argmax already returns global indices
new_max_idx = torch.argmax(summed, dim=0)
```

## `for tile in hl.tile(N)` is parallel, not sequential

Each tile becomes a separate GPU program that runs in parallel.
Cross-tile communication through shared tensors is a race condition.
This can look correct for some vocabulary sizes and produce broken distributions for others.

**Fix**: Use `hl.barrier()` to synchronize stages within a single kernel:
1. **Stage 1**: Each `(V, H)` tile writes its local max/argmax to `tile_maxs[tile_v.id, :]`.
2. `hl.barrier()` for grid-wide synchronization.
3. **Stage 2**: Reduce across tiles with `argmax` + `gather` inside the same kernel.

This eliminates Python-side tensor allocations and separate reduction launches.
In the recorded RTX 3090 experiment, the barrier version was about 3% slower at H=1 because of cooperative-launch constraints.
The removed host overhead was about 0.01 ms, and the larger gap under Proton was an instrumentation artifact.
See `findings/helion-barrier-single-kernel.md` for the historical measurements.

## Tensor allocations inside kernels trigger warnings

`TensorOperationInWrapper` warning fires for tensor ops outside `hl.tile` loops. Allocate output buffers in the Python wrapper and pass them as kernel arguments instead.

## Advanced indexing does not work for gather

Helion interprets `tensor[idx_tensor, tile_var]` as a Cartesian product (producing a higher-rank result), not element-wise gather. Use `torch.gather` instead:

```python
# WRONG: Cartesian product, produces 2D
out[tile_h] = tile_max_idxs[best_tile, tile_h]  # RankMismatch error

# CORRECT: element-wise gather
out[tile_h] = torch.gather(
    tile_max_idxs[:, tile_h], 0, best_tile.unsqueeze(0)
).squeeze(0)
```

## Random number generation

Use `hl.rand([tile_v, n], seed=seed)` instead of `torch.rand` or `torch.rand_like`.
The `hl.rand` API uses Philox with per-tile offsets.
Issues #1041 and #1309 were fixed in the Helion version used for the recorded work, but should be revalidated after dependency upgrades.

The recorded Helion version crashed when `hl.rand` received a dimension specialized to one.
The original workaround patched `_rand_codegen` and `_randint_codegen` inside site-packages.
That patch was not durable and must not be assumed present in a new environment.
Reproduce the issue against the installed Helion version before applying any workaround, then use `findings/helion-hl-rand-specialize-1-bug.md` for the historical patch and minimal reproduction.

## Autotuning

- `autotune_effort`: `"none"` / `"quick"` / `"full"`. Controlled via `HELION_AUTOTUNE_EFFORT` env var (default: `"quick"`). Tests set it to `"none"` for speed.
- `LocalAutotuneCache` caches best config per GPU on disk. Cache dir set to `helion-cache/` in the repo root via `HELION_CACHE_DIR` env var (gitignored). Different GPUs autotune independently.
- First run with a new specialization key (e.g. new `n_hidden_states` value via `hl.specialize`) triggers autotuning (~3 min for `"full"`). Subsequent runs use the cache instantly.
- Set `HELION_AUTOTUNE_ACCURACY_CHECK=0` for stochastic kernels (the kernel output changes each run, so accuracy checks would always fail).
- To force re-tuning: delete the cache dir or set `HELION_SKIP_CACHE=1`.

## Performance: barrier kernel vs two-stage

Rigorous benchmarking (25 warmup + 100 runs, RTX 3090, V=128K, D=8192, H=1) shows the **two-stage version is ~3% faster** (2.32ms vs 2.38ms median). The host-side overhead eliminated by the barrier (tensor alloc + 3 auxiliary kernel launches) is only ~0.01ms. The barrier kernel pays for cooperative launch constraints: `num_stages=1` (no pipelining), 164 persistent blocks vs 1,002 one-shot blocks, and 52% of CPI spent on barrier sync stalls.

Initial Proton profiling showed a roughly 5 ms wall-clock advantage for the barrier version, but this was an instrumentation artifact.
Always cross-reference Proton with uninstrumented speed tests when launch counts differ.

See `findings/helion-barrier-single-kernel.md` for full NCU and Proton analysis, and `findings/rtx3090-barrier-comparison/` for raw data.

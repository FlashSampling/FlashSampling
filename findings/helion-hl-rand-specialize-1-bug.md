# Helion `hl.rand` crashes with `hl.specialize(1)` dimensions

## Problem

`hl.rand([tile_v, hl_n_hidden_states], seed=seed)` raises an `InductorLoweringError` when `hl_n_hidden_states = hl.specialize(n_hidden_states)` and `n_hidden_states=1`.

```
helion.exc.InductorLoweringError: Error in codegen for node unif_noise (hl.rand): list index out of range
```

The crash originates in `helion/language/random_ops.py` in `_rand_codegen`:

```python
# random_ops.py line ~108-110
rdim = env.allocate_reduction_dimension(size)
index_vars.append(state.codegen.index_var(rdim.block_id))  # IndexError here
```

Other creation ops like `hl.zeros` / `hl.full` are not affected because they only need the shape (via `tile_strategy.shape_str()`), not per-dimension index variables.

## Root cause

`hl.rand` needs index variables for each dimension to compute a unique Philox RNG offset per element. When `hl.specialize(1)` collapses a dimension to a compile-time constant of 1:

1. `env.get_block_id(1)` returns `None` (no tile loop for this dimension)
2. Falls through to `env.allocate_reduction_dimension(1)`, which allocates a new reduction dimension
3. `state.codegen.index_var(rdim.block_id)` crashes because there is no active device loop for this block ID

## Fix

A size-1 dimension has only one possible index (0), so it contributes nothing to the RNG offset. The fix adds an early check in both `_rand_codegen` and `_randint_codegen`:

```python
for i in range(ndim):
    size = tensor_shape[i]
    # Handle specialized dimensions of size 1: index is always 0,
    # no block/reduction loop needed.
    if isinstance(size, int) and size == 1:
        index_vars.append("tl.full([1], 0, tl.int32)")
        size_names.append("1")
        continue
    block_id = env.get_block_id(size)
    # ... rest unchanged
```

Using `tl.full([1], 0, tl.int32)` (not bare `0`) because the index variable participates in Triton broadcasting expressions like `index_var[None, :]`.

## How to apply

Patch `random_ops.py` in the installed Helion package:

```
.venv/lib/python3.12/site-packages/helion/language/random_ops.py
```

Add the 4-line early-return block in both `_rand_codegen` (around line 93) and `_randint_codegen` (around line 253). This fix will be lost on reinstall/upgrade.

## Reproduction

```python
import helion
import helion.language as hl
import torch

@helion.kernel(autotune_effort="none")
def fill_random_blockwise(out: torch.Tensor, seed: int):
    """Fill a [M, N] tensor with random numbers, processing M in blocks.
    Crashes when N=1 because hl.specialize(1) eliminates the dimension."""
    m = out.size(0)
    n = hl.specialize(out.size(1))  # N=1 triggers the bug
    for tile_m in hl.tile(m):
        out[tile_m, :] = hl.rand([tile_m, n], seed=seed)

# Works with N > 1
out = torch.empty(128, 4, device="cuda")
fill_random_blockwise(out, 42)  # OK

# Crashes with N = 1
out = torch.empty(128, 1, device="cuda")
fill_random_blockwise(out, 42)  # InductorLoweringError: list index out of range
```

## Upstream issue

Reported as https://github.com/pytorch/helion/issues/1397

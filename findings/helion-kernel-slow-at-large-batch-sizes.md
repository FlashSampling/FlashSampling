# Helion kernel slow at large batch sizes (H dimension not tiled)

## Problem

The Helion kernel is ~33% slower than the hand-written Triton kernel at `n_hidden_states=256`, while being slightly faster at `n_hidden_states=1`.

## Benchmark results (RTX 3090, V=256000, D=8192, num_samples=1)

| Kernel | H=1 (ms) | H=256 (ms) |
|--------|----------|-------------|
| Triton | 4.70     | 16.82       |
| Helion | 4.63     | 22.45       |

## Root cause

The Helion kernel does not tile the H (hidden states / batch) dimension. The Triton kernel does.

**Triton** — 2D grid over `[V, H]`:
- `BLOCK_SIZE_H = 64` for H=256 (via `bsz_h()` heuristic in `core.py`)
- Grid: `[cdiv(V, 128), cdiv(H, 64)]` = `[2000, 4]` = 8000 thread blocks
- Each block handles a `128 x 64` logit tile (8,192 elements)
- Uses `tl.swizzle2d` for L2 cache-friendly access across the 2D grid

**Helion** — 1D grid over `[V]` only:
- `hl.specialize(n_hidden_states)` makes H=256 a compile-time constant, but does not tile it
- Grid: 2000 thread blocks (V tiles only)
- Each block handles a `128 x 256` logit tile (32,768 elements — 4x more per block)

### Consequence 1: Less parallelism

2000 thread blocks (Helion) vs 8000 (Triton). The RTX 3090 has 82 SMs. More blocks means better occupancy and scheduling flexibility across multiple waves of work.

### Consequence 2: Higher per-block memory pressure

Each Helion block must hold `128 x 256 = 32K` float32 logits (128 KB) plus same-sized Gumbel noise and intermediates. This likely causes register spilling to local memory. Triton blocks need only `128 x 64 = 8K` elements (32 KB) — 4x less.

### Consequence 3: No L2 cache optimization

The Triton kernel uses `tl.swizzle2d` to reorder its 2D grid for spatial locality when loading `weights` and `hidden_states`. The Helion kernel has a 1D grid over V only, so each block loads the full `hidden_states[:, 0:256]` independently with no cross-block L2 reuse in the H dimension.

## Why it doesn't matter at H=1

With H=1, both kernels have equivalent structure:
- Triton: `BLOCK_SIZE_H = 16`, grid = `[2000, 1]`, each block processes `128 x 1` useful elements
- Helion: grid = 2000 blocks, each block processes `128 x 1` elements

There is no H dimension to tile, so the architectural difference disappears. Helion is marginally faster (4.63 vs 4.70 ms) because it avoids the overhead of Triton's autotuning heuristics and swizzle logic.

## Potential fix

Add a second tiling dimension to the Helion kernel:

```python
# Current: 1D tiling over V only
for tile_v in hl.tile(V, block_size=BLOCK_SIZE_V):
    logits_blk = hl.zeros([tile_v, hl_n_hidden_states], ...)
    ...

# Proposed: 2D tiling over V and H
for tile_v, tile_h in hl.tile([V, H]):
    logits_blk = hl.zeros([tile_v, tile_h], ...)
    ...
```

This would match the Triton kernel's parallelization strategy and should close the gap at large batch sizes.

## Relevant code

- `src/fused_mm_sampling/helion_impl.py` — Helion kernel (1D grid, `hl.tile(V)`)
- `src/fused_mm_sampling/core.py:226` — Triton kernel (2D grid with `BLOCK_SIZE_H` heuristic)
- `src/fused_mm_sampling/core.py:522` — `bsz_h()` heuristic for `BLOCK_SIZE_H`
- `benchmarking/logs/speed_test_bsz1.log` — H=1 benchmark
- `benchmarking/logs/speed_test_bsz256.log` — H=256 benchmark

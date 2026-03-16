# Merging stage 2 into the Helion kernel with `hl.barrier()`

## Problem

The FMMS Helion kernel uses a two-stage approach split across GPU and host:

1. **Stage 1 (GPU kernel)**: Each `(V, H)` tile computes logits + Gumbel noise, writes its local max/argmax to `tile_maxs[tile_v.id, tile_h]` and `tile_max_idxs[tile_v.id, tile_h]`.
2. **Stage 2 (Python)**: Reduces across tiles with `tile_maxs.argmax(dim=0)` + `tile_max_idxs.gather(...)`.

Stage 2 requires two separate PyTorch kernel launches (argmax, gather) plus the overhead of returning control to the Python wrapper between stages. For V=128K with `BLOCK_SIZE_V=128`, there are 1000 tiles; the stage-2 argmax costs ~0.1-0.5ms. For V=256K the cost rises to ~1ms.

## Observation

Helion supports `hl.barrier()`, a grid-wide synchronization primitive. The [split-k matmul example](https://helionlang.com/examples/split_k_barrier.html) demonstrates the pattern: two `hl.tile` loops separated by `hl.barrier()` inside a single kernel. Stage 1 writes partial results to a temporary buffer, the barrier ensures all tiles complete, then stage 2 reduces the partials.

The split-k example:

```python
@helion.kernel(static_shapes=True, dot_precision="ieee")
def split_k_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    m, k = a.shape
    _, n = b.shape
    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(16, 512, 64))
    block_k = helion.next_power_of_2(helion.cdiv(k, split_k))
    tmp = torch.zeros((m, n, split_k), device=a.device, dtype=a.dtype)
    out = torch.empty((m, n), device=a.device, dtype=a.dtype)

    for tile_m, tile_n, tile_k_outer in hl.tile([m, n, k], block_size=[None, None, block_k]):
        acc = hl.zeros([tile_m, tile_n], device=a.device, dtype=a.dtype)
        for tile_k_inner in hl.tile(tile_k_outer.begin, tile_k_outer.end):
            acc = torch.addmm(acc, a[tile_m, tile_k_inner], b[tile_k_inner, tile_n])
        tmp[tile_m, tile_n, tile_k_outer.id] = acc

    hl.barrier()

    for tile_m, tile_n in hl.tile([m, n]):
        out[tile_m, tile_n] = torch.sum(tmp[tile_m, tile_n, :], dim=-1)

    return out
```

## Proposed change

Merge both stages into a single Helion kernel using `hl.barrier()`:

```python
@helion.kernel(...)
def fused_sample_helion_kernel(
    weights: torch.Tensor,       # [V, D]
    hidden_states: torch.Tensor, # [D, H]
    out_idxs: torch.Tensor,     # [H] — final sampled token indices
    temperature: float,
    seed: int,
):
    V, D = weights.size()
    H = hidden_states.size(1)
    n_tiles_v = helion.cdiv(V, BLOCK_SIZE_V)

    tile_maxs = torch.full(
        (n_tiles_v, H), float("-inf"), device=weights.device, dtype=torch.float32
    )
    tile_max_idxs = torch.empty(
        (n_tiles_v, H), dtype=torch.long, device=weights.device
    )

    # Stage 1: per-tile matmul + Gumbel-max (parallel over V and H)
    for tile_v, tile_h in hl.tile([V, H], block_size=[BLOCK_SIZE_V, BLOCK_SIZE_H]):
        logits_blk = hl.zeros([tile_v, tile_h], dtype=torch.float32)
        for tile_d in hl.tile(D):
            mm = torch.matmul(weights[tile_v, tile_d], hidden_states[tile_d, tile_h])
            logits_blk = logits_blk + mm
        logits_blk = logits_blk / temperature

        unif_noise = hl.rand([tile_v, tile_h], seed=seed)
        gumbel_noise = -(-unif_noise.log()).log()
        summed = logits_blk + gumbel_noise

        tile_maxs[tile_v.id, tile_h] = hl.reduce(torch.max, summed, dim=0)
        tile_max_idxs[tile_v.id, tile_h] = torch.argmax(summed, dim=0)

    hl.barrier()

    # Stage 2: reduce across V-tiles to find global argmax (parallel over H)
    for tile_h in hl.tile(H):
        best_tile = torch.argmax(tile_maxs[:, tile_h], dim=0)
        out_idxs[tile_h] = tile_max_idxs[best_tile, tile_h]
```

The Python wrapper simplifies to just calling the kernel and collecting `out_idxs` — no more buffer allocation or host-side reduction.

## Result: barrier version is ~3% slower (RTX 3090, H=1)

The barrier version was implemented and passes all correctness tests. Rigorous benchmarking shows it is slightly slower at the kernel level, with the host-side overhead it eliminates being negligible in practice.

### Speed test (25 warmup + 100 runs, no profiling overhead)

RTX 3090, V=128256, D=8192, H=1:

| Version | Median (ms) | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---------|-------------|-----------|----------|----------|----------|
| Two-stage | 2.32 | 2.32 | 0.036 | 2.31 | 2.67 |
| Barrier | 2.38 | 2.39 | 0.014 | 2.37 | 2.52 |

The barrier kernel is **~3% slower** (2.38ms vs 2.32ms median), but has **lower variance** (std 0.014 vs 0.036).

### NCU kernel analysis

| Metric | Barrier | Two-Stage |
|--------|---------|-----------|
| Kernel duration | 2.39ms | 2.31ms |
| Grid size | 164 blocks | 1,002 blocks |
| Block size | 512 threads | 64 threads |
| Registers/thread | 64 | 62 |
| SM compute throughput | 29.6% | 7.3% |
| Achieved occupancy | 66.6% | 12.4% |
| Theoretical occupancy | 66.7% (register-limited) | 12.5% (shared-memory-limited) |
| Active warps/scheduler | 8.02 | 1.49 |
| L1 hit rate | 2.2% | 0.7% |
| L2 hit rate | 1.6% | 0.08% |
| Top stall reason | CTA barrier (52% of CPI) | L1TEX scoreboard (85% of CPI) |
| Issue rate | 1 insn every 6.7 cycles | 1 insn every 34.9 cycles |

The barrier kernel uses hardware **much** more efficiently (5.4x occupancy, 4x compute throughput, 5x issue rate). However, it executes 2x more FP32 instructions because 164 persistent blocks each iterate over multiple tiles, vs 1,002 one-shot blocks in the two-stage version. The net effect is a ~3% slower kernel.

The barrier kernel's top stall is CTA barrier sync (52%) — warps finish tile work and wait for siblings. The two-stage kernel is 85% stalled on memory (L1TEX scoreboard) because with only 1.49 active warps, there's nothing to hide memory latency.

### Proton profiling (5 warmup + 20 runs) — beware of artifacts

| Version | Median (ms) | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---------|-------------|-----------|----------|----------|----------|
| Two-stage | 30.12 | 30.04 | 1.21 | 27.36 | 32.60 |
| Barrier | 25.19 | 24.83 | 2.09 | 18.77 | 27.30 |

**The ~5ms proton difference is an instrumentation artifact, not real overhead.** Proton adds fixed overhead per kernel launch and per CUDA operation. The two-stage version launches 4 CUDA kernels (main + fill + argmax + gather) vs 1 for barrier, so it accumulates ~5ms more profiling overhead. The uninstrumented speed test shows the actual host-side overhead is only ~0.01ms (2.32ms wall-clock vs 2.31ms NCU kernel-only for two-stage).

**Lesson learned**: Always cross-reference proton wall-clock times with uninstrumented `speed_test.py` measurements. Proton is useful for relative kernel-internal breakdowns but misleading for comparing approaches with different numbers of kernel launches.

### Why the barrier version is slower

The host-side overhead eliminated by barrier (tensor allocations + 3 auxiliary kernel launches) is **negligible** at H=1 — only ~0.01ms. The barrier version pays more than it saves:
1. **Cooperative launch constraints**: `num_stages=1` (no software pipelining), `persistent_blocked` scheduling, 164 blocks vs 1,002
2. **Barrier sync stalls**: 52% of cycles spent waiting at CTA barrier
3. **2x instruction count**: persistent blocks iterate over multiple tiles

### Tradeoffs

- **Cooperative launch overhead**: Requires `launch_cooperative_grid=True`, constraining all thread blocks to be co-resident. Forces `persistent_blocked` scheduling with fewer blocks (164 vs 1,002) and `num_stages=1`.
- **Higher occupancy but wasted on sync**: 66.6% occupancy vs 12.4%, but 52% of CPI is barrier stalls. The two-stage version's low occupancy doesn't hurt because the work is purely memory-bound — each block independently streams through its tile.
- **Lower variance**: std 0.014 vs 0.036, likely from predictable persistent scheduling.
- **`autotune_ignore_errors=True` required**: Some autotuner configs exceed the cooperative launch block limit, reducing the search space.
- **Potential at larger H**: The host-side overhead grows with batch size (argmax/gather over `n_tiles x H`). At larger H the barrier version may break even or win.

## Open questions (resolved)

1. **Tensor allocation inside kernel**: Works fine. `torch.full(...)` and `torch.empty(...)` inside the kernel (but outside `hl.tile` loops) compile and run correctly. The `TensorOperationInWrapper` warning fires but does not block compilation.

2. **Stage 2 argmax over `:` slice**: `torch.argmax(tile_maxs[:, tile_h], dim=0)` works. However, advanced indexing with `tile_max_idxs[best_tile, tile_h]` does **not** work — Helion interprets it as a Cartesian product (producing a 2D tensor), not element-wise gather. **Fix**: use `torch.gather(tile_max_idxs[:, tile_h], 0, best_tile.unsqueeze(0)).squeeze(0)` instead.

3. **`hl.barrier()` version support**: Confirmed available via `hasattr(hl, 'barrier')`.

4. **Interaction with 2D tiling**: Works correctly. Stage 1 tiles `(V, H)`, stage 2 tiles `(H)` only. The barrier properly synchronizes all stage-1 tiles before stage-2 runs.

## Profile details

RTX 3090 comparison results checked into `findings/rtx3090-barrier-comparison/`:
- `{barrier,two-stage}/speed-test.txt` — full speed_test.py output
- `{barrier,two-stage}/time-distribution.csv` — statistical summary
- `{barrier,two-stage}/proton/kernel-by-line.txt` — per-line kernel breakdown
- `{barrier,two-stage}/proton/speed-test.txt` — proton run output

Full NCU reports (`.ncu-rep`) and Proton hatchet files are in `benchmarking/profiles/barrier-comparison/` (gitignored).

**Two-stage kernel config**: `num_stages=2, num_warps=2, pid_type='flat'`, 1002 blocks, `block_sizes=[1, 128]`
**Barrier kernel config**: `num_stages=1, num_warps=16, pid_type='persistent_blocked'`, 164 blocks, `block_sizes=[1, 128, 1]`

Both are memory-bound: weight matrix loads dominate kernel time (~55% in two-stage, split across `triton_helpers` lines in barrier due to Helion codegen differences).

## Relevant code

- `src/fused_mm_sampling/helion_impl.py` — barrier version (current)
- Helion split-k barrier example: https://helionlang.com/examples/split_k_barrier.html

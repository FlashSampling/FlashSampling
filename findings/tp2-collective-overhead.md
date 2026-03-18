# TP2 Collective Overhead Analysis (B200, 2026-03-16)

## Setup

GPU: NVIDIA B200 x2 (NVLink), CUDA 13.0, PyTorch 2.10.0, Triton 3.6.0.
Benchmarks run on Modal. Each data point is the median of 6 independent runs (Triton `do_bench` internally).

Two configs:
- **Large**: V=128,256, d=8,192 (Llama 3 70B style)
- **Small**: V=151,936, d=4,096 (Qwen3 8B style)

## Method

To isolate the collective overhead, compare:
- **TP2**: per-GPU compute (V/2) + collective ops
- **TP1 at V/2**: per-GPU compute only (same V/2, no collectives)

The difference is the collective overhead.

TP1 at V/2 was measured by adding `large-halfv` (V=64,128, d=8,192) and `small-halfv` (V=75,968, d=4,096) benchmark cases.
FMMS was measured across 6 runs, fi:sample across 6 runs.

## FMMS TP2 code path (before symmetric memory)

After the Triton kernel produces per-tile maxes `[num_samples, n_tiles, H]`, the reduction had 4 sequential ops:

1. `_local_reduce` (compiled kernel) - reduce across V-tiles on this rank
2. `dist.all_gather(max_values)` - gather `[H, 1]` scalars from all ranks
3. `dist.all_gather(samples)` - gather `[H, 1]` indices from all ranks
4. `_stack_and_select_winner` (compiled kernel) - pick global winner

## fi:sample TP2 code path

After the matmul produces local logits `[H, V/2]`:

1. `dist.all_gather(logits)` - gather `[H, V/2]` from all ranks, producing `[H, V]`
2. `flashinfer.sampling.sampling_from_logits` on full V

## Results: FMMS collective overhead

### Large config (V=128K, d=8192)

| H | TP1@V/2 | TP2 | overhead (ms) | overhead % |
|---|---------|-----|---------------|------------|
| 1 | 0.190 | 0.304 | 0.114 | 37% |
| 8 | 0.193 | 0.331 | 0.138 | 42% |
| 32 | 0.196 | 0.354 | 0.159 | 45% |
| 64 | 0.204 | 0.347 | 0.143 | 41% |
| 128 | 0.236 | 0.337 | 0.101 | 30% |
| 256 | 0.381 | 0.380 | -0.001 | 0% |

### Small config (V=152K, d=4096)

| H | TP1@V/2 | TP2 | overhead (ms) | overhead % |
|---|---------|-----|---------------|------------|
| 1 | 0.128 | 0.329 | 0.201 | 61% |
| 8 | 0.131 | 0.341 | 0.210 | 62% |
| 32 | 0.136 | 0.339 | 0.203 | 60% |
| 64 | 0.142 | 0.363 | 0.220 | 61% |
| 128 | 0.161 | 0.342 | 0.181 | 53% |
| 256 | 0.267 | 0.341 | 0.074 | 22% |

## Results: fi:sample collective overhead

### Large config

| H | TP1@V/2 | TP2 | overhead (ms) | overhead % |
|---|---------|-----|---------------|------------|
| 1 | 0.214 | 0.263 | 0.049 | 19% |
| 8 | 0.210 | 0.263 | 0.053 | 20% |
| 32 | 0.236 | 0.308 | 0.072 | 23% |
| 64 | 0.243 | 0.327 | 0.084 | 26% |
| 128 | 0.259 | 0.389 | 0.130 | 33% |
| 256 | 0.337 | 0.578 | 0.242 | 42% |

### Small config

| H | TP1@V/2 | TP2 | overhead (ms) | overhead % |
|---|---------|-----|---------------|------------|
| 1 | 0.161 | 0.285 | 0.124 | 44% |
| 8 | 0.159 | 0.293 | 0.135 | 46% |
| 32 | 0.165 | 0.265 | 0.100 | 38% |
| 64 | 0.171 | 0.287 | 0.116 | 40% |
| 128 | 0.196 | 0.347 | 0.152 | 44% |
| 256 | 0.279 | 0.578 | 0.299 | 52% |

## Key findings

### 1. FMMS overhead is constant, fi:sample overhead grows with H

FMMS communicates tiny `[H, 1]` scalars via 2x all_gather.
The overhead is ~0.12-0.16ms (large) and ~0.20ms (small), **constant across H=1-64**.
This is dominated by the fixed latency of 4 sequential ops (kernel launch + NCCL collective latency), not data volume.

fi:sample communicates `[H, V/2]` logits via 1x all_gather, then samples on V instead of V/2.
The overhead starts small (0.049ms large, 0.124ms small at H=1) but **grows linearly with H** (0.242ms, 0.299ms at H=256) because the data volume scales with batch size.

### 2. FMMS overhead is higher than fi:sample at low H despite less data

At H=1 large: FMMS overhead = 0.114ms, fi:sample = 0.049ms.
FMMS sends bytes, fi:sample sends 125 KB. On B200 NVLink, even 125 KB transfers in nanoseconds.
The difference is the number of sequential ops: FMMS has 4 (local_reduce + 2x all_gather + select_winner), fi:sample has 2 (all_gather + sampling delta).

### 3. At H=256, FMMS overhead vanishes

At H=256 large, TP1@V/2 = 0.381ms matches TP2 = 0.380ms.
The per-GPU compute is large enough that the collective cost is fully hidden.
fi:sample's overhead keeps growing at H=256 (0.242ms) because the logit all_gather scales with H.

### 4. TP2 compiled baseline is misleading

On TP2, `torch.compile` falls back to `sample_compiled_with_breaks` (no `fullgraph=True`) because collective ops cause graph breaks.
This makes compiled slower than eager at low batch sizes (e.g., 0.469ms compiled vs 0.300ms eager at H=1 small).
Relative performance numbers using compiled as baseline are inflated on TP2.

### 5. fi:sample overhead at low H differs between large and small

fi:sample overhead at H=1: 0.049ms (large) vs 0.124ms (small).
This is because the overhead includes not just the all_gather but also the delta cost of sampling on V_full vs V/2.
Small has larger V (152K vs 128K), so the sampling delta is bigger.

## Optimization attempts

### Attempt 1: Skip local_reduce, all_gather raw per-tile outputs

Idea: skip `_local_reduce`, all_gather the raw `[num_samples, n_tiles, H]` tensors, reduce across all tiles from all ranks in one compiled kernel.
This reduces from 4 ops to 3 ops.

Result: **No measurable improvement.** The all_gather data increased from bytes to ~6 KB (501 tiles), but the saved kernel launch (~25us) was within noise.

### Attempt 2: async_op=True for overlapping all_gathers

Idea: issue both all_gathers as async ops (`async_op=True`), then `.wait()` on both.
This should avoid CPU blocking between the two collectives.

Result: **~0.1ms slower.** The explicit `.wait()` calls add synchronization overhead that exceeds the savings.
NCCL's blocking collectives on CUDA are already non-blocking on the CPU for GPU work.
They enqueue work on the NCCL stream without waiting for completion.
The "blocking" is just a CPU-side wait for the operation to be *enqueued*, not *completed*.

## Conclusion

The FMMS TP2 collective overhead (~0.12-0.20ms) was a fixed cost from NCCL collective latency, not from data volume or CPU blocking.
Python-level optimizations (reordering ops, async mode) could not reduce it.
The only path to eliminate it was fusing the collective into the Triton kernel itself.

## Solution: symmetric memory (implemented 2026-03-18)

The kernel output buffers (`maxs`, `maxs_idx`) are allocated in symmetric memory via `torch.distributed._symmetric_memory.get_symm_mem_workspace`.
This means the kernel's existing TMA stores write directly to NVLink-mapped addresses, visible to all ranks.
No NCCL collectives are needed.

### How it works

1. `allocate_symm_mem_outputs()` creates `maxs` and `maxs_idx` as views into the symmetric memory workspace (one per rank).
2. The main Triton kernel writes per-tile Gumbel-max results via TMA stores (unchanged kernel code, just different backing memory).
3. `symm_mem_hdl.barrier()` ensures all ranks' kernel writes are visible.
4. Each rank reads all ranks' per-tile outputs from symmetric memory, runs `_local_reduce` per rank, and picks the global winner via `_stack_and_select_winner`.

This eliminates the 2x NCCL all_gather entirely.
The reductions are cheap and stay on the host side.

### Implementation details

- `src/fused_mm_sampling/kraken_reduce.py` contains `allocate_symm_mem_outputs()` and `kraken_post_kernel_reduce()`.
- The `maxs_idx` (int64) buffer must start at a 128-byte-aligned offset after `maxs` (bfloat16) for TMA compatibility.
- The workspace is allocated once per process group and reused across calls.
- Requires NVLink-connected GPUs, PyTorch >= 2.6, CUDA >= 12.4.
- Cannot be tested on a single GPU (symmetric memory requires distinct devices per rank).

### Benchmark results (B200 x2, averaged over 5 runs)

#### Large config (V=128K, d=8192)

| H | FMMS (ms) | Eager (ms) | Compiled (ms) | FlashInfer (ms) | FI top-k/p (ms) |
|---|-----------|------------|---------------|-----------------|------------------|
| 1 | 0.246 | 0.383 | 0.468 | 0.307 | 0.364 |
| 8 | 0.289 | 0.384 | 0.501 | 0.329 | 0.391 |
| 32 | 0.276 | 0.453 | 0.489 | 0.320 | 0.383 |
| 64 | 0.280 | 0.493 | 0.494 | 0.335 | 0.391 |
| 128 | 0.275 | 0.639 | 0.542 | 0.393 | 0.456 |
| 256 | 0.383 | 0.997 | 0.787 | 0.590 | 0.702 |

#### Small config (V=152K, d=4096)

| H | FMMS (ms) | Eager (ms) | Compiled (ms) | FlashInfer (ms) | FI top-k/p (ms) |
|---|-----------|------------|---------------|-----------------|------------------|
| 1 | 0.254 | 0.300 | 0.523 | 0.334 | 0.383 |
| 8 | 0.266 | 0.328 | 0.497 | 0.328 | 0.384 |
| 32 | 0.278 | 0.384 | 0.502 | 0.333 | 0.401 |
| 64 | 0.279 | 0.439 | 0.508 | 0.324 | 0.383 |
| 128 | 0.271 | 0.611 | 0.556 | 0.348 | 0.441 |
| 256 | 0.290 | 1.016 | 0.895 | 0.576 | 0.728 |

FMMS with symmetric memory is fastest across all batch sizes in both configs.
At H=256, it is 1.5-2x faster than FlashInfer and 2.5-3.5x faster than Eager.

Compared to the previous NCCL-based TP2 results (see tables above), the symmetric memory approach reduced FMMS TP2 latency at H=1 from 0.304ms to 0.246ms (large) and from 0.329ms to 0.254ms (small).

### Potential follow-ups

**1. Reduce NVLink traffic by exchanging reduced winners instead of per-tile data.**
Currently `kraken_post_kernel_reduce` reads each remote rank's full per-tile outputs (`[num_samples, grid_size_v, H]`, ~1.3MB per remote rank at H=256) over NVLink, then runs `_local_reduce` on each.
Instead: run `_local_reduce` locally first, write only the reduced `[H, num_samples]` winners (~2KB) to symmetric memory, barrier, then read remote winners and `_stack_and_select_winner`.
This cuts NVLink traffic by ~500x while still avoiding NCCL.
The write would be a simple `tensor.copy_()` to a symmetric memory view.

**2. Fuse `_local_reduce` x world_size + `_stack_and_select_winner` into one compiled kernel.**
Currently 3 compiled kernel launches for TP2 (2x `_local_reduce` + 1x `_stack_and_select_winner`).
A single `@torch.compile(fullgraph=True)` function that takes all ranks' buffers and does both reductions in one pass would save kernel launch overhead.

Both are small wins given the reductions are already cheap relative to the matmul.

### Rejected approach: pre-kernel all_gather of weights

All_gather the weight shards so each GPU has full V, then run the normal TP1 kernel.
This eliminates the post-kernel collective but doubles the memory read per GPU (full V instead of V/2).
At low H the kernel is memory-bound, so reading 2x the weights would roughly double compute time, negating the benefit.

## TP4 results (B200 x4, 2026-03-18)

5 independent runs on Modal. Results show bimodality in absolute times across all providers (runs {1,3} ~20% faster than runs {2,4}, a machine-level effect), so both per-run and median values are reported.

### FMMS speedup vs PyTorch Compiled (median of 5 runs)

| H | TP4 | TP2 | TP1 |
|---|-----|-----|-----|
| **Large config** | | | |
| 1 | 1.26x | 1.89x | 1.43x |
| 8 | 1.18x | 1.80x | 1.32x |
| 64 | 1.19x | 1.77x | 1.52x |
| 128 | 1.18x | 1.97x | 1.44x |
| 256 | 1.92x | 2.06x | 1.15x |
| **Small config** | | | |
| 1 | 1.38x | 2.05x | 1.46x |
| 8 | 1.22x | 1.90x | 1.53x |
| 64 | 1.16x | 1.79x | 1.84x |
| 128 | 1.46x | 2.05x | 1.89x |
| 256 | 2.26x | 3.08x | 1.58x |

### FMMS is the only provider that regresses at TP4

Comparing TP4 fast-mode runs (1,3) against the full TP2 range to control for machine-level bimodality:

| Provider | H | TP4 fast (ms) | TP2 range (ms) | TP4/TP2 median |
|----------|---|---------------|----------------|----------------|
| FMMS | 1 | 0.290 | 0.226-0.309 | 1.25 (slower) |
| FMMS | 64 | 0.359 | 0.257-0.294 | 1.26 (slower) |
| FMMS | 256 | 0.352 | 0.380-0.386 | 0.92 (faster) |
| FI_sampling | 1 | 0.261 | 0.300-0.324 | 0.86 (faster) |
| FI_sampling | 64 | 0.274 | 0.331-0.337 | 0.82 (faster) |
| FI_sampling | 256 | 0.506 | 0.582-0.601 | 0.86 (faster) |
| FI_topk | 1 | 0.303 | 0.341-0.381 | 0.83 (faster) |
| FI_topk | 64 | 0.333 | 0.387-0.392 | 0.85 (faster) |
| FI_topk | 256 | 0.626 | 0.696-0.705 | 0.89 (faster) |
| Compiled | 1 | 0.366 | 0.431-0.561 | 0.83 (faster) |
| Compiled | 64 | 0.432 | 0.437-0.518 | 0.86 (faster) |
| Compiled | 256 | 0.714 | 0.774-0.794 | 0.90 (faster) |

All baselines get ~15% faster at TP4 vs TP2 (benefiting from quartering the matmul). FMMS gets ~25% slower for H=1-128. Only at H=256 does FMMS improve at TP4. The cause of the FMMS regression is not yet understood.

### Conclusion

TP2 is the sweet spot for FMMS. At TP4, FMMS's advantage over baselines drops back to roughly TP1 levels (1.2-1.4x) for H < 256.

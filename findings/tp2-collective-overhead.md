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

## FMMS TP2 code path

After the Triton kernel produces per-tile maxes `[num_samples, n_tiles, H]`, the reduction has 4 sequential ops:

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

The FMMS TP2 collective overhead (~0.12-0.20ms) is a fixed cost from NCCL collective latency, not from data volume or CPU blocking.
Python-level optimizations (reordering ops, async mode) cannot reduce it.
The only path to eliminate it is fusing the collective into the Triton kernel itself.

## Next step: fuse the all_gather into the Triton kernel via symmetric memory

[Kraken](https://github.com/meta-pytorch/kraken) (Meta) provides Triton-level communication primitives that bypass NCCL entirely, using **symmetric memory** (direct NVLink peer access) and PTX barrier instructions.

Kraken's `all_gather_matmul` demonstrates the pattern: a persistent Triton kernel polls a `progress` tensor to consume data chunks as they arrive from remote GPUs via a DMA copy engine on a separate stream.

### How FMMS differs from Kraken's all_gather_matmul

Kraken gathers the *input* matrix before the matmul.
In FMMS TP2, each GPU already has its local weight shard, so the matmul needs no all_gather.
The collective happens *after* the kernel, to exchange the Gumbel-max winners (`[H, 1]` scalars per rank).

### Two approaches considered

**Approach 1 (promising): post-kernel symmetric memory exchange.**
After the Triton kernel computes local max values + indices, use symmetric memory to write them directly to peer GPU memory from within the kernel's epilogue.
The kernel on each GPU could then poll for the remote values and do the final argmax, eliminating all 4 Python-side post-kernel ops.
The data exchanged is tiny (`[H, 1]` scalars per rank), so the symmetric memory write + barrier would be much faster than NCCL collectives.
Kraken's `symm_mem_sync` and `wait_gmem_barrier` PTX primitives are the building blocks.

**Approach 2 (not promising): pre-kernel all_gather of weights (Kraken style).**
All_gather the weight shards so each GPU has full V, then run the normal TP1 kernel.
This eliminates the post-kernel collective but doubles the memory read per GPU (full V instead of V/2).
At low H the kernel is memory-bound, so reading 2x the weights would roughly double compute time, negating the benefit.

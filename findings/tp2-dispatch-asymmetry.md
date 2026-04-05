# TP2 dispatch asymmetry

## Observation

nsys traces of TP2 runs on Modal B200x2 show that one rank dispatches kernels ~300-700us slower per iteration than the other. The slow rank has large inter-iteration gaps on the GPU timeline, while the fast rank runs back-to-back. The fast rank then spin-waits in the symmetric memory barrier for the slow rank to catch up.

The kernel execution itself is symmetric (~120us on both ranks). The asymmetry is entirely in host-side dispatch overhead.

Which rank is slow varies by run (not consistently device 0 or device 1), confirming this is a scheduling effect, not a hardware asymmetry.

## What the barrier actually measures

The barrier asymmetry does not measure absolute dispatch speed. It measures the **difference** between the two ranks' dispatch speeds. A symmetric barrier (both ~5us) can mean either:
- Both ranks dispatch fast (good), or
- Both ranks dispatch equally slow (bad but symmetric)

This was observed in the NUMA-bound experiments. Comparing two runs with identical hardware topology (both GPUs on NUMA node 0, both ranks pinned to node 0, CPUs 10-47):

**numabound-r3 (symmetric, both slow):**

| Rank | fmms (us) | Gap to next (us) | Kernels in gap (us) | Idle (us) | Barrier (us) |
|------|-----------|-------------------|---------------------|-----------|--------------|
| 0    | 120       | ~450              | 52                  | ~400      | 9.6          |
| 1    | 120       | ~450              | 51                  | ~400      | 5.5          |

**numabound-r5 (asymmetric, one fast):**

| Rank | fmms (us) | Gap to next (us) | Kernels in gap (us) | Idle (us) | Barrier (us) |
|------|-----------|-------------------|---------------------|-----------|--------------|
| 0    | 123       | ~350              | 54                  | ~290      | 5.8          |
| 1    | 124       | ~72               | 52                  | ~19       | 283          |

In r3, both ranks spend ~400us idle between kernel launches, so they arrive at the barrier together. In r5, rank 1 dispatches with only 19us idle (15x faster) and spin-waits 283us for rank 0. The 6 intermediate GPU kernels take the same ~52us in both cases. The difference is purely CPU idle time between kernel launches.

## CPU-side root cause analysis

Per-rank nsys traces include CUDA runtime API calls. Both ranks execute identical code and issue the same total number of CUDA API calls (142 each, identical breakdown: 60 cudaLaunchKernel, 20 cudaEventCreate, 20 cudaEventRecord, 10 cuLaunchKernelEx, 10 cuLaunchKernel, etc.).

The difference is **when** these calls happen relative to the GPU timeline:

**Slow rank (rank 0, ~880us gap):** 11-12 CPU calls are visible in each inter-iteration gap. The CPU has fallen behind the GPU: by the time the barrier kernel finishes, the CPU hasn't yet issued the post-reduce kernels (_local_reduce, _stack_and_select_winner), so the GPU sits idle waiting for launches. Large idle periods (100-200us) between CPU calls indicate the CPU is busy with Python/torch overhead.

**Fast rank (rank 1, ~71us gap):** Only 1-2 CPU calls appear in the gap. The remaining 10+ calls for that iteration were issued *before* the gap, while the GPU was still executing the previous fmms_kernel. The fast rank's CPU runs ahead of the GPU, pre-queuing work so kernels execute back-to-back.

The asymmetry is therefore about whether the CPU can keep ahead of the GPU pipeline. When one rank's CPU falls behind (possibly due to OS scheduling, cache misses, or other contention on shared cloud hardware), its GPU stalls waiting for kernel launches.

## Ruled out

- **mp.spawn parent process overhead**: torchrun (fully independent processes, no parent sentinel polling) shows the same asymmetry pattern. mp.spawn is not the cause.
- **Hardware asymmetry**: Which rank is slow varies by run, not consistently tied to one device.
- **NUMA distance**: NUMA pinning (binding each rank to its GPU's NUMA node) did not eliminate the asymmetry. With 5 NUMA-bound runs: median asymmetry 277us (vs 364us unbound). The reduction is modest and within variance. Two runs with identical topology (both GPUs on node 0, both ranks bound to node 0) produced opposite results: r3 was symmetric (4us), r5 was asymmetric (277us). NUMA contributes but is not the primary cause.

## All providers affected equally

The asymmetry is not specific to FMMS's symmetric memory barrier. All three providers show the same pattern (torchrun, B200x2, 10 timed iterations):

| Provider        | Sync mechanism  | Slow rank sync median (us) | # runs |
|-----------------|-----------------|---------------------------|--------|
| fused-triton    | symm barrier    | 327-685                   | 8      |
| naive-compiled  | NCCL AllGather  | 228-638                   | 3      |
| flashinfer      | NCCL AllGather  | 421-498                   | 2      |

This confirms the root cause is host-side dispatch overhead, not the synchronization mechanism.

## NUMA binding experiment

`benchmarking/nsys_wrapper.py` pins each rank to the NUMA node of its GPU (via `os.sched_setaffinity`). Comparing 5 NUMA-bound runs vs 13 unbound runs:

| Condition  | Median asymmetry (us) | Range (us) |
|------------|----------------------|------------|
| Unbound    | 364                  | 132-411    |
| NUMA-bound | 277                  | 4-405      |

The lower bound (4us in numabound-r3) shows that near-zero asymmetry is possible, but not reliably achieved with NUMA binding alone. The asymmetry appears to be driven by OS-level CPU scheduling noise on shared cloud hardware.

## CPU-side profiling blocked on Modal

Modal uses gVisor (Google's sandboxed container runtime), which does not implement the `perf_event_open` syscall (returns ENODEV). `perf_event_paranoid` is set to 4. This means nsys `--sample=process-tree` and `--cpuctxsw=process-tree` cannot collect CPU sampling or context switch data. Further CPU-level investigation requires bare metal.

## Impact

At bsz=1, the dispatch asymmetry adds ~300-700us per iteration to the slower rank. The symmetric memory barrier absorbs this as spin-wait time on the fast rank. The impact at higher batch sizes has not been measured.

## Not yet investigated

- **CUDA driver contention**: Two processes issuing CUDA work to different GPUs may contend for CUDA driver locks.
- **CPU context switches**: Could confirm if the slow rank is being preempted. Requires bare metal with perf_event_paranoid <= 1.

## nsys profiling notes

### Per-rank nsys (recommended for torchrun)

nsys cannot capture both devices when wrapping torchrun from outside (`nsys profile torchrun ...`). The `--capture-range=cudaProfilerApi` only captures the first child process's CUDA context. Solution: each torchrun worker launches its own nsys instance via `benchmarking/nsys_wrapper.py`, producing separate `.nsys-rep` files per rank. The wrapper also handles NUMA binding and exit code normalization (nsys exits with 143/SIGTERM after cudaProfilerStop, which is treated as success).

### dist.barrier() breaks CUPTI capture

Calling `dist.barrier()` (NCCL AllReduce) inside a `--capture-range=cudaProfilerApi` window prevents CUPTI from recording subsequent kernels on device 1. This appears to be an nsys bug. Workaround: do not place NCCL collectives between `cudaProfilerStart` and the region of interest.

## nsys profiles

Stored in `benchmarking/modal-results/nsys-profiles/b200/tp2/case-small/`. Key directories:
- `bsz1-numabound-r{1..5}/`: torchrun + per-rank nsys + NUMA binding (fused-triton, 10 timed iterations)
- `bsz1-numa-r{1..15}/`: torchrun + per-rank nsys, no NUMA binding (fused-triton, 10 timed iterations)
- `bsz1-perrank-r{1..8}/`: torchrun + per-rank nsys (fused-triton + naive-compiled + flashinfer)
- `nsys-profiles-before-kraken-refactor/`: older mp.spawn profiles for comparison

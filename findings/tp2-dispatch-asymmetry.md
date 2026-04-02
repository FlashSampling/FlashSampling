# TP2 dispatch asymmetry: device 1 kernel gaps

## Observation

nsys traces of TP2 runs on Modal B200x2 show that device 0's GPU queue is tightly packed (1-5us gaps between kernels), while device 1 consistently has 50-250us idle gaps between kernel launches. This causes device 0 to race ahead each iteration, with the barrier (or NCCL AllGather) absorbing the skew.

Typical device 1 steady-state gaps per iteration:
- pre-`_local_reduce`: 30-80us
- pre-`fill` (next iteration setup): 80-110us
- pre-`fmms_kernel`: 180-280us

Both the FMMS (symmetric memory + barrier) and naive-compiled (NCCL AllGather) code paths exhibit the same pattern. The asymmetry is in host-side dispatch, not in kernel execution (both devices run `fmms_kernel` in ~121us).

## Ruled out: NUMA

NUMA pinning via `torch.numa.binding` was tested. On a Modal run where both GPUs were on the **same NUMA node** (node 1), the dispatch gaps persisted with identical magnitude. NUMA is not the cause.

Additionally, Modal's Kubernetes-based scheduling often places CPUs from only one NUMA node in the container's cgroup, while assigning GPUs from both NUMA nodes. This causes `torch.numa.binding` to fail for the GPU on the other NUMA node (empty CPU intersection). This is a known Kubernetes Topology Manager limitation (kubernetes/kubernetes#122295).

## Likely cause: mp.spawn process model

The benchmark uses `torch.multiprocessing.spawn` (start_method="spawn") which creates child processes from a parent Python process. Despite being independent OS processes, device 0's process consistently dispatches kernels faster. Possible contributing factors:
- Parent process overhead (monitoring sentinels in `ProcessContext.join()`) competing for CPU with child processes.
- OS scheduler bias toward the first-spawned process.
- nsys profiler overhead distributed asymmetrically across processes.

None of these have been confirmed. The asymmetry may also be inherent to how CUDA dispatches from two processes to two GPUs on the same PCIe root complex.

## Impact

At bsz=1 (the worst case), the dispatch gaps add ~350-500us of idle time per iteration on device 1. At higher batch sizes the kernel runtime dominates and the relative impact shrinks.

## Open questions

- Does `torchrun` (fully independent OS processes) eliminate the asymmetry?
- Does the asymmetry exist without nsys profiling?
- Is there a way to pre-queue work on device 1's CUDA stream to hide the dispatch latency?

## nsys profiles

Stored in `benchmarking/modal-results/nsys-profiles/b200/tp2/case-small/`. The `-numa` suffix indicates runs with NUMA binding enabled.

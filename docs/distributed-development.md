# Distributed development

## Symmetric-memory TP reduction

`src/fused_mm_sampling/tensor_parallel_reduce.py` replaces NCCL all-gather in the TP>1 FMMS path with symmetric memory.
It is selected automatically when `tp.size > 1`.

The kernel output buffers (`maxs` and `maxs_idx`) are allocated through `get_symm_mem_workspace`, so kernel stores write directly to NVLink-mapped addresses.
After the kernel finishes, a host-side barrier makes every rank's writes visible.
Each rank reads all per-rank tile outputs, applies `_local_reduce`, and selects the global winner with `_stack_and_select_winner`.

This path requires NVLink-connected GPUs, PyTorch 2.6 or newer, and CUDA 12.4 or newer.
See `findings/tp2-collective-overhead.md` for the motivation and measurements.

PyTorch 2.11 fabric handles might extend the raw-pointer approach across one NVL72 rack, but that requires TP72 validation.
Separate NVLink domains require explicit NVSHMEM operations or a hierarchical node-local reduction followed by a small NCCL or InfiniBand exchange.
See `findings/inter-node-scale-out.md` for the scale-out options.

## Process launching

`run_maybe_distributed()` in `src/fused_mm_sampling/tp_info.py` supports two backends.

- `torchrun` is preferred for profiling.
  It is detected through `RANK` and `WORLD_SIZE`, uses `init_method="env://"`, and has no parent polling process.
- `mp.spawn` is the fallback when the torchrun environment is absent.
  It uses a `tcp://` init method, keeps a parent process that polls child sentinels, and does not apply NUMA binding.

Modal Triton benchmarks and distributed correctness tests launch through torchrun.
Triton benchmarks pass `--numa-binding=node`, and the shared Modal image installs `numactl` for PyTorch's supported binding interface.
Do not call private functions from `torch.numa.binding`.
`_apply_numa_binding_to_current_thread` is absent in PyTorch 2.11.
The worker logs `os.sched_getaffinity(0)` after launch so the effective binding remains observable.

NUMA binding is intentionally unconditional.
A temporary toggle did not make slow B200 hosts fast, so the diagnostic option was removed.

## Nsight Systems with TP

`modal-nsys-profile` launches one nsys process per rank through `benchmarking/nsys_wrapper.py`, producing one `.nsys-rep` file per rank.
Wrapping torchrun once from outside captures only the first child CUDA context when using `--capture-range=cudaProfilerApi`, so it cannot profile both devices correctly.
The rank dispatch asymmetry persists under torchrun, which shows that it is not specific to `mp.spawn`.
See `findings/tp2-dispatch-asymmetry.md` for the evidence.

`speed_test.py` has two paths selected by `--nsys_profile=true`.

- `benchmark()` uses CUDA-event timing and no profiler API.
- `nsys_profile()` uses `cudaProfilerStart` and `cudaProfilerStop`, a distributed barrier, and NVTX ranges without timing events.

The flag is a pydantic-settings Boolean, so pass `--nsys_profile=true` rather than a bare `--nsys_profile`.

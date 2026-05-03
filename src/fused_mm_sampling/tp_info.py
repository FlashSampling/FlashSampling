import os
import socket
import subprocess
import warnings
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


@dataclass(frozen=True)
class TPInfo:
    """Tensor parallel context."""

    rank: int
    size: int

    def is_rank0(self) -> bool:
        return self.rank == 0

    def rank0_print(self, *args, **kwargs) -> None:
        if self.is_rank0():
            print(*args, **kwargs)

    @classmethod
    def from_world(cls) -> "TPInfo":
        return cls(rank=dist.get_rank(), size=dist.get_world_size())


TP1 = TPInfo(rank=0, size=1)


def run_maybe_distributed(fn: Callable, n_procs: int, *args) -> None:
    """Run fn(*args) in a single process or spawn n_procs distributed workers.

    fn and args must be picklable (top-level functions, dataclasses, etc.)
    because mp.spawn serializes them across processes. Do not pass lambdas.

    If the process was launched by torchrun (RANK env var is set), skips
    mp.spawn and uses the torchrun-provided environment directly.
    """
    if _is_torchrun():
        _torchrun_worker(fn, args)
    elif n_procs > 1:
        print(f"Running {fn.__name__} with {n_procs} processes")

        mp.spawn(
            _distributed_worker,
            args=(n_procs, _find_free_port(), fn, args),
            nprocs=n_procs,
            join=True,
        )
    else:
        print(f"Running {fn.__name__} single-process")
        fn(*args)


def _is_torchrun() -> bool:
    """Check if the current process was launched by torchrun."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _torchrun_worker(fn: Callable, args: tuple) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if rank == 0:
        _print_gpu_topology()
    torch.cuda.set_device(local_rank)
    backend = "nccl"
    if rank == 0:
        print(f"Using distributed backend: '{backend}' (torchrun)")

    dist.init_process_group(backend=backend, init_method="env://", device_id=local_rank)
    try:
        fn(*args)
    finally:
        dist.destroy_process_group()


def _distributed_worker(rank: int, world_size: int, port: int, fn: Callable, args: tuple) -> None:
    if rank == 0:
        _print_gpu_topology()
    print(
        f"Rank {rank}: available CPUs (before NUMA bind): {sorted(os.sched_getaffinity(0))}",
        flush=True,
    )
    _numa_bind(rank)
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)
    backend = "nccl" if torch.cuda.device_count() >= world_size else "gloo"
    if rank == 0:
        print(f"Using distributed backend: '{backend}'")

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device_id,
    )
    try:
        fn(*args)
    finally:
        dist.destroy_process_group()


def _print_gpu_topology() -> None:
    try:
        topo = subprocess.check_output(["nvidia-smi", "topo", "-m"], text=True).strip()
        print(f"GPU topology:\n{topo}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        warnings.warn(f"Failed to get GPU topology: {e}", stacklevel=2)


def _numa_bind(rank: int) -> None:
    """Pin this process to the CPU cores on the same NUMA node as the GPU.

    A binding failure (e.g. Modal's gVisor returning an empty CPU set) leaves
    the process unbound and produces inconsistent benchmark numbers, so we
    abort instead of warn. Missing torch.numa.binding (older torch) is still
    tolerated since the platform simply does not support binding.
    """
    try:
        from torch.numa.binding import (
            AffinityMode,
            NumaOptions,
            _apply_numa_binding_to_current_thread,
        )
    except ImportError as e:
        warnings.warn(
            f"Rank {rank}: torch.numa.binding unavailable, skipping bind: {e}",
            stacklevel=2,
        )
        return

    _apply_numa_binding_to_current_thread(
        gpu_index=rank,
        numa_options=NumaOptions(affinity_mode=AffinityMode.NODE),
    )
    print(f"Rank {rank}: NUMA-bound to CPUs {sorted(os.sched_getaffinity(0))}", flush=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

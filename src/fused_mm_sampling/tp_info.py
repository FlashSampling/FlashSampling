import socket
from dataclasses import dataclass
from typing import Callable


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
        import torch.distributed as dist

        return cls(rank=dist.get_rank(), size=dist.get_world_size())


TP1 = TPInfo(rank=0, size=1)


def run_maybe_distributed(fn: Callable, n_procs: int, *args) -> None:
    """Run fn(*args) in a single process or spawn n_procs distributed workers.

    fn and args must be picklable (top-level functions, dataclasses, etc.)
    because mp.spawn serializes them across processes. Do not pass lambdas.
    """
    if n_procs > 1:
        import torch.multiprocessing as mp

        mp.spawn(
            _distributed_worker,
            args=(n_procs, _find_free_port(), fn, args),
            nprocs=n_procs,
            join=True,
        )
    else:
        fn(*args)


def _distributed_worker(rank: int, world_size: int, port: int, fn: Callable, args: tuple) -> None:
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank % torch.cuda.device_count())
    backend = "nccl" if torch.cuda.device_count() >= world_size else "gloo"
    if rank == 0:
        print(f"Using distributed backend: '{backend}'")

    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        fn(*args)
    finally:
        dist.destroy_process_group()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

from dataclasses import dataclass


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

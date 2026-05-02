"""Symmetric-memory TP reduction (replaces NCCL all_gather).
Inspired from https://github.com/meta-pytorch/kraken

The kernel's output buffers (maxs, maxs_idx) are allocated in symmetric memory.
For the fan-out path, each rank's kernel writes its own per-tile winners into
every peer rank's buffer. After the kernel completes, a host-side barrier
ensures all remote writes are visible, then each rank reduces its local copy.

Requires: NVLink-connected GPUs, PyTorch >= 2.6, CUDA >= 12.4.
"""

import nvtx
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


@nvtx.annotate()
def allocate_symm_mem_outputs(
    num_samples: int,
    max_grid_size_v: int,
    H: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, object, int, int]:
    """Allocate kernel output buffers (maxs, maxs_idx) in symmetric memory.

    Returns (maxs, maxs_idx, symm_mem_hdl, storage_offset_maxs_idx, world_size).
    maxs and maxs_idx are views into this rank's symmetric memory buffer,
    usable as regular tensors. They have a leading source-rank dimension; each
    rank's kernel fans out writes to that source slot in every peer buffer.
    """
    group = dist.distributed_c10d._get_default_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    n_elements = world_size * num_samples * max_grid_size_v * H
    bytes_maxs = n_elements * 2  # bfloat16
    # TMA requires 128-byte aligned base addresses for tensor descriptors.
    # Align maxs_idx start to 128 bytes, expressed in int64 elements.
    offset_bytes = (bytes_maxs + 127) & ~127
    storage_offset_maxs_idx = offset_bytes // 8  # always exact (128 divisible by 8)
    total_bytes = offset_bytes + n_elements * 8

    symm_mem_hdl = symm_mem.get_symm_mem_workspace(
        group.group_name,
        min_size=total_bytes,
    )

    shape = (world_size, num_samples, max_grid_size_v, H)
    maxs = symm_mem_hdl.get_buffer(rank, shape, torch.bfloat16, storage_offset=0)
    maxs_idx = symm_mem_hdl.get_buffer(
        rank,
        shape,
        torch.int64,
        storage_offset=storage_offset_maxs_idx,
    )

    return maxs, maxs_idx, symm_mem_hdl, storage_offset_maxs_idx, world_size


def kraken_post_kernel_reduce_fanout(
    local_maxs: torch.Tensor,
    local_maxs_idx: torch.Tensor,
    symm_mem_hdl,
    grid_size_v: int,
    H: int,
    num_samples: int,
) -> torch.Tensor:  # [H, num_samples]
    """Barrier + local reduction over fan-out outputs.

    The Triton kernel has already written every source rank's per-tile winners
    into this rank's local symmetric-memory buffer. The barrier only establishes
    visibility for those remote writes; the reduction reads local memory.
    """
    from .core import _local_reduce

    symm_mem_hdl.barrier()

    world_size = symm_mem_hdl.world_size
    maxs = local_maxs[:, :, :grid_size_v, :].movedim(0, 1)
    maxs = maxs.reshape(num_samples, world_size * grid_size_v, H)
    maxs_idx = local_maxs_idx[:, :, :grid_size_v, :].movedim(0, 1)
    maxs_idx = maxs_idx.reshape(num_samples, world_size * grid_size_v, H)
    samples, _ = _local_reduce(maxs, maxs_idx, vocab_start_index=0)
    return samples


def kraken_post_kernel_reduce_remote_read(
    symm_mem_hdl,
    storage_offset_maxs_idx: int,
    grid_size_v: int,
    max_grid_size_v: int,
    H: int,
    num_samples: int,
    vocab_size_per_rank: int,
) -> torch.Tensor:  # [H, num_samples]
    """Barrier + cross-rank reduction reading per-tile outputs from symmetric memory.

    After the main kernel has written per-tile Gumbel-max results to symmetric
    memory, this function:
    1. Barriers to ensure all ranks' kernel writes are visible.
    2. Reads each rank's per-tile outputs from symmetric memory.
    3. Concatenates all ranks' tiles (adjusting indices to global vocab space).
    4. Runs a single _local_reduce to pick the global winner.
    """
    from .core import _local_reduce

    symm_mem_hdl.barrier()

    world_size = symm_mem_hdl.world_size
    full_shape = (num_samples, max_grid_size_v, H)

    all_maxs = []
    all_maxs_idx = []
    for r in range(world_size):
        maxs_r = symm_mem_hdl.get_buffer(r, full_shape, torch.bfloat16, storage_offset=0)
        maxs_idx_r = symm_mem_hdl.get_buffer(
            r,
            full_shape,
            torch.int64,
            storage_offset=storage_offset_maxs_idx,
        )
        # Slice to actual grid size (max_grid_size_v may be larger)
        all_maxs.append(maxs_r[:, :grid_size_v, :])
        all_maxs_idx.append(maxs_idx_r[:, :grid_size_v, :] + r * vocab_size_per_rank)

    maxs = torch.cat(all_maxs, dim=1)
    maxs_idx = torch.cat(all_maxs_idx, dim=1)
    samples, _ = _local_reduce(maxs, maxs_idx, vocab_start_index=0)
    return samples

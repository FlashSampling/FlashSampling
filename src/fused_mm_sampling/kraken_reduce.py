"""Symmetric-memory TP reduction (replaces NCCL all_gather).

The kernel's output buffers (maxs, maxs_idx) are allocated in symmetric memory
so the kernel's existing TMA stores write directly to NVLink-mapped addresses.
After the kernel completes, a host-side barrier ensures all ranks' writes are
visible, then each rank reads all ranks' per-tile outputs and reduces locally.

Requires: NVLink-connected GPUs, PyTorch >= 2.6, CUDA >= 12.4.
"""

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem


def allocate_symm_mem_outputs(
    num_samples: int,
    max_grid_size_v: int,
    H: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, object, int]:
    """Allocate kernel output buffers (maxs, maxs_idx) in symmetric memory.

    Returns (maxs, maxs_idx, symm_mem_hdl, storage_offset_maxs_idx).
    maxs and maxs_idx are views into this rank's symmetric memory buffer,
    usable as regular tensors (including for TMA descriptors).
    """
    group = dist.distributed_c10d._get_default_group()
    rank = dist.get_rank()

    n_elements = num_samples * max_grid_size_v * H
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

    shape = (num_samples, max_grid_size_v, H)
    maxs = symm_mem_hdl.get_buffer(rank, shape, torch.bfloat16, storage_offset=0)
    maxs_idx = symm_mem_hdl.get_buffer(
        rank,
        shape,
        torch.int64,
        storage_offset=storage_offset_maxs_idx,
    )

    return maxs, maxs_idx, symm_mem_hdl, storage_offset_maxs_idx


def kraken_post_kernel_reduce(
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
    3. Runs _local_reduce per rank (reduce across V-tiles, adjust to global indices).
    4. Picks the global winner via _stack_and_select_winner.
    """
    from .core import _local_reduce, _stack_and_select_winner

    symm_mem_hdl.barrier()

    world_size = symm_mem_hdl.world_size
    full_shape = (num_samples, max_grid_size_v, H)

    all_max_values = []
    all_samples = []
    for r in range(world_size):
        maxs_r = symm_mem_hdl.get_buffer(r, full_shape, torch.bfloat16, storage_offset=0)
        maxs_idx_r = symm_mem_hdl.get_buffer(
            r,
            full_shape,
            torch.int64,
            storage_offset=storage_offset_maxs_idx,
        )
        # Slice to actual grid size (max_grid_size_v may be larger)
        maxs_r = maxs_r[:, :grid_size_v, :]
        maxs_idx_r = maxs_idx_r[:, :grid_size_v, :]
        vocab_start_index = r * vocab_size_per_rank
        samples_r, max_values_r = _local_reduce(maxs_r, maxs_idx_r, vocab_start_index)
        all_max_values.append(max_values_r)
        all_samples.append(samples_r)

    return _stack_and_select_winner(all_max_values, all_samples)


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b

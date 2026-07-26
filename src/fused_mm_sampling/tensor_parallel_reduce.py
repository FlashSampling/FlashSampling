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
import triton
import triton.language as tl


@nvtx.annotate()
def allocate_symm_mem_outputs(
    num_samples: int,
    max_grid_size_v: int,
    H: int,
) -> tuple[torch.Tensor, torch.Tensor, object, int]:
    """Allocate kernel output buffers (maxs, maxs_idx) in symmetric memory.

    Returns (maxs, maxs_idx, symm_mem_hdl, storage_offset_maxs_idx).
    maxs and maxs_idx are views into this rank's symmetric memory buffer,
    usable as regular tensors. They have a leading source-rank dimension; each
    rank's kernel fans out writes to that source slot in every peer buffer.
    """
    group = dist.distributed_c10d._get_default_group()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    n_elements = world_size * num_samples * max_grid_size_v * H
    bytes_maxs = n_elements * 4  # float32
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
    maxs = symm_mem_hdl.get_buffer(rank, shape, torch.float32, storage_offset=0)
    maxs_idx = symm_mem_hdl.get_buffer(
        rank,
        shape,
        torch.int64,
        storage_offset=storage_offset_maxs_idx,
    )

    return maxs, maxs_idx, symm_mem_hdl, storage_offset_maxs_idx


@nvtx.annotate()
def tp_post_kernel_reduce(
    local_maxs: torch.Tensor,
    local_maxs_idx: torch.Tensor,
    symm_mem_hdl,
    grid_size_v: int,
) -> torch.Tensor:  # [H, num_samples]
    """Barrier + local reduction over fan-out outputs.

    The Triton kernel has already written every source rank's per-tile winners
    into this rank's local symmetric-memory buffer. The barrier only establishes
    visibility for those remote writes; the reduction reads local memory.
    """
    from .core import _local_reduce

    symm_mem_hdl.barrier()

    world_size, num_samples, _, H = local_maxs.shape
    maxs = local_maxs[:, :, :grid_size_v, :].movedim(0, 1)
    maxs = maxs.reshape(num_samples, world_size * grid_size_v, H)
    maxs_idx = local_maxs_idx[:, :, :grid_size_v, :].movedim(0, 1)
    maxs_idx = maxs_idx.reshape(num_samples, world_size * grid_size_v, H)
    samples, _ = _local_reduce(maxs, maxs_idx, vocab_start_index=0)
    return samples


@nvtx.annotate()
def tp_post_kernel_p2p_reduce(
    local_maxs: torch.Tensor,
    local_maxs_idx: torch.Tensor,
    symm_mem_hdl,
    storage_offset_maxs_idx: int,
    grid_size_v: int,
) -> torch.Tensor:
    """Fan out local candidates with P2P stores, then run the shared reduction.

    Unlike the default path, the FMMS kernel only writes local candidates.
    This separate Triton kernel performs the same peer-memory fan-out after
    computation has finished, isolating the benefit of overlapping P2P stores
    with the matrix multiplication.
    """
    rank = dist.get_rank()
    _, num_samples, max_grid_size_v, H = local_maxs.shape
    n_elements = num_samples * grid_size_v * H
    _fan_out_candidates_kernel[(triton.cdiv(n_elements, 256),)](
        source_maxs=local_maxs[rank],
        source_maxs_idx=local_maxs_idx[rank],
        symm_mem_buffer_ptrs=symm_mem_hdl.buffer_ptrs_dev,
        n_elements=n_elements,
        n_hidden_states=H,
        grid_size_v=grid_size_v,
        max_grid_size_v=max_grid_size_v,
        num_samples=num_samples,
        storage_offset_maxs_idx=storage_offset_maxs_idx,
        tp_rank=rank,
        tp_world_size=dist.get_world_size(),
        BLOCK_SIZE=256,
    )
    return tp_post_kernel_reduce(local_maxs, local_maxs_idx, symm_mem_hdl, grid_size_v)


@triton.jit
def _fan_out_candidates_kernel(
    source_maxs,
    source_maxs_idx,
    symm_mem_buffer_ptrs,
    n_elements,
    n_hidden_states: tl.constexpr,
    grid_size_v: tl.constexpr,
    max_grid_size_v: tl.constexpr,
    num_samples: tl.constexpr,
    storage_offset_maxs_idx: tl.constexpr,
    tp_rank: tl.constexpr,
    tp_world_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    h_idx = offsets % n_hidden_states
    tile_and_sample_idx = offsets // n_hidden_states
    tile_idx = tile_and_sample_idx % grid_size_v
    sample_idx = tile_and_sample_idx // grid_size_v
    source_offset = (
        sample_idx * max_grid_size_v * n_hidden_states + tile_idx * n_hidden_states + h_idx
    )
    destination_offset = (
        tp_rank * num_samples * max_grid_size_v * n_hidden_states + source_offset
    )

    max_values = tl.load(source_maxs + source_offset, mask=mask)
    max_indices = tl.load(source_maxs_idx + source_offset, mask=mask)
    buffer_ptrs = symm_mem_buffer_ptrs.to(tl.pointer_type(tl.uint64))
    for peer_rank in tl.static_range(0, tp_world_size):
        if peer_rank != tp_rank:
            peer_base = tl.load(buffer_ptrs + peer_rank)
            peer_maxs = peer_base.to(tl.pointer_type(tl.float32))
            peer_maxs_idx = peer_base.to(tl.pointer_type(tl.int64))
            tl.store(peer_maxs + destination_offset, max_values, mask=mask)
            tl.store(
                peer_maxs_idx + storage_offset_maxs_idx + destination_offset,
                max_indices,
                mask=mask,
            )

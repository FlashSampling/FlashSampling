"""Fused matmul + top-k reduction Triton kernel.

Computes logits = weights @ hidden_states^T tile-by-tile, then extracts the
top-k logits (and their vocab indices) per tile. The caller merges top-k
candidates across tiles and performs sampling on the CPU/GPU side.

This is intentionally separate from the Gumbel-max FMMS kernel in core.py:
the Gumbel-max kernel fuses matmul + sampling into one pass, while this kernel
fuses matmul + top-k reduction (sampling happens outside).

TODO: Reduce code duplication with the FMMS kernel in core.py.
"""

import nvtx
import torch
import triton
import triton.language as tl

from .tl_argsort import argsort

MIN_BLOCK_SIZE_V = 128


@nvtx.annotate()
def fused_mm_topk_and_sample(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    top_k: int,
    top_p: float | None = None,
):
    """Fused matmul + top-k reduction, followed by top-p filtering and sampling."""
    topk_vals, topk_ids = fused_mm_topk_triton(weights, hidden_states, top_k, temperature)
    return _topk_merge_and_sample(topk_vals, topk_ids, top_k, top_p, num_samples)


def fused_mm_topk_triton(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    top_k: int,
    temperature: torch.Tensor,  # scalar (0-d)
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused matmul + per-tile top-k reduction.

    Returns:
        topk_vals: [num_tiles_v, H, top_k] float32 logits (temperature-scaled)
        topk_ids:  [num_tiles_v, H, top_k] int64 global vocab indices
    """
    V, D = weights.shape  # noqa: N806
    H, D2 = hidden_states.shape  # noqa: N806
    if D2 != D:
        raise ValueError(
            f"hidden_states second dimension ({D2}) must match weights second dimension ({D})"
        )

    max_grid_size = triton.cdiv(V, MIN_BLOCK_SIZE_V)
    out_vals = torch.empty(
        (max_grid_size, H, top_k),
        dtype=torch.float32,
        device=weights.device,
    )
    out_ids = torch.empty(
        (max_grid_size, H, top_k),
        dtype=torch.long,
        device=weights.device,
    )

    num_tiles_v = _launch_kernel(
        weights, hidden_states, out_vals, out_ids, V, D, H, top_k, temperature
    )
    return out_vals[:num_tiles_v], out_ids[:num_tiles_v]


@torch.compile(fullgraph=True)
def _topk_merge_and_sample(
    topk_logits: torch.Tensor,  # [num_tiles, H, top_k]
    topk_idxs: torch.Tensor,  # [num_tiles, H, top_k]
    top_k: int,
    top_p: float | None,
    num_samples: int,
) -> torch.Tensor:
    # Merge across tiles: [num_tiles, H, top_k] -> [H, num_tiles * top_k]
    logits = topk_logits.permute(1, 0, 2).reshape(topk_logits.shape[1], -1)
    idxs = topk_idxs.permute(1, 0, 2).reshape(topk_idxs.shape[1], -1)

    # Global top-k per sequence
    global_logits, merge_idx = logits.topk(top_k, dim=-1)  # [H, top_k]
    global_idxs = idxs.gather(dim=-1, index=merge_idx)  # [H, top_k]

    # Softmax on top-k logits (already temperature-scaled in kernel)
    probs = global_logits.softmax(dim=-1)

    if top_p is not None:
        cumsum = torch.cumsum(probs, dim=-1)
        mask = (cumsum - probs) >= top_p
        mask[:, 0] = False  # always keep the most probable token
        probs.masked_fill_(mask, 0.0)
        probs.div_(probs.sum(dim=-1, keepdim=True))

    # Sample from the top-k candidates
    samples_local = torch.multinomial(
        probs,
        num_samples=num_samples,
        replacement=True,
    )  # [H, num_samples]
    return global_idxs.gather(dim=-1, index=samples_local)  # [H, num_samples]


def bsz_h(H: int) -> int:  # noqa: N803
    if H <= 16:
        return 16
    elif H <= 32:
        return 32
    return 64


@torch.compiler.disable
def _launch_kernel(weights, hidden_states, out_vals, out_ids, V, D, H, top_k, temperature):  # noqa: N803
    grid_size = {"v": None}

    def grid(meta):
        grid_size_v = triton.cdiv(V, meta["BLOCK_SIZE_V"])
        grid_size["v"] = grid_size_v
        return (
            grid_size_v,
            triton.cdiv(H, meta["BLOCK_SIZE_H"]),
        )

    fused_mm_topk_kernel[grid](
        weights_ptr=weights,
        hidden_states_ptr=hidden_states,
        topk_vals_ptr=out_vals,
        topk_ids_ptr=out_ids,
        vocab_size=V,
        hidden_size=D,
        n_hidden_states=H,
        top_k=top_k,
        temperature_ptr=temperature,
    )

    assert grid_size["v"] is not None
    return grid_size["v"]


@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_V": bsz_v,
                "BLOCK_SIZE_D": bsz_d,
                "GROUP_SIZE_V": 4,
            },
            num_warps=num_warps,
            num_stages=num_stages,
            maxnreg=maxnreg,
        )
        for bsz_v in [MIN_BLOCK_SIZE_V, 2 * MIN_BLOCK_SIZE_V]
        for bsz_d in [64, 128]
        for num_warps in [8]
        for maxnreg in [128]
        for num_stages in [4]
    ],
    key=["vocab_size", "hidden_size", "BLOCK_SIZE_H", "top_k"],
    cache_results=True,
)
@triton.heuristics(values={"BLOCK_SIZE_H": lambda args: bsz_h(args["n_hidden_states"])})
@triton.jit
def fused_mm_topk_kernel(
    weights_ptr,  # [V, D]
    hidden_states_ptr,  # [n_hidden_states, D]
    topk_vals_ptr,  # [grid_size_v, H, top_k]
    topk_ids_ptr,  # [grid_size_v, H, top_k]
    vocab_size,  # V
    hidden_size: tl.constexpr,  # D
    n_hidden_states: int,
    top_k: tl.constexpr,
    temperature_ptr,  # scalar (0-d tensor)
    BLOCK_SIZE_V: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_D: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_H: tl.constexpr,  # noqa: N803
    GROUP_SIZE_V: tl.constexpr,  # noqa: N803
):
    temperature = tl.load(temperature_ptr)

    pid_v = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    num_pid_v = tl.cdiv(vocab_size, BLOCK_SIZE_V)
    num_pid_h = tl.cdiv(n_hidden_states, BLOCK_SIZE_H)
    pid_v, pid_h = tl.swizzle2d(pid_v, pid_h, num_pid_v, num_pid_h, GROUP_SIZE_V)
    v_start = pid_v * BLOCK_SIZE_V
    h_start = pid_h * BLOCK_SIZE_H

    offsets_v = v_start + tl.arange(0, BLOCK_SIZE_V)
    mask_v = offsets_v < vocab_size
    offsets_h = h_start + tl.arange(0, BLOCK_SIZE_H)
    mask_h = offsets_h < n_hidden_states
    logits_blk = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_H), dtype=tl.float32)

    # Tiled matmul: logits_blk = weights[v_tile, :] @ hidden_states[h_tile, :].T
    for d_start in range(0, hidden_size, BLOCK_SIZE_D):
        offsets_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offsets_d < hidden_size

        w_blk = tl.load(
            weights_ptr + offsets_d[None, :] + hidden_size * offsets_v[:, None],
            mask=mask_v[:, None] & mask_d[None, :],
        )
        hidden_states_blk = tl.load(
            hidden_states_ptr + offsets_d[None, :] + hidden_size * offsets_h[:, None],
            mask=mask_h[:, None] & mask_d[None, :],
        )
        logits_blk = tl.dot(w_blk, hidden_states_blk.T, acc=logits_blk)

    # Mask out-of-vocab rows and apply temperature
    logits_blk = tl.where(mask_v[:, None], logits_blk, -float("inf"))
    logits_blk = logits_blk / temperature  # [BLOCK_SIZE_V, BLOCK_SIZE_H]

    # Argsort along V (last dim after transpose), keep top-k
    logits_t = logits_blk.T  # [BLOCK_SIZE_H, BLOCK_SIZE_V]
    ids_t = tl.broadcast_to(tl.arange(0, BLOCK_SIZE_V)[None, :], [BLOCK_SIZE_H, BLOCK_SIZE_V]).to(
        tl.int64
    )
    sorted_vals, sorted_ids = argsort(logits_t, ids_t, descending=True)

    # Store top-k values and global vocab indices
    h_local = tl.arange(0, BLOCK_SIZE_H)
    v_local = tl.arange(0, BLOCK_SIZE_V)
    out_base = pid_v * n_hidden_states * top_k
    topk_offsets = (h_start + h_local)[:, None] * top_k + v_local[None, :]
    topk_mask = mask_h[:, None] & (v_local[None, :] < top_k)

    tl.store(topk_vals_ptr + out_base + topk_offsets, sorted_vals, mask=topk_mask)
    tl.store(
        topk_ids_ptr + out_base + topk_offsets,
        sorted_ids + v_start,
        mask=topk_mask,
    )

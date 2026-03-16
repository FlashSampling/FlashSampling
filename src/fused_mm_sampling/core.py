# import os

# os.environ["TRITON_INTERPRET"] = "1"
import functools
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, NamedTuple, Protocol

import flashinfer
import nvtx
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import triton.profiler.language as pl

from .tl_matmul import matmul
from .tp_info import TP1, TPInfo


def sample(
    weights: torch.Tensor,  # [V, D] (may be a TP shard over dim V)
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    return_probs: bool = False,
    seed: int = None,
    tl_matmul: bool = False,
    top_k: int | None = None,
    top_p: float | None = None,
    use_qitra: bool = False,
    tp: "TPInfo" = TP1,
):
    if seed is not None:
        torch.manual_seed(seed)
    if tl_matmul:
        logits = matmul(hidden_states, weights)  # [n_hidden_states, V]
    else:
        logits = hidden_states @ weights.T  # [n_hidden_states, V]
    if tp.size > 1:
        logits = _allgather_logits(logits)  # shape [H, V_local] -> [H, V]
    logits /= temperature
    if use_qitra:
        probs = apply_top_k_top_p_qitra(logits, top_k, top_p)
    else:
        probs = apply_top_k_top_p(logits, top_k, top_p)
    # Upcast to float32: torch.multinomial produces imprecise distributions with
    # bfloat16 inputs (chi-squared p≈0). See findings/upcasting-before-softmax.md.
    samples = torch.multinomial(probs.float(), num_samples, replacement=True)
    if return_probs:
        return samples, probs
    return samples


def _fast_multinomial(probs: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Sample from a categorical distribution using the exponential race method.

    Avoids torch.multinomial's 10-kernel validation overhead (~2/3 of its runtime).
    For each row, draws exponential noise, computes probs / noise, and takes argmax.
    See https://github.com/pytorch/pytorch/issues/177127
    """
    H, V = probs.shape  # noqa: N806
    q = torch.empty(num_samples, H, V, device=probs.device, dtype=probs.dtype)
    q.exponential_()
    return probs.unsqueeze(0).div(q).argmax(dim=-1).T  # [H, num_samples]


@torch.compiler.disable
def _allgather_logits(
    logits: torch.Tensor,  # [H, V_local]
) -> torch.Tensor:
    """All-gather local logits along the vocab dimension to reconstruct [H, V_global]."""
    tp_size = dist.get_world_size()
    all_logits = [torch.empty_like(logits) for _ in range(tp_size)]
    dist.all_gather(all_logits, logits)
    return torch.cat(all_logits, dim=1)  # [H, V_global]


def apply_top_k_top_p(
    logits: torch.Tensor,  # [batch, V], float32
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Apply top-k and top-p filtering and return probabilities. Single softmax."""
    if top_k is None and top_p is None:
        return logits.softmax(dim=-1)

    k = top_k if top_k is not None else logits.shape[-1]
    topk_vals, topk_idx = logits.topk(k, dim=-1)
    probs = topk_vals.softmax(dim=-1)

    if top_p is not None:
        cumsum = torch.cumsum(probs, dim=-1)
        top_p_mask = (cumsum - probs) >= top_p
        top_p_mask[:, 0] = False  # always keep the most probable token
        probs.masked_fill_(top_p_mask, 0.0)
        probs.div_(probs.sum(dim=-1, keepdim=True))

    out = torch.zeros_like(logits)
    return out.scatter_(dim=-1, index=topk_idx, src=probs)


@lru_cache
def print_once(msg: str):
    print(msg)


def apply_top_k_top_p_qitra(
    logits: torch.Tensor,  # [batch, V], float32
    top_k: int | None,
    top_p: float | None,
) -> torch.Tensor:
    """Apply top-k and top-p filtering using vLLM's Qitra Triton kernel and return probabilities."""
    from .qitra import apply_top_k_top_p_triton

    print_once(msg="Using Qitra for top-k/top-p filtering")

    batch_size = logits.shape[0]
    k = (
        torch.full((batch_size,), top_k, dtype=torch.int32, device=logits.device)
        if top_k is not None
        else None
    )
    p = (
        torch.full((batch_size,), top_p, dtype=torch.float32, device=logits.device)
        if top_p is not None
        else None
    )
    logits = apply_top_k_top_p_triton(logits, k, p)
    return logits.softmax(dim=-1)


sample_compiled_fullgraph = torch.compile(sample, fullgraph=True)
sample_compiled_with_breaks = torch.compile(sample)


@nvtx.annotate()
@functools.wraps(sample)
def sample_compiled(*args, tp: TPInfo = TP1, **kwargs):
    if tp.size > 1:
        if tp.is_rank0():
            print_once("Using sample_compiled_with_breaks")
        return sample_compiled_with_breaks(*args, tp=tp, **kwargs)
    if tp.is_rank0():
        print_once("Using sample_compiled_fullgraph")
    return sample_compiled_fullgraph(*args, tp=tp, **kwargs)


@nvtx.annotate()
@torch.compile(fullgraph=True)
def sequential_sample_pt(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
):
    device = weights.device
    V, D = weights.shape  # noqa: N806
    H, D2 = hidden_states.shape  # noqa: N806
    if D2 != D:
        raise ValueError(
            f"hidden_states second dimension ({D2}) must match weights first dimension ({D})"
        )
    block_size = 8192
    # compute logits blocks
    gumbel_max = torch.full((num_samples, H), float("-inf"), device=device)
    gumbel_max_idx = torch.empty(size=(num_samples, H), dtype=torch.long, device=device)
    n_blocks = cdiv(V, block_size)
    for blk_idx in range(n_blocks):
        idx_from = blk_idx * block_size
        idx_to = (blk_idx + 1) * block_size
        w_blk = weights[idx_from:idx_to, :]  # [block_size, D]
        logits_blk = hidden_states @ w_blk.T / temperature  # [n_hidden_states, block_size]
        unif_noise = torch.rand((num_samples, *logits_blk.shape), device=device)
        gumbel_noise = -(-unif_noise.log()).log()
        new_max, new_max_idx_local = torch.max(logits_blk + gumbel_noise, dim=2)
        new_max_idx_global = idx_from + new_max_idx_local

        replace_mask = new_max > gumbel_max
        gumbel_max = torch.where(replace_mask, new_max, gumbel_max)
        gumbel_max_idx = torch.where(replace_mask, new_max_idx_global, gumbel_max_idx)
    return gumbel_max_idx.T


def cdiv(n: int, div: int) -> int:
    return (n + div - 1) // div


MIN_BLOCK_SIZE_V = 128


# @torch.compile(fullgraph=True)
@nvtx.annotate()
def fused_mm_sample_triton(
    weights: torch.Tensor,  # [V_local, D] (may be a TP shard)
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    seed: int,
    GUMBEL: bool = True,  # noqa: N803
    tp: "TPInfo" = TP1,
):
    V, D = weights.shape  # noqa: N806
    H, D2 = hidden_states.shape  # noqa: N806
    if D2 != D:
        raise ValueError(
            f"hidden_states second dimension ({D2}) must match weights second dimension ({D})"
        )

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count  # noqa: N806

    max_grid_size_v = triton.cdiv(V, MIN_BLOCK_SIZE_V)
    maxs = torch.empty(
        (num_samples, max_grid_size_v, H),
        dtype=torch.bfloat16,
        device=weights.device,
    )
    maxs_idx = torch.empty_like(maxs, dtype=torch.long)

    grid_size = {"v": None}

    def grid(meta):
        grid_size_v = triton.cdiv(V, meta["BLOCK_SIZE_V"])
        grid_size_h = triton.cdiv(H, meta["BLOCK_SIZE_H"])
        grid_size["v"] = grid_size_v
        num_tiles = grid_size_v * grid_size_h
        # Persistent kernel: launch min(NUM_SMS, num_tiles) programs
        return (min(NUM_SMS, num_tiles),)

    fused_mm_sample_triton_kernel[grid](
        weights_ptr=weights,
        hidden_states_ptr=hidden_states,
        max_out_ptr=maxs,
        max_out_idx_ptr=maxs_idx,
        vocab_size=V,
        hidden_size=D,
        n_hidden_states=H,
        num_samples=num_samples,
        temperature_ptr=temperature,
        seed=seed,
        max_grid_size_v=max_grid_size_v,
        WARP_SPECIALIZE=supports_warp_specialization(),
        NUM_SMS=NUM_SMS,
        GUMBEL=GUMBEL,
        USE_PROTON_SCOPES=os.environ.get("USE_PROTON_SCOPES", "0") == "1",
    )

    assert grid_size["v"] is not None

    # Stage 2: local reduction across V-tiles on this rank.
    vocab_start_index = tp.rank * V
    samples, max_values = _local_reduce(
        maxs[:, : grid_size["v"], :], maxs_idx[:, : grid_size["v"], :], vocab_start_index
    )

    # Stage 3: cross-rank reduction via Gumbel-max decomposition
    if tp.size > 1:
        samples = _tensor_parallel_reduce(samples, max_values)

    return samples  # [n_hidden_states, num_samples]


@lru_cache
def supports_warp_specialization():
    is_cuda = triton.runtime.driver.active.get_current_target().backend == "cuda"
    supports_ws = is_cuda and torch.cuda.get_device_capability()[0] >= 9
    print("Supports warp specialization:", supports_ws)
    return supports_ws


@torch.compile(fullgraph=True)
def _local_reduce(
    maxs: torch.Tensor,  # [num_samples, n_tiles, H]
    maxs_idx: torch.Tensor,  # [num_samples, n_tiles, H]
    vocab_start_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce across V-tiles (dim=1) on this rank and adjust to global vocab indices."""
    idxs = maxs.max(dim=1).indices  # [num_samples, H]
    samples = maxs_idx.gather(1, idxs.unsqueeze(1)).squeeze(1)  # [num_samples, H]
    max_values = maxs.gather(1, idxs.unsqueeze(1)).squeeze(1)  # [num_samples, H]
    samples += vocab_start_index
    return samples.T.contiguous(), max_values.T.contiguous()  # [H, num_samples]


def _tensor_parallel_reduce(
    samples: torch.Tensor,  # [H, num_samples]
    max_values: torch.Tensor,  # [H, num_samples]
) -> torch.Tensor:  # [H, num_samples]
    """All-gather local Gumbel maxes across TP ranks and pick the global winner."""
    tp_size = dist.get_world_size()
    all_max_values = [torch.empty_like(max_values) for _ in range(tp_size)]
    all_samples = [torch.empty_like(samples) for _ in range(tp_size)]
    dist.all_gather(all_max_values, max_values)
    dist.all_gather(all_samples, samples)
    samples = _stack_and_select_winner(all_max_values, all_samples)
    return samples  # [H, num_samples]


@torch.compile(fullgraph=True)
def _stack_and_select_winner(all_max_values, all_samples) -> torch.Tensor:  # [H, num_samples]
    stacked_values = torch.stack(all_max_values)  # [world_size, H, num_samples]
    stacked_samples = torch.stack(all_samples)  # [world_size, H, num_samples]
    winner_rank = stacked_values.argmax(dim=0)  # [H, num_samples]
    samples = stacked_samples.gather(0, winner_rank.unsqueeze(0)).squeeze(0)
    return samples  # [H, num_samples]


def clip(low, high, x):
    return min(max(x, low), high)


def is_config_valid(bsz_v, bsz_d, bsz_h):
    # Derive limit from hardware constraints:
    # - H100/A100 shared memory: 232448 (from Triton logs)
    max_bytes = 232448

    # Memory usage in kernel:
    # - logits_blk: bsz_v * bsz_h * 4 bytes (float32, persists)
    # - w_blk: bsz_v * bsz_d * 2 bytes (bfloat16, during matmul)
    # - hidden_states_blk: bsz_h * bsz_d * 2 bytes (bfloat16, during matmul)
    # - noise: bsz_v * bsz_h * 4 bytes (float32)
    # - gumbel_noise: bsz_v * bsz_h * 4 bytes (float32)

    # Peak memory during sampling phase:
    # logits_blk + gumbel_noise = bsz_v * bsz_h * (4 + 4) bytes
    # = bsz_v * bsz_h * 8 bytes per element
    bytes_per_elem = 8
    max_elements = max_bytes / bytes_per_elem  # ~16,384 elements

    if bsz_v * bsz_h > max_elements:
        return False

    # Also check matmul phase memory (w_blk + hidden_states_blk)
    matmul_bytes = bsz_v * bsz_d * 2 + bsz_h * bsz_d * 2
    if matmul_bytes > max_bytes:
        return False

    return True


@triton.jit
def _compute_tile_pid(tile_id, num_pid_in_group, num_pid_v, GROUP_SIZE_V):  # noqa: N803
    """Compute pid_v, pid_h from tile_id using grouped ordering for L2 cache efficiency."""
    group_id = tile_id // num_pid_in_group
    first_pid_v = group_id * GROUP_SIZE_V
    group_size_v = tl.minimum(num_pid_v - first_pid_v, GROUP_SIZE_V)
    pid_v = first_pid_v + (tile_id % group_size_v)
    pid_h = (tile_id % num_pid_in_group) // group_size_v
    return pid_v, pid_h


def metadata_fn(
    grid: tuple,
    metadata: NamedTuple,
    args: dict,
):
    """Copied from https://github.com/triton-lang/triton/blob/main/third_party/proton/tutorials/matmul.py"""
    grid_x, grid_y, grid_z = unpack_grid(grid)
    num_warps = metadata.num_warps
    num_stages = metadata.num_stages
    cluster_x, cluster_y, cluster_z = unpack_grid((metadata.num_ctas,))
    shared_memory = metadata.shared
    return {
        "name": f"fused_mm_sample_triton_<grid:{grid_x}x{grid_y}x{grid_z}>_<cluster:{cluster_x}x{cluster_y}x{cluster_z}>_<warps:{num_warps}>_<shared:{shared_memory}>_<stages:{num_stages}>",
    }


def unpack_grid(grid):
    if len(grid) == 1:
        return grid[0], 1, 1
    if len(grid) == 2:
        return grid[0], grid[1], 1
    if len(grid) == 3:
        return grid[0], grid[1], grid[2]


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
        for num_warps in [8]  # Default 4
        for maxnreg in [128]  # Previously 255, not sure either is better
        for num_stages in [4]  # 4 outpeforms 2, and 3
    ],
    key=["vocab_size", "hidden_size", "BLOCK_SIZE_H", "num_samples", "GUMBEL"],
    cache_results=True,
)
@triton.heuristics(values={"BLOCK_SIZE_H": lambda args: bsz_h(args["n_hidden_states"])})
@triton.jit(launch_metadata=metadata_fn)
def fused_mm_sample_triton_kernel(
    weights_ptr,  # [V, D]
    hidden_states_ptr,  # [n_hidden_states, D]
    max_out_ptr,  # [num_samples, max_grid_size_v, n_hidden_states]
    max_out_idx_ptr,  # [num_samples, max_grid_size_v, n_hidden_states]
    vocab_size,  # V
    hidden_size: tl.constexpr,  # D
    n_hidden_states: int,
    num_samples: tl.constexpr,
    temperature_ptr,  # scalar (0-d tensor)
    seed: int,
    BLOCK_SIZE_V: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_D: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_H: tl.constexpr,  # noqa: N803
    GROUP_SIZE_V: tl.constexpr,  # noqa: N803
    max_grid_size_v: tl.constexpr,
    WARP_SPECIALIZE: tl.constexpr,  # noqa: N803
    NUM_SMS: tl.constexpr,  # noqa: N803
    GUMBEL: tl.constexpr,  # noqa: N803
    USE_PROTON_SCOPES: tl.constexpr,  # noqa: N803
):
    """Persistent kernel for fused matmul + Gumbel-max sampling.

    Each SM processes multiple tiles in a loop, staying persistent on the SM
    rather than exiting after processing a single tile.
    """
    if USE_PROTON_SCOPES:
        pl.enter_scope("setup")
    temperature = tl.load(temperature_ptr)
    start_pid = tl.program_id(axis=0)
    num_pid_v = tl.cdiv(vocab_size, BLOCK_SIZE_V)
    num_pid_h = tl.cdiv(n_hidden_states, BLOCK_SIZE_H)
    num_tiles = num_pid_v * num_pid_h
    num_pid_in_group = GROUP_SIZE_V * num_pid_h

    # Create tensor descriptors outside the tile loop (they describe full tensors)
    w_desc = tl.make_tensor_descriptor(
        weights_ptr,
        shape=[vocab_size, hidden_size],
        strides=[hidden_size, 1],
        block_shape=[BLOCK_SIZE_V, BLOCK_SIZE_D],
    )
    hidden_states_desc = tl.make_tensor_descriptor(
        hidden_states_ptr,
        shape=[n_hidden_states, hidden_size],
        strides=[hidden_size, 1],
        block_shape=[BLOCK_SIZE_H, BLOCK_SIZE_D],
    )
    max_desc = tl.make_tensor_descriptor(
        max_out_ptr,
        shape=[num_samples, max_grid_size_v, n_hidden_states],
        strides=[max_grid_size_v * n_hidden_states, n_hidden_states, 1],
        block_shape=[1, 1, BLOCK_SIZE_H],
    )
    max_idx_desc = tl.make_tensor_descriptor(
        max_out_idx_ptr,
        shape=[num_samples, max_grid_size_v, n_hidden_states],
        strides=[max_grid_size_v * n_hidden_states, n_hidden_states, 1],
        block_shape=[1, 1, BLOCK_SIZE_H],
    )
    if USE_PROTON_SCOPES:
        pl.exit_scope("setup")
    # tile_id_c is used in the epilogue to break the dependency between
    # the prologue and the epilogue (workaround for Blackwell pipelining bug)
    tile_id_c = start_pid - NUM_SMS

    # Persistent loop: each SM processes multiple tiles
    for tile_id in tl.range(
        start_pid, num_tiles, NUM_SMS, flatten=True, warp_specialize=WARP_SPECIALIZE
    ):
        # Compute pid_v, pid_h from tile_id using grouped ordering for L2 cache
        pid_v, pid_h = _compute_tile_pid(tile_id, num_pid_in_group, num_pid_v, GROUP_SIZE_V)

        v_start = pid_v * BLOCK_SIZE_V
        h_start = pid_h * BLOCK_SIZE_H

        offsets_v = v_start + tl.arange(0, BLOCK_SIZE_V)
        mask_v = offsets_v < vocab_size

        logits_blk = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_H), dtype=tl.float32)

        # Compute a block of logits logits_blk
        if USE_PROTON_SCOPES:
            pl.enter_scope("matmul-tile")
        for d_start in range(0, hidden_size, BLOCK_SIZE_D):
            # load weights tile [BLOCK_SIZE_V, BLOCK_SIZE_D]
            w_blk = w_desc.load([v_start, d_start])
            # load hidden_states tile [BLOCK_SIZE_H, BLOCK_SIZE_D]
            hidden_states_blk = hidden_states_desc.load([h_start, d_start])
            logits_blk = tl.dot(w_blk, hidden_states_blk.T, acc=logits_blk)
        if USE_PROTON_SCOPES:
            pl.exit_scope("matmul-tile")

        # Later we will take max over logits + noise, but rows outside the mask
        # should not be considered. Setting them to -inf achieves this.
        logits_blk = tl.where(mask_v[:, None], logits_blk, -float("inf"))
        logits_blk = logits_blk / temperature  # [Vblk, n_hidden_states]

        # Epilogue: use tile_id_c to break dependency with prologue
        tile_id_c += NUM_SMS
        pid_v_c, pid_h_c = _compute_tile_pid(tile_id_c, num_pid_in_group, num_pid_v, GROUP_SIZE_V)
        v_start_c = pid_v_c * BLOCK_SIZE_V
        h_start_c = pid_h_c * BLOCK_SIZE_H

        for sample_idx in range(num_samples):
            if USE_PROTON_SCOPES:
                pl.enter_scope("sample")
            # Note: Creating appropriately sized tensors is tricky because
            # tl.arange() only accepts tl.constexpr that are powers of 2.
            noise_size: tl.constexpr = BLOCK_SIZE_V * BLOCK_SIZE_H
            noise_offsets = tl.arange(0, noise_size).reshape((BLOCK_SIZE_V, BLOCK_SIZE_H))

            if GUMBEL:
                # Note: Each tile (v, h) and sample needs a different seed,
                # otherwise they all create the same noise, leading to sampling artifacts.
                # Compute gumbel noise directly to reduce register pressure
                gumbel_noise = -tl.log(
                    -tl.log(
                        tl.rand(
                            seed + pid_v_c * 100 + pid_h_c * 1_000 + sample_idx * 10_000,
                            noise_offsets,
                        )
                    )
                )
                gumbel_max, gumbel_max_idx_local = tl.max(
                    logits_blk + gumbel_noise, axis=0, return_indices=True
                )
            else:
                gumbel_max, gumbel_max_idx_local = tl.max(logits_blk, axis=0, return_indices=True)
            gumbel_max_idx_global = gumbel_max_idx_local + v_start_c
            if USE_PROTON_SCOPES:
                pl.exit_scope("sample")

            if USE_PROTON_SCOPES:
                pl.enter_scope("store")
            max_desc.store(
                [sample_idx, pid_v_c, h_start_c],
                gumbel_max[None, None, :],
            )
            max_idx_desc.store(
                [sample_idx, pid_v_c, h_start_c],
                gumbel_max_idx_global[None, None, :],
            )
            if USE_PROTON_SCOPES:
                pl.exit_scope("store")


def set_torch_allocator_for_tma_descriptors():
    """From https://triton-lang.org/main/python-api/generated/triton.language.make_tensor_descriptor.html"""
    # TMA descriptors require a global memory allocation
    triton.set_allocator(alloc_on_cuda)


def alloc_on_cuda(size: int, alignment: int, stream: int | None):
    return torch.empty(size, device="cuda", dtype=torch.int8)


class Sampler(Protocol):
    def prepare(self) -> "Sampler":
        raise NotImplementedError()

    def sample(self, **kwargs) -> torch.Tensor:
        raise NotImplementedError()


@dataclass
class SimpleSampler(Sampler):
    fn: Callable[..., torch.Tensor]

    def prepare(self) -> "SimpleSampler":
        return self

    def sample(self, **kwargs) -> torch.Tensor:
        return self.fn(**kwargs)


@dataclass
class JLSampler(Sampler):
    weights: torch.Tensor  # [V, D]
    k: int
    prepared: bool = False

    @classmethod
    def from_weights(
        cls,
        weights: torch.Tensor,  # [V, D]
        epsilon: float = 0.2,
    ) -> "JLSampler":
        k = optimal_k(n=weights.shape[0], epsilon=epsilon)
        print(f"JLSampler optimal k={k}")
        return cls(weights, k=k)

    def prepare(self) -> "JLSampler":
        D = self.weights.shape[1]  # noqa: N806
        self.rand_mat = torch.randn(
            (D, self.k),
            dtype=self.weights.dtype,
            device=self.weights.device,
        ) / math.sqrt(self.k)
        self.w_p = self.weights @ self.rand_mat  # [V, k]
        self.w_p = self.w_p.contiguous()
        self.prepared = True
        self.weights = None  # not needed anymore
        return self

    @torch.compile(fullgraph=True)
    def sample(
        self,
        hidden_states: torch.Tensor,  # [n_hidden_states, D]
        temperature: torch.Tensor,  # scalar (0-d)
        num_samples: int,
        seed: int | None = None,  # ignored
        weights: torch.Tensor = None,  # ignored
    ):
        """
        Sampling using low-dimensional random projections (Johnson-Lindenstrauss lemma).
        """
        if not self.prepared:
            raise ValueError("Sampler not prepared. Call .prepare() first.")
        logits_p = self.compute_logits(hidden_states)
        probs = (logits_p / temperature).softmax(dim=1)
        samples = _fast_multinomial(probs, num_samples)
        return samples

    def compute_logits(
        self,
        hidden_states: torch.Tensor,  # [n_hidden_states, D]
    ) -> torch.Tensor:
        h_p = hidden_states @ self.rand_mat  # [n_hidden_states, k]
        return h_p @ self.w_p.T  # [n_hidden_states, V]

    def rrt(self) -> torch.Tensor:
        """Return R @ Rᵀ, which should be close to the identity matrix."""
        m = self.rand_mat
        return m @ m.T


def optimal_k(n: int, epsilon: float) -> int:
    """Source: https://cs.stanford.edu/people/mmahoney/cs369m/Lectures/lecture1.pdf"""
    k_float = 24 * math.log(n, math.e) / (3 * epsilon**2 - 2 * epsilon**3)
    return int(math.ceil(k_float))


def get_sampler(provider: str, weights: torch.Tensor) -> Sampler:
    match provider:
        case "fused-triton":
            return SimpleSampler(lambda **kwargs: fused_mm_sample_triton(**{"seed": 0, **kwargs}))
        case "fused-triton-no-gumbel":
            return SimpleSampler(
                lambda **kwargs: fused_mm_sample_triton(**{"seed": 0, "GUMBEL": False, **kwargs})
            )
        case "naive-pt":
            return SimpleSampler(sample)
        case "naive-compiled":
            return SimpleSampler(sample_compiled)
        case "pt-qitra":
            return SimpleSampler(lambda **kwargs: sample(**kwargs, use_qitra=True))
        case "sequential-compiled":
            return SimpleSampler(sequential_sample_pt)
        case "naive-tl-matmul":
            return SimpleSampler(lambda **kwargs: sample_compiled(**kwargs, tl_matmul=True))
        case "jl-compiled":
            return JLSampler.from_weights(weights)
        case "fused-topk":
            from .tl_fused_mm_topk import fused_mm_topk_and_sample

            return SimpleSampler(fused_mm_topk_and_sample)
        case "flashinfer:top_k_top_p_sampling_from_logits":
            return SimpleSampler(
                lambda **kwargs: flashinfer_top_k_top_p_sampling_from_logits(
                    **_default_top_k_top_p(kwargs),
                )
            )
        case "fused-cuda":
            from .cuda_impl import fused_mm_sample_cuda

            return SimpleSampler(lambda **kwargs: fused_mm_sample_cuda(**kwargs, seed=0))
        case "helion":
            from .helion_impl import fused_mm_sample_helion

            return SimpleSampler(fused_mm_sample_helion)
        case "flashinfer:sampling_from_logits":
            return SimpleSampler(flashinfer_sampling_from_logits)
        case _:
            raise NotImplementedError()


def _default_top_k_top_p(kwargs: dict) -> dict:
    """Set flashinfer defaults for top_k/top_p when not provided by the caller."""
    if "top_k" not in kwargs:
        kwargs["top_k"] = -1
    if "top_p" not in kwargs:
        kwargs["top_p"] = 1.0
    return kwargs


def flashinfer_top_k_top_p_sampling_from_logits(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    top_p: float,
    top_k: int,
    tp: "TPInfo" = TP1,
    **_kwargs,
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    logits, indices = flashinfer_create_logits_and_indices(
        weights, hidden_states, num_samples, temperature
    )
    if tp.size > 1:
        logits = _allgather_logits(logits)
    result = flashinfer.sampling.top_k_top_p_sampling_from_logits(
        logits=logits,
        top_k=top_k,
        top_p=top_p,
        indices=indices,
    )
    return result.reshape(batch_size, num_samples)


@nvtx.annotate()
@torch.compile(fullgraph=True)
def flashinfer_create_logits_and_indices(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
) -> tuple[torch.Tensor, torch.Tensor]:
    device = weights.device
    batch_size = hidden_states.shape[0]
    assert weights.shape[1] == hidden_states.shape[1], "weights must transposed"
    logits = hidden_states @ weights.T  # [batch_size, vocab]
    logits /= temperature
    indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.int32), num_samples
    )
    return logits, indices


@nvtx.annotate()
def flashinfer_sampling_from_logits(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    tp: "TPInfo" = TP1,
    **_kwargs,
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    logits, indices = flashinfer_create_logits_and_indices(
        weights, hidden_states, num_samples, temperature
    )
    if tp.size > 1:
        logits = _allgather_logits(logits)
    result = flashinfer.sampling.sampling_from_logits(logits=logits, indices=indices)
    return result.reshape(batch_size, num_samples)


def bsz_h(H: int) -> int:  # noqa: N803
    if H <= 16:
        return 16
    elif H <= 32:
        return 32
    return 64

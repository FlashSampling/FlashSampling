# import os

# os.environ["TRITON_INTERPRET"] = "1"
import math
import os
from dataclasses import dataclass
from typing import Callable, NamedTuple, Protocol

import flashinfer
import nvtx
import torch
import triton
import triton.language as tl
import triton.profiler.language as pl

from .tl_matmul import matmul


@nvtx.annotate()
def sample(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    return_probs: bool = False,
    seed: int = None,
    tl_matmul: bool = False,
    top_k: int | None = None,
    top_p: float | None = None,
):
    if seed is not None:
        torch.manual_seed(seed)
    if tl_matmul:
        logits = matmul(hidden_states, weights)  # [n_hidden_states, V]
    else:
        logits = hidden_states @ weights.T  # [n_hidden_states, V]
    # Upcast to float32: torch.multinomial produces incorrect distributions with bfloat16.
    # If we remove the cast, the correctness test fails (a chi-squared test).
    logits = logits.float() / temperature
    probs = apply_top_k_top_p(logits, top_k, top_p)
    samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)
    if return_probs:
        return samples, probs
    return samples


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


sample_compiled = torch.compile(sample)


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
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    seed: int,
    GUMBEL: bool = True,  # noqa: N803
):
    V, D = weights.shape  # noqa: N806
    H, D2 = hidden_states.shape  # noqa: N806
    if D2 != D:
        raise ValueError(
            f"hidden_states second dimension ({D2}) must match weights second dimension ({D})"
        )

    max_grid_size = triton.cdiv(V, MIN_BLOCK_SIZE_V)
    maxs = torch.empty(
        (max_grid_size, H, num_samples),
        dtype=torch.bfloat16,
        device=weights.device,
    )
    maxs_idx = torch.empty_like(maxs, dtype=torch.long)

    grid_size = {"v": None}

    def grid(meta):
        grid_size_v = triton.cdiv(V, meta["BLOCK_SIZE_V"])
        grid_size["v"] = grid_size_v
        return (
            grid_size_v,
            triton.cdiv(H, meta["BLOCK_SIZE_H"]),
        )

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
        GUMBEL=GUMBEL,
        USE_PROTON_SCOPES=os.environ.get("USE_PROTON_SCOPES", "0") == "1",
    )

    # 2nd stage: reduction
    assert grid_size["v"] is not None
    idxs = maxs[: grid_size["v"], :, :].max(axis=0).indices
    samples = maxs_idx.gather(dim=0, index=idxs[None, :])
    return samples.squeeze(0)  # [n_hidden_states, num_samples]


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
    # - noise: bsz_v * bsz_h * 4 bytes (float32, BLOCK_SIZE_NSAMPLES=1)
    # - gumbel_noise: bsz_v * bsz_h * 4 bytes (float32, BLOCK_SIZE_NSAMPLES=1)

    # Peak memory during sampling phase (BLOCK_SIZE_NSAMPLES=1):
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
                "BLOCK_SIZE_NSAMPLES": 1,
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
    max_out_ptr,  # [grid_size, n_hidden_states, num_samples]
    max_out_idx_ptr,  # [grid_size, n_hidden_states, num_samples]
    vocab_size,  # V
    hidden_size: tl.constexpr,  # D
    n_hidden_states: int,
    num_samples: tl.constexpr,
    temperature_ptr,  # scalar (0-d tensor)
    seed: int,
    BLOCK_SIZE_V: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_D: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_H: tl.constexpr,  # noqa: N803
    BLOCK_SIZE_NSAMPLES: tl.constexpr,  # noqa: N803
    GROUP_SIZE_V: tl.constexpr,  # noqa: N803
    GUMBEL: tl.constexpr,  # noqa: N803
    USE_PROTON_SCOPES: tl.constexpr,  # noqa: N803
):
    if USE_PROTON_SCOPES:
        pl.enter_scope("setup")
    temperature = tl.load(temperature_ptr)

    # Compute a different program ordering to exploit L2 cache, as suggested in
    # https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
    pid_v = tl.program_id(axis=0)
    pid_h = tl.program_id(axis=1)
    num_pid_v = tl.cdiv(vocab_size, BLOCK_SIZE_V)
    num_pid_h = tl.cdiv(n_hidden_states, BLOCK_SIZE_H)
    pid_v, pid_h = tl.swizzle2d(pid_v, pid_h, num_pid_v, num_pid_h, GROUP_SIZE_V)
    v_start = pid_v * BLOCK_SIZE_V
    h_start = pid_h * BLOCK_SIZE_H

    # We don't instantiate gumbel_max yet, because each program just writes
    # its local max into main memory for a parallel reduction in stage 2.

    offsets_v = v_start + tl.arange(0, BLOCK_SIZE_V)
    mask_v = offsets_v < vocab_size

    # H-tile within n_hidden_states
    offsets_h = h_start + tl.arange(0, BLOCK_SIZE_H)
    mask_h = offsets_h < n_hidden_states
    logits_blk = tl.zeros((BLOCK_SIZE_V, BLOCK_SIZE_H), dtype=tl.float32)
    if USE_PROTON_SCOPES:
        pl.exit_scope("setup")

    # Compute a block of logits logits_blk
    if USE_PROTON_SCOPES:
        pl.enter_scope("matmul-tile")
    for d_start in range(0, hidden_size, BLOCK_SIZE_D):
        offsets_d = d_start + tl.arange(0, BLOCK_SIZE_D)
        mask_d = offsets_d < hidden_size

        # load weights tile [BLOCK_SIZE_V, BLOCK_SIZE_D]
        w_blk = tl.load(
            weights_ptr + offsets_d[None, :] + hidden_size * offsets_v[:, None],
            mask=mask_v[:, None] & mask_d[None, :],
        )

        # load hidden_states tile [BLOCK_SIZE_H, BLOCK_SIZE_D]
        hidden_states_blk = tl.load(
            hidden_states_ptr + offsets_d[None, :] + hidden_size * offsets_h[:, None],
            mask=mask_h[:, None] & mask_d[None, :],
        )
        logits_blk = tl.dot(w_blk, hidden_states_blk.T, acc=logits_blk)
    if USE_PROTON_SCOPES:
        pl.exit_scope("matmul-tile")

    # Later we will take max over logits + noise, but rows outside the mask
    # should not be considered. Setting them to -inf achieves this.
    logits_blk = tl.where(mask_v[:, None], logits_blk, -float("inf"))
    logits_blk = logits_blk / temperature  # [Vblk, n_hidden_states]

    # Process samples in batches to limit memory usage
    samples_n_batches: tl.constexpr = triton.cdiv(num_samples, BLOCK_SIZE_NSAMPLES)
    for batch_idx in range(samples_n_batches):
        if USE_PROTON_SCOPES:
            pl.enter_scope("sample")
        # Calculate how many samples in this batch
        batch_start = batch_idx * BLOCK_SIZE_NSAMPLES
        batch_end = min(batch_start + BLOCK_SIZE_NSAMPLES, num_samples)
        actual_batch_size = batch_end - batch_start

        # Note: Creating appropriately sized tensors is tricky because
        # tl.arange() only accepts tl.constexpr that are powers of 2.
        noise_size: tl.constexpr = BLOCK_SIZE_V * BLOCK_SIZE_H * BLOCK_SIZE_NSAMPLES
        noise_offsets = tl.arange(0, noise_size).reshape(
            (BLOCK_SIZE_NSAMPLES, BLOCK_SIZE_V, BLOCK_SIZE_H)
        )
        # Note: Each tile (v, h) and batch of samples needs a different seed,
        # otherwise they all create the same noise, leading to sampling artifacts.
        # Compute gumbel noise directly to reduce register pressure
        if GUMBEL:
            logits_plus_noise = logits_blk - tl.log(
                -tl.log(
                    tl.rand(
                        seed + pid_v * 100 + pid_h * 1_000 + batch_idx * 10_000,
                        noise_offsets,
                    )
                )
            )
        else:
            logits_plus_noise = logits_blk[None, :, :]

        gumbel_max, gumbel_max_idx_local = tl.max(logits_plus_noise, axis=1, return_indices=True)
        gumbel_max_idx_global = gumbel_max_idx_local + v_start
        if USE_PROTON_SCOPES:
            pl.exit_scope("sample")

        if USE_PROTON_SCOPES:
            pl.enter_scope("store")
        # Output offset for this batch
        out_blk_start = pid_v * n_hidden_states * num_samples + batch_start

        # Note: It makes a difference if indices are row-major or column-major
        # Note: The stride needs to match the non-padded shape!
        out_offsets = (
            tl.arange(0, BLOCK_SIZE_NSAMPLES)[:, None]
            + num_samples * (h_start + tl.arange(0, BLOCK_SIZE_H))[None, :]
        )
        mask_nsamples = tl.arange(0, BLOCK_SIZE_NSAMPLES) < actual_batch_size
        out_mask = mask_nsamples[:, None] & mask_h[None, :]
        tl.store(
            max_out_ptr + out_blk_start + out_offsets,
            gumbel_max,
            mask=out_mask,
        )
        tl.store(
            max_out_idx_ptr + out_blk_start + out_offsets,
            gumbel_max_idx_global,
            mask=out_mask,
        )
        if USE_PROTON_SCOPES:
            pl.exit_scope("store")


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
        samples = torch.multinomial(probs, num_samples=num_samples, replacement=True)
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
            return SimpleSampler(lambda **kwargs: fused_mm_sample_triton(**kwargs, seed=0))
        case "fused-triton-no-gumbel":
            return SimpleSampler(
                lambda **kwargs: fused_mm_sample_triton(**kwargs, seed=0, GUMBEL=False)
            )
        case "naive-pt":
            return SimpleSampler(sample)
        case "naive-compiled":
            return SimpleSampler(sample_compiled)
        case "sequential-compiled":
            return SimpleSampler(sequential_sample_pt)
        case "naive-tl-matmul":
            return SimpleSampler(lambda **kwargs: sample(**kwargs, tl_matmul=True))
        case "jl-compiled":
            return JLSampler.from_weights(weights)
        case "flashinfer:top_k_top_p_sampling_from_logits":
            return SimpleSampler(
                lambda **kwargs: flashinfer_top_k_top_p_sampling_from_logits(
                    **kwargs,
                    # these params effectively disable top-p and top-k filtering
                    top_p=1.0,
                    top_k=-1,
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


def flashinfer_top_k_top_p_sampling_from_logits(
    weights: torch.Tensor,  # [V, D]
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    top_p: float,
    top_k: int,
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    logits, indices = flashinfer_create_logits_and_indices(
        weights, hidden_states, num_samples, temperature
    )
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
    logits = (logits / temperature).contiguous()
    indices = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.int32), num_samples
    )
    return logits, indices


@nvtx.annotate()
def flashinfer_sampling_from_logits(
    weights: torch.Tensor,  # [D, V]
    hidden_states: torch.Tensor,  # [n_hidden_states, D]
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    logits, indices = flashinfer_create_logits_and_indices(
        weights, hidden_states, num_samples, temperature
    )
    result = flashinfer.sampling.sampling_from_logits(logits=logits, indices=indices)
    return result.reshape(batch_size, num_samples)


def get_gpu_name() -> str:
    return torch.cuda.get_device_name()


def bsz_h(H: int) -> int:  # noqa: N803
    if H <= 16:
        return 16
    elif H <= 32:
        return 32
    return 64

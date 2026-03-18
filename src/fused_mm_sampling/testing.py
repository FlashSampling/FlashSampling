from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import torch
from scipy.stats import chisquare

from .core import get_sampler, set_torch_allocator_for_tma_descriptors
from .tp_info import TP1, TPInfo


@dataclass
class SyntheticInputs:
    """In the tensor-parallel case V is sharded among ranks"""

    weights: torch.Tensor  # [V, D], bfloat16
    hidden_states: torch.Tensor  # [n_hidden_states, D], bfloat16
    logits: (
        torch.Tensor
    )  # [n_hidden_states, V], float32 (the exact logits before bf16 quantization)
    vocab_size: int
    hidden_size: int


def make_synthetic_inputs(
    vocab_size: int = 256,
    hidden_size: int = 10,
    n_hidden_states: int = 2,
    device: torch.device = torch.device("cuda"),
    tp: TPInfo = TP1,
) -> SyntheticInputs:
    """Build weights and hidden_states that produce known logits.

    Creates up to two hidden states: one with ascending logits (favors high
    token indices) and one with descending logits (favors low token indices).
    All logits are shifted negative via :func:`shift_logits_negative`.
    """
    logits1 = torch.arange(-vocab_size / 2, vocab_size / 2, dtype=torch.float32)[None, :]
    logits2 = torch.arange(vocab_size / 2, -vocab_size / 2, step=-1, dtype=torch.float32)[None, :]
    all_logits = [logits1, logits2]
    logits = torch.cat(all_logits[:n_hidden_states], dim=0).to(device)
    n_hidden_states = logits.shape[0]

    U, _, _ = torch.linalg.svd(logits, full_matrices=False)  # noqa: N806

    torch.manual_seed(0)
    hidden_states = torch.cat(
        [U, torch.rand((n_hidden_states, hidden_size - n_hidden_states), device=device)],
        dim=1,
    ).to(device)
    weights = torch.linalg.pinv(hidden_states) @ logits  # [D, V]

    weights_bf16 = weights.bfloat16().T.contiguous()  # [V, D]
    hidden_states_bf16 = hidden_states.bfloat16()
    weights_bf16, hidden_states_bf16 = shift_logits_negative(
        weights_bf16,
        hidden_states_bf16,
        offset=float(vocab_size),
    )

    weights_bf16, hidden_states_bf16 = pad_to_tma_alignment(weights_bf16, hidden_states_bf16)
    weights_bf16 = shard_weights(weights_bf16, tp)

    return SyntheticInputs(
        weights=weights_bf16,
        hidden_states=hidden_states_bf16,
        logits=logits,
        vocab_size=vocab_size,
        hidden_size=weights_bf16.shape[1],
    )


def pad_to_tma_alignment(
    weights: torch.Tensor, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad D to 16-byte alignment for TMA on SM 90+ (H100, B200, etc.).

    TMA requires the innermost tensor dimension to be 16-byte aligned.
    For bf16 (2 bytes), D must be a multiple of 8. After shift_logits_negative
    adds a bias column (D = hidden_size + 1), D=11 is not aligned. Zero-padding
    extra columns preserves logits (they contribute nothing to the matmul).
    """
    d = weights.shape[1]
    aligned_d = (d + 7) & ~7  # next multiple of 8 bf16 elements = 16 bytes
    if aligned_d > d:
        pad = aligned_d - d
        weights = torch.nn.functional.pad(weights, (0, pad))
        hidden_states = torch.nn.functional.pad(hidden_states, (0, pad))
    return weights, hidden_states


def shard_weights(weights: torch.Tensor, tp: TPInfo) -> torch.Tensor:
    """Shard weights along vocab dim (same as vLLM's VocabParallelEmbedding.weight_loader)."""
    if tp.size == 1:
        return weights  # early return for single-GPU case
    shard_size = weights.shape[0] // tp.size
    start_idx = tp.rank * shard_size
    shard = weights.narrow(0, start_idx, shard_size)
    assert shard.is_contiguous()
    return shard


def assert_sampling_distribution(
    provider: str,
    vocab_size: int,
    n_hidden_states: int,
    num_samples: int = 10_000,
    temperature_val: float = 5.0,
    tp: TPInfo = TP1,
) -> None:
    """Verify that a sampler produces the correct distribution.

    Uses synthetic inputs with known logit vectors (ascending and/or descending),
    draws many samples, and checks that each empirical distribution fits the
    theoretical softmax probabilities via a chi-squared test.
    """
    device = torch.device("cuda")
    temperature = torch.tensor(temperature_val, device=device)

    inputs = make_synthetic_inputs(
        vocab_size=vocab_size,
        n_hidden_states=n_hidden_states,
        tp=tp,
    )
    sampler = get_sampler(provider, weights=inputs.weights)
    sampler.prepare()
    samples = sampler.sample(
        weights=inputs.weights,
        hidden_states=inputs.hidden_states,
        num_samples=num_samples,
        temperature=temperature,
        tp=tp,
        seed=tp.rank * 1_000_000,
    )

    for seq_idx in range(inputs.logits.shape[0]):
        expected_probs = (inputs.logits[seq_idx] / temperature).softmax(dim=0)
        expected_counts = (expected_probs * num_samples).cpu().numpy()
        empirical_counts = (
            torch.bincount(samples[seq_idx], minlength=inputs.vocab_size).float().cpu().numpy()
        )

        mask = expected_counts >= 5
        obs = empirical_counts[mask]
        exp = expected_counts[mask]
        exp = exp * (obs.sum() / exp.sum())
        _, p_value = chisquare(obs, exp)
        assert not np.isnan(p_value), (
            f"Chi-squared returned NaN for seq {seq_idx} — likely all samples "
            f"landed in a single tile. {provider} may have a masked-fill bug."
        )
        assert p_value > 0.001, (
            f"Sampling distribution mismatch for seq {seq_idx}: p={p_value:.6f}. "
            f"{provider} does not match the expected softmax distribution."
        )


def run_sampling_distribution_tp2() -> None:
    """Worker function for TP2 sampling distribution tests (passed to run_maybe_distributed)."""
    set_torch_allocator_for_tma_descriptors()
    tp = TPInfo.from_world()
    providers = [
        "fused-triton",
        "naive-pt",
        "naive-compiled",
        "flashinfer:sampling_from_logits",
        "flashinfer:top_k_top_p_sampling_from_logits",
    ]
    for provider, vocab_size, n_hidden_states in product(providers, [100, 200, 256], [1, 2]):
        assert_sampling_distribution(provider, vocab_size, n_hidden_states, tp=tp)
        tp.rank0_print(f"✅ Passed: {provider} V={vocab_size} H={n_hidden_states}")


def run_greedy_tp2() -> None:
    """Worker function for TP2 greedy tests (passed to run_maybe_distributed)."""
    set_torch_allocator_for_tma_descriptors()
    tp = TPInfo.from_world()
    for vocab_size, n_hidden_states in product([100, 200, 256], [1, 2]):
        inputs = make_synthetic_inputs(
            vocab_size=vocab_size, n_hidden_states=n_hidden_states, tp=tp
        )
        sampler = get_sampler("greedy", weights=inputs.weights)
        sampler.prepare()
        samples = sampler.sample(
            weights=inputs.weights,
            hidden_states=inputs.hidden_states,
            num_samples=1,
            temperature=torch.empty((), device="cuda"),
            tp=tp,
        )
        expected = inputs.logits.argmax(dim=-1)
        torch.testing.assert_close(samples[:, 0], expected)
        tp.rank0_print(f"✅ Passed: greedy V={vocab_size} H={n_hidden_states}")


def shift_logits_negative(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    offset: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift all logits by -offset without touching the existing weights.

    Appends a bias column so that ``h_new @ W_new^T = h @ W^T - offset``.
    Since softmax is shift-invariant the expected sampling distribution is
    unchanged, but the all-negative logits exercise masked-fill handling in
    partial V-tiles (kernels must fill masked rows with -inf, not 0, or the
    0 beats all real negative values in the tile-max reduction).

    We use a bias column instead of baking the offset into the logits before
    computing the pseudoinverse because bf16 cannot represent fine-grained
    all-negative logits for vocab sizes above ~128.  The bias column keeps the
    original weights (centered near 0) intact and encodes the offset exactly.
    """
    vocab_size = weights.shape[0]
    n_hidden_states = hidden_states.shape[0]
    device = weights.device
    dtype = weights.dtype
    bias_w = torch.ones(vocab_size, 1, dtype=dtype, device=device)
    bias_h = torch.full((n_hidden_states, 1), -offset, dtype=dtype, device=device)
    return (
        torch.cat([weights, bias_w], dim=1),
        torch.cat([hidden_states, bias_h], dim=1),
    )

import os
from itertools import product
from pathlib import Path

os.environ.setdefault("HELION_AUTOTUNE_EFFORT", "none")

import numpy as np
import pytest
import torch
from scipy.stats import chisquare

from fused_mm_sampling.bench.speed_test import Args, run_speed_test
from fused_mm_sampling.core import JLSampler, bsz_h, get_sampler
from fused_mm_sampling.testing import assert_sampling_distribution, make_synthetic_inputs
from fused_mm_sampling.tp_info import TPInfo, run_maybe_distributed

device = torch.device("cuda")


def test_jl_sampling_aproximate_correctness():
    folder = Path(__file__).parent / "qwen3-0.6b"
    assert folder.exists(), f"{folder} does not exist. Run generate_inputs.py first"
    weights = torch.load(folder / "weights.pt", map_location=device)
    hidden_states = torch.load(folder / "hidden_states.pt", map_location=device)
    expected_logits = hidden_states @ weights.T
    expected_probs = expected_logits.softmax(dim=1)

    jl_sampler = JLSampler.from_weights(weights, epsilon=0.2).prepare()
    actual_logits = jl_sampler.compute_logits(hidden_states)
    actual_probs = actual_logits.softmax(dim=1)
    assert torch.allclose(actual_probs, expected_probs, atol=0.2)

    print(f"{weights.shape=}")
    print(f"{hidden_states.shape=}")


@pytest.mark.parametrize(
    "args",
    [
        # H, expected BSZ_H
        (1, 16),
        (16, 16),
        (17, 32),
        (32, 32),
        (33, 64),
        (64, 64),
        (65, 64),
    ],
)
def test_bsz_h(args):
    h, expected_bsz_h = args
    assert bsz_h(h) == expected_bsz_h


@pytest.mark.parametrize("n_hidden_states", [1, 2])
@pytest.mark.parametrize("vocab_size", [100, 200, 256])
@pytest.mark.parametrize(
    "provider",
    [
        "fused-triton",
        "fused-cuda",
        "naive-pt",
        "naive-compiled",
        "pt-qitra",
        "sequential-compiled",
        "helion",
        "flashinfer:sampling_from_logits",
        "flashinfer:top_k_top_p_sampling_from_logits",
    ],
)
def test_sampling_distribution(provider, vocab_size, n_hidden_states):
    """Verify that a sampler produces the correct distribution.

    Uses synthetic inputs with known logit vectors (ascending and/or descending),
    draws many samples, and checks that each empirical distribution fits the
    theoretical softmax probabilities via a chi-squared test.
    """
    inputs = make_synthetic_inputs(vocab_size=vocab_size, n_hidden_states=n_hidden_states)
    num_samples = 10_000
    temperature = torch.tensor(5.0, device=device)

    sampler = get_sampler(provider, weights=inputs.weights)
    sampler.prepare()
    samples = sampler.sample(
        weights=inputs.weights,
        hidden_states=inputs.hidden_states,
        num_samples=num_samples,
        temperature=temperature,
    )

    for seq_idx in range(inputs.logits.shape[0]):
        # Compare empirical counts against theoretical expected counts from softmax.
        expected_probs = (inputs.logits[seq_idx] / temperature).softmax(dim=0)
        expected_counts = (expected_probs * num_samples).cpu().numpy()
        empirical_counts = (
            torch.bincount(samples[seq_idx], minlength=inputs.vocab_size).float().cpu().numpy()
        )

        # Only test bins with enough expected counts for chi-squared to be valid.
        mask = expected_counts >= 5
        obs = empirical_counts[mask]
        exp = expected_counts[mask]
        # Rescale expected counts so sums match (samples in excluded bins shift the totals).
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


@pytest.mark.skipif(
    not os.environ.get("FMMS_TEST_DISTRIBUTED"),
    reason="Set FMMS_TEST_DISTRIBUTED=1 to run",
)
def test_sampling_distribution_tp2() -> None:
    run_maybe_distributed(_run_distributed_testcases, n_procs=2)


def _run_distributed_testcases() -> None:
    tp = TPInfo.from_world()
    providers = [
        "fused-triton",
        "naive-pt",
        "naive-compiled",
        "flashinfer:sampling_from_logits",
        "flashinfer:top_k_top_p_sampling_from_logits",
    ]
    vocab_sizes = [100, 200, 256]
    n_hidden_states_list = [1, 2]
    combinations = product(providers, vocab_sizes, n_hidden_states_list)
    for provider, vocab_size, n_hidden_states in combinations:
        assert_sampling_distribution(provider, vocab_size, n_hidden_states, tp=tp)
        if tp.rank == 0:
            msg = f"✅ Passed: {provider} vocab_size={vocab_size} n_hidden_states={n_hidden_states}"
            print(msg)


@pytest.mark.parametrize("n_hidden_states", [1, 2])
@pytest.mark.parametrize("vocab_size", [100, 200, 256])
@pytest.mark.parametrize("provider", ["naive-pt", "naive-compiled", "fused-topk"])
def test_top_k_top_p(provider, vocab_size, n_hidden_states):
    """Verify that top-k and top-p filtering restricts samples to the expected tokens."""
    inputs = make_synthetic_inputs(vocab_size=vocab_size, n_hidden_states=n_hidden_states)
    temperature = torch.tensor(1.0, device=device)
    top_k = 10
    top_p = 0.9
    num_samples = 5_000

    sampler = get_sampler(provider, weights=inputs.weights)
    sampler.prepare()
    samples = sampler.sample(
        weights=inputs.weights,
        hidden_states=inputs.hidden_states,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Use the same bf16 matmul logits the sampler sees, not the exact float32 logits.
    ref_logits = inputs.hidden_states @ inputs.weights.T
    for seq_idx in range(n_hidden_states):
        allowed_tokens = reference_top_k_top_p(
            logits=ref_logits[seq_idx], temperature=temperature, top_k=top_k, top_p=top_p
        )
        unique_sampled = torch.unique(samples[seq_idx])
        allowed_set = set(allowed_tokens.cpu().tolist())
        sampled_set = set(unique_sampled.cpu().tolist())
        assert sampled_set <= allowed_set, (
            f"seq {seq_idx}: sampled tokens not in allowed set. Extra: {sampled_set - allowed_set}"
        )
        assert sampled_set == allowed_set, (
            f"seq {seq_idx}: not all allowed tokens sampled. Missing: {allowed_set - sampled_set}"
        )


@pytest.mark.parametrize("provider", ["naive-pt", "fused-topk"])
def test_top_k_top_p_large_vocab(provider):
    """Test top-k + top-p at V=151936 with real Qwen3-0.6b weights.

    At large vocab sizes, tl.dot and cuBLAS produce different logits due to bf16
    accumulation order, so we can't compare against a shared reference. Instead we
    verify: no crash, valid token range, and top-k actually restricts output.
    """
    folder = Path(__file__).parent / "qwen3-0.6b"
    assert folder.exists(), f"{folder} does not exist. Run generate_inputs.py first"
    weights = torch.load(folder / "weights.pt", map_location=device)
    hidden_states = torch.load(folder / "hidden_states.pt", map_location=device)
    V = weights.shape[0]  # noqa: N806
    H = hidden_states.shape[0]  # noqa: N806
    top_k = 10
    top_p = 0.9
    num_samples = 5_000
    temperature = torch.tensor(1.0, device=device)

    sampler = get_sampler(provider, weights=weights)
    sampler.prepare()
    samples = sampler.sample(
        weights=weights,
        hidden_states=hidden_states,
        num_samples=num_samples,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    assert samples.shape == (H, num_samples)
    for seq_idx in range(H):
        unique_sampled = torch.unique(samples[seq_idx])
        assert unique_sampled.min() >= 0, f"seq {seq_idx}: negative token id"
        assert unique_sampled.max() < V, f"seq {seq_idx}: token id >= vocab_size"
        assert len(unique_sampled) <= top_k, (
            f"seq {seq_idx}: {len(unique_sampled)} unique tokens, expected <= {top_k}"
        )


def reference_top_k_top_p(
    logits: torch.Tensor,  # [V], float32
    temperature: torch.Tensor,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Return the set of token indices allowed after top-k then top-p filtering."""
    scaled = logits.float() / temperature
    topk_values, topk_indices = scaled.topk(top_k)
    probs_topk = topk_values.softmax(dim=-1)
    sorted_probs, sorted_order = probs_topk.sort(descending=True)
    cumsum = sorted_probs.cumsum(dim=0)
    mask = cumsum - sorted_probs < top_p
    return topk_indices[sorted_order[mask]]


def test_speed_test_smoke():
    # name=None means all providers are tested
    args = Args(name=None, n_runs_warmup=1, n_runs_benchmark=1, case="small")
    run_speed_test(args)

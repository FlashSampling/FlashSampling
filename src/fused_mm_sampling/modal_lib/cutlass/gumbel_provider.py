"""Gate 4 Phase 1 deterministic checks for B200 CUTLASS Gumbel-Max."""

import json
import re
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pandas as pd

from ..utils import make_app, make_volumes, set_volume_caches
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(make_cutlass_provider_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/18-gumbel-provider")
SEEDS = (0, 1, 0x0123456789ABCDEF)
MODEL_SHAPES = ((151_936, 4_096), (128_256, 8_192))
HIDDEN_STATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


@app.function(
    gpu="B200", image=image, volumes=make_volumes(), timeout=30 * 60
)
def record_sm100() -> list[dict]:
    import torch

    from fused_mm_sampling.core import get_sampler

    set_volume_caches()
    rows = []
    for seed in SEEDS:
        rows.append(_run_reference_case(get_sampler, torch, seed, 257, 2, 1.0))
    rows.append(_run_reference_case(get_sampler, torch, 7, 257, 2, 0.5))
    rows.append(_run_reference_case(get_sampler, torch, 7, 257, 2, 2.0))
    rows.append(_run_batched_reference_case(get_sampler, torch))
    rows.extend(_run_reproducibility_cases(get_sampler, torch))
    rows.append(_run_schedule_invariance_case(get_sampler, torch))
    return rows


@app.function(
    gpu="B200", image=image, volumes=make_volumes(), timeout=2 * 60 * 60
)
def record_distribution_sm100() -> dict:
    from fused_mm_sampling.testing import assert_sampling_distribution_large_vocab

    set_volume_caches()
    output = StringIO()
    with redirect_stdout(output):
        assert_sampling_distribution_large_vocab(
            vocab_size=128_000,
            num_samples=10_000_000,
            samples_per_call=10_000,
            hidden_size=16,
            provider="fused-cutlass",
        )
    log = output.getvalue()
    match = re.search(
        r"tested_bins=(\d+), tested_probability_mass=([0-9.]+), "
        r"statistic=([0-9.]+), df=(\d+), "
        r"reduced_statistic=([0-9.]+), p=([0-9.eE+-]+)",
        log,
    )
    if match is None:
        raise RuntimeError(f"Could not parse distribution statistics:\n{log}")
    return {
        "log": log,
        "tested_bins": int(match.group(1)),
        "tested_probability_mass": float(match.group(2)),
        "chi_squared": float(match.group(3)),
        "degrees_of_freedom": int(match.group(4)),
        "reduced_chi_squared": float(match.group(5)),
        "p_value": float(match.group(6)),
    }


@app.function(
    gpu="B200", image=image, volumes=make_volumes(), timeout=60 * 60
)
def record_performance_sm100() -> list[dict]:
    import torch

    from fused_mm_sampling.core import get_sampler

    set_volume_caches()
    rows = []
    torch.manual_seed(0)
    temperature = torch.tensor(1.0, device="cuda")
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")
    for vocab_size, hidden_size in MODEL_SHAPES:
        weights = torch.randn(
            (vocab_size, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        greedy_sampler = get_sampler("fused-cutlass-greedy", weights=weights)
        sampling_sampler = get_sampler("fused-cutlass", weights=weights)
        for hidden_count in HIDDEN_STATES:
            hidden_states = torch.randn(
                (hidden_count, hidden_size), dtype=torch.bfloat16, device="cuda"
            )
            common = {
                "weights": weights,
                "hidden_states": hidden_states,
                "num_samples": 1,
                "temperature": temperature,
            }

            def greedy():
                return greedy_sampler.sample(**common)

            def sampling():
                return sampling_sampler.sample(**common, seed=17)

            for _ in range(10):
                greedy()
                sampling()
            torch.cuda.synchronize()
            for repetition in range(50):
                order = (("greedy", greedy), ("gumbel", sampling))
                if repetition % 2:
                    order = tuple(reversed(order))
                for provider, function in order:
                    cache.zero_()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    function()
                    end.record()
                    end.synchronize()
                    rows.append(
                        {
                            "vocab_size": vocab_size,
                            "hidden_size": hidden_size,
                            "n_hidden_states": hidden_count,
                            "provider": provider,
                            "repetition": repetition,
                            "latency_ms": start.elapsed_time(end),
                        }
                    )
    return rows


def _run_reference_case(get_sampler, torch, seed, vocab_size, hidden_count, temperature):
    weights = torch.zeros(
        (vocab_size, 64), dtype=torch.bfloat16, device="cuda"
    )
    hidden_states = torch.zeros(
        (hidden_count, 64), dtype=torch.bfloat16, device="cuda"
    )
    hidden_states[:, 0] = 1
    weights[:, 0] = torch.linspace(
        -1, 1, vocab_size, dtype=torch.bfloat16, device="cuda"
    )
    temperature_tensor = torch.tensor(temperature, device="cuda")
    sampler = get_sampler("fused-cutlass", weights=weights)
    actual = sampler.sample(
        weights=weights,
        hidden_states=hidden_states,
        num_samples=1,
        temperature=temperature_tensor,
        seed=seed,
    )[:, 0].tolist()
    logits = weights[:, 0].float().cpu().tolist()
    expected = [
        max(
            range(vocab_size),
            key=lambda vocab: (
                logits[vocab] / temperature
                + _gumbel(seed, 0, hidden, vocab),
                -vocab,
            ),
        )
        for hidden in range(hidden_count)
    ]
    passed = actual == expected
    row = {
        "case": f"reference_seed_{seed}_temperature_{temperature}",
        "expected": ";".join(map(str, expected)),
        "actual": ";".join(map(str, actual)),
        "pass": int(passed),
    }
    if not passed:
        raise RuntimeError(f"CUTLASS Gumbel reference mismatch: {row}")
    return row


def _run_batched_reference_case(get_sampler, torch):
    vocab_size = 257
    hidden_count = 2
    num_samples = 17
    seed = 314
    weights = torch.zeros((vocab_size, 64), dtype=torch.bfloat16, device="cuda")
    hidden_states = torch.zeros(
        (hidden_count, 64), dtype=torch.bfloat16, device="cuda"
    )
    temperature = torch.tensor(1.0, device="cuda")
    sampler = get_sampler("fused-cutlass", weights=weights)
    actual = sampler.sample(
        weights=weights,
        hidden_states=hidden_states,
        num_samples=num_samples,
        temperature=temperature,
        seed=seed,
    ).tolist()
    expected = [
        [
            max(
                range(vocab_size),
                key=lambda vocab: (_gumbel(seed, sample, hidden, vocab), -vocab),
            )
            for sample in range(num_samples)
        ]
        for hidden in range(hidden_count)
    ]
    passed = actual == expected
    row = {
        "case": "batched_exact_reference_h2_s17",
        "expected": ";".join(map(str, expected[0])),
        "actual": ";".join(map(str, actual[0])),
        "pass": int(passed),
    }
    if not passed:
        raise RuntimeError(f"CUTLASS batched Gumbel mismatch: {row}")
    return row


def _run_reproducibility_cases(get_sampler, torch):
    weights = torch.zeros((4096, 64), dtype=torch.bfloat16, device="cuda")
    hidden_states = torch.zeros((16, 64), dtype=torch.bfloat16, device="cuda")
    temperature = torch.tensor(1.0, device="cuda")
    sampler = get_sampler("fused-cutlass", weights=weights)

    def sample(seed):
        return sampler.sample(
            weights=weights,
            hidden_states=hidden_states,
            num_samples=1,
            temperature=temperature,
            seed=seed,
        )[:, 0]

    first = sample(99)
    repeated = sample(99)
    different = sample(100)
    return [
        {
            "case": "same_seed_reproducibility",
            "expected": "identical",
            "actual": "identical" if torch.equal(first, repeated) else "different",
            "pass": int(torch.equal(first, repeated)),
        },
        {
            "case": "different_seed_separation",
            "expected": "at_least_one_difference",
            "actual": str(int(torch.count_nonzero(first != different))),
            "pass": int(torch.any(first != different)),
        },
    ]


def _run_schedule_invariance_case(get_sampler, torch):
    weights = torch.zeros((4096, 64), dtype=torch.bfloat16, device="cuda")
    temperature = torch.tensor(1.0, device="cuda")
    sampler = get_sampler("fused-cutlass", weights=weights)
    outputs = []
    for hidden_count in (64, 128, 256):
        hidden_states = torch.zeros(
            (hidden_count, 64), dtype=torch.bfloat16, device="cuda"
        )
        outputs.append(
            sampler.sample(
                weights=weights,
                hidden_states=hidden_states,
                num_samples=1,
                temperature=temperature,
                seed=1234,
            )[:, 0]
        )
    passed = all(torch.equal(outputs[0], output[:64]) for output in outputs[1:])
    return {
        "case": "schedule_invariance_h64_h128_h256",
        "expected": "identical_first_64",
        "actual": "identical" if passed else "different",
        "pass": int(passed),
    }


def _gumbel(seed: int, sample: int, hidden: int, vocab: int) -> float:
    import math

    uniform = ((_philox(seed, sample, hidden, vocab)[0] >> 9) + 0.5) * 2**-23
    return -math.log(-math.log(uniform))


def _philox(seed: int, sample: int, hidden: int, vocab: int) -> tuple[int, ...]:
    mask = 0xFFFFFFFF
    value = [vocab & mask, vocab >> 32, hidden, sample]
    key0, key1 = seed & mask, seed >> 32
    for _ in range(10):
        product0 = 0xD2511F53 * value[0]
        product1 = 0xCD9E8D57 * value[2]
        value = [
            ((product1 >> 32) ^ value[1] ^ key0) & mask,
            product1 & mask,
            ((product0 >> 32) ^ value[3] ^ key1) & mask,
            product0 & mask,
        ]
        key0 = (key0 + 0x9E3779B9) & mask
        key1 = (key1 + 0xBB67AE85) & mask
    return tuple(value)


@app.local_entrypoint()
def main(phase: str = "deterministic") -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if phase == "distribution":
        result = record_distribution_sm100.remote()
        (OUTPUT_DIR / "distribution-log.txt").write_text(result.pop("log"))
        summary = {
            "gate": "4-phase-2",
            "command": (
                "make modal-cutlass GATE=gumbel-provider "
                'MODAL_ARGS="--phase distribution"'
            ),
            "status": "pass",
            "gpu": "B200",
            "provider": "fused-cutlass",
            "vocab_size": 128_000,
            "num_samples": 10_000_000,
            "samples_per_call": 10_000,
            **result,
            "remaining_phase": "performance feasibility decision",
        }
        (OUTPUT_DIR / "distribution-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        return
    if phase == "performance":
        timings = pd.DataFrame(record_performance_sm100.remote())
        timings.to_csv(OUTPUT_DIR / "performance-timings.csv", index=False)
        keys = ["vocab_size", "hidden_size", "n_hidden_states", "provider"]
        summary = (
            timings.groupby(keys, as_index=False)
            .agg(
                repetitions=("latency_ms", "count"),
                median_ms=("latency_ms", "median"),
                p10_ms=("latency_ms", lambda values: values.quantile(0.1)),
                p90_ms=("latency_ms", lambda values: values.quantile(0.9)),
            )
        )
        configs = keys[:-1]
        greedy = (
            summary.query("provider == 'greedy'")[configs + ["median_ms"]]
            .rename(columns={"median_ms": "greedy_median_ms"})
        )
        summary = summary.merge(
            greedy, on=configs, how="left", validate="many_to_one"
        ).assign(ratio_to_greedy=lambda frame: frame["median_ms"] / frame["greedy_median_ms"])
        summary.to_csv(OUTPUT_DIR / "performance-summary.csv", index=False)
        gumbel = summary.query("provider == 'gumbel'")
        decision = {
            "gate": "4-phase-3-timing",
            "command": (
                "make modal-cutlass GATE=gumbel-provider "
                'MODAL_ARGS="--phase performance"'
            ),
            "status": "pass" if gumbel["ratio_to_greedy"].le(1.2).all() else "fail",
            "gpu": "B200",
            "repetitions": 50,
            "measurement": "cold-L2 CUDA events, alternating provider order",
            "predeclared_ratio_threshold": 1.2,
            "worst_ratio": float(gumbel["ratio_to_greedy"].max()),
            "ncu_command": "make modal-cutlass GATE=gumbel-ncu",
        }
        (OUTPUT_DIR / "performance-decision.json").write_text(
            json.dumps(decision, indent=2) + "\n"
        )
        return
    if phase != "deterministic":
        raise ValueError(
            "phase must be 'deterministic', 'distribution', or 'performance'"
        )
    cases = pd.DataFrame(record_sm100.remote())
    cases.to_csv(OUTPUT_DIR / "cases.csv", index=False)
    passed = cases["pass"].eq(1).all()
    summary = {
        "gate": "4-phase-1",
        "command": "make modal-cutlass GATE=gumbel-provider",
        "status": "pass" if passed else "fail",
        "gpu": "B200",
        "provider": "fused-cutlass",
        "case_count": len(cases),
        "failure_count": int(cases["pass"].ne(1).sum()),
        "limitations": ["BF16", "TP1", "SM100", "batched GEMM columns <= 256"],
        "remaining_phases": [
            "10M-sample large-vocabulary chi-squared test",
            "matched greedy-versus-Gumbel timing and NCU profile",
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if not passed:
        raise RuntimeError("Gate 4 deterministic phase failed")

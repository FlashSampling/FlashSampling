"""Shared correctness and timing runner for CUTLASS sampling experiments."""

import json
import math
from pathlib import Path

import pandas as pd

from ...cutlass_experiments import get_cutlass_sampling_experiment
from ...dev_metrics import emit_dev_event, timed_dev_stage
from ..utils import (
    make_app,
    make_volumes,
    reload_shared_volume,
    set_volume_caches,
)
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(make_cutlass_provider_image())

MODEL_SHAPES = ((151_936, 4_096), (128_256, 8_192))
HIDDEN_STATES = (64, 128, 256)
REFERENCE_HIDDEN_STATES = (128, 256)
REPETITIONS = 20


@app.function(gpu="B200", image=image, volumes=make_volumes(), timeout=60 * 60)
def record_sm100(variant: str) -> dict:
    import torch

    from fused_mm_sampling import cutlass_impl
    from fused_mm_sampling.core import get_sampler
    from fused_mm_sampling.cutlass_experiments import (
        get_cutlass_sampling_experiment,
    )

    reload_shared_volume()
    set_volume_caches()
    get_cutlass_sampling_experiment(variant)

    def candidate(**kwargs):
        return cutlass_impl.fused_mm_sample_cutlass_experimental(
            **kwargs, variant=variant
        )

    emit_dev_event("remote_start", variant=variant)
    cutlass_impl._get_experimental_sampling_module(variant)
    with timed_dev_stage("correctness", variant=variant):
        correctness = _run_reference_cases(candidate, torch)
    with timed_dev_stage("timing", variant=variant):
        rows = _run_timings(candidate, get_sampler, torch, variant)
    return {"correctness": correctness, "timings": rows}


def _run_timings(candidate, get_sampler, torch, variant: str) -> list[dict]:
    torch.manual_seed(0)
    temperature = torch.tensor(1.0, device="cuda")
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")
    rows = []
    for vocab_size, hidden_size in MODEL_SHAPES:
        weights = torch.randn(
            (vocab_size, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        triton_sampler = get_sampler("fused-triton", weights=weights)
        for hidden_count in HIDDEN_STATES:
            hidden_states = torch.randn(
                (hidden_count, hidden_size), dtype=torch.bfloat16, device="cuda"
            )
            common = {
                "weights": weights,
                "hidden_states": hidden_states,
                "num_samples": 1,
                "temperature": temperature,
                "seed": 17,
            }
            providers = {
                variant: lambda: candidate(**common),
                "triton": lambda: triton_sampler.sample(**common),
            }
            for _ in range(5):
                for function in providers.values():
                    function()
            torch.cuda.synchronize()
            names = tuple(providers)
            for repetition in range(REPETITIONS):
                offset = repetition % len(names)
                order = names[offset:] + names[:offset]
                for provider in order:
                    cache.zero_()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    providers[provider]()
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


def _run_reference_cases(candidate, torch) -> list[dict]:
    vocab_size = 257
    hidden_size = 64
    seed = 17
    temperature = torch.tensor(1.0, device="cuda")
    weights = torch.zeros(
        (vocab_size, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    weights[:, 0] = torch.linspace(
        -1, 1, vocab_size, dtype=torch.bfloat16, device="cuda"
    )
    logits = weights[:, 0].float().cpu().tolist()
    rows = []
    for hidden_count in REFERENCE_HIDDEN_STATES:
        hidden_states = torch.zeros(
            (hidden_count, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        hidden_states[:, 0] = 1
        actual = candidate(
            weights=weights,
            hidden_states=hidden_states,
            num_samples=1,
            temperature=temperature,
            seed=seed,
        )[:, 0].tolist()
        expected = [
            max(
                range(vocab_size),
                key=lambda vocab: (
                    logits[vocab] + _gumbel(seed, 0, hidden, vocab),
                    -vocab,
                ),
            )
            for hidden in range(hidden_count)
        ]
        passed = actual == expected
        rows.append(
            {
                "n_hidden_states": hidden_count,
                "expected": ";".join(map(str, expected)),
                "actual": ";".join(map(str, actual)),
                "pass": int(passed),
            }
        )
        if not passed:
            raise RuntimeError(f"CUTLASS experiment reference mismatch at H={hidden_count}")
    return rows


def _gumbel(seed: int, sample: int, hidden: int, vocab: int) -> float:
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
def main(variant: str, output_dir: str) -> None:
    get_cutlass_sampling_experiment(variant)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = record_sm100.remote(variant)
    correctness = pd.DataFrame(result["correctness"])
    correctness.to_csv(output_dir / "correctness.csv", index=False)
    timings = pd.DataFrame(result["timings"])
    timings.to_csv(output_dir / "timings.csv", index=False)
    keys = ["vocab_size", "hidden_size", "n_hidden_states", "provider"]
    summary = timings.groupby(keys, as_index=False).agg(
        repetitions=("latency_ms", "count"),
        median_ms=("latency_ms", "median"),
        p10_ms=("latency_ms", lambda values: values.quantile(0.1)),
        p90_ms=("latency_ms", lambda values: values.quantile(0.9)),
    )
    configs = keys[:-1]
    triton = summary.query("provider == 'triton'")[
        configs + ["median_ms"]
    ].rename(columns={"median_ms": "triton_median_ms"})
    summary = summary.merge(
        triton, on=configs, how="left", validate="many_to_one"
    ).assign(
        ratio_to_triton=lambda frame: frame["median_ms"]
        / frame["triton_median_ms"]
    )
    summary.to_csv(output_dir / "timing-summary.csv", index=False)
    candidate = summary.query("provider == @variant")
    decision = {
        "gate": f"4-{variant}-screen",
        "gpu": "B200",
        "correctness_cases": len(correctness),
        "correctness_passed": bool(
            correctness.empty or correctness["pass"].eq(1).all()
        ),
        "repetitions": REPETITIONS,
        "measurement": "cold-L2 CUDA events, alternating provider order",
        "cutlass_faster_cells": int(candidate["ratio_to_triton"].lt(1).sum()),
        "cell_count": len(candidate),
        "best_ratio_to_triton": float(candidate["ratio_to_triton"].min()),
        "worst_ratio_to_triton": float(candidate["ratio_to_triton"].max()),
        "ncu_policy": "run for diagnostic value regardless of timing decision",
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )

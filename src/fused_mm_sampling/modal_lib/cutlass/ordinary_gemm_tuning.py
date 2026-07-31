"""Retained-candidate tuning sweep for the ordinary CUTLASS GEMM prerequisite."""

import json
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .ordinary_gemm_common import (
    BENCHMARK_REPETITIONS,
    HIDDEN_STATES,
    MAXIMUM_RATIO,
    MODEL_SHAPES,
    WARMUP_REPETITIONS,
    benchmark,
)
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(make_cutlass_provider_image())

OUTPUT_DIR = Path(
    "benchmarking/modal-results/cutlass/14-ordinary-gemm-tuning"
)
ARCHITECTURES = ("sm90", "sm100")
VARIANTS = (
    "tile-64x128x64-auto",
    "tile-128x64x64-auto",
    "tile-128x128x64-auto",
    "tile-64x128x64-native",
    "tile-128x64x64-native",
    "tile-128x128x64-native",
)


@app.function(gpu="H100", image=image, timeout=60 * 60)
def record_sm90() -> dict:
    return _run("sm90")


@app.function(gpu="B200", image=image, timeout=60 * 60)
def record_sm100() -> dict:
    return _run("sm100")


def _run(architecture: str) -> dict:
    import torch

    from fused_mm_sampling.cutlass_impl import (
        cutlass_launch_plain_gemm_variant,
        cutlass_make_plain_gemm_buffers,
    )

    rows = []
    correctness = []
    for vocab_size, hidden_size in MODEL_SHAPES:
        weights = torch.randn(
            (vocab_size, hidden_size),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for n_hidden_states in HIDDEN_STATES:
            hidden_states = torch.randn(
                (n_hidden_states, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            padded, cutlass_output, gemm_n = (
                cutlass_make_plain_gemm_buffers(weights, hidden_states)
            )
            cublas_output = torch.empty_like(cutlass_output)

            def cublas():
                torch.mm(weights, padded.T, out=cublas_output)

            cublas()
            torch.cuda.synchronize()
            _append_timings(
                rows,
                architecture,
                vocab_size,
                hidden_size,
                n_hidden_states,
                gemm_n,
                "cublas",
                cublas,
            )
            for variant in VARIANTS:

                def cutlass(variant=variant):
                    cutlass_launch_plain_gemm_variant(
                        variant, weights, padded, cutlass_output
                    )

                cutlass()
                torch.cuda.synchronize()
                difference = (
                    cutlass_output.float() - cublas_output.float()
                ).abs()
                correctness.append(
                    {
                        "architecture": architecture,
                        "variant": variant,
                        "vocab_size": vocab_size,
                        "hidden_size": hidden_size,
                        "n_hidden_states": n_hidden_states,
                        "gemm_n": gemm_n,
                        "max_abs_difference": float(difference.max()),
                        "mean_abs_difference": float(difference.mean()),
                        "finite": int(
                            torch.isfinite(cutlass_output).all()
                            and torch.isfinite(cublas_output).all()
                        ),
                    }
                )
                _append_timings(
                    rows,
                    architecture,
                    vocab_size,
                    hidden_size,
                    n_hidden_states,
                    gemm_n,
                    variant,
                    cutlass,
                )
        del weights
    return {"timings": rows, "correctness": correctness}


def _append_timings(
    rows: list[dict],
    architecture: str,
    vocab_size: int,
    hidden_size: int,
    n_hidden_states: int,
    gemm_n: int,
    variant: str,
    function,
) -> None:
    for repetition, latency_ms in enumerate(benchmark(function)):
        rows.append(
            {
                "architecture": architecture,
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "n_hidden_states": n_hidden_states,
                "gemm_n": gemm_n,
                "variant": variant,
                "repetition": repetition,
                "latency_ms": latency_ms,
            }
        )


def _write_packet(results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings = pd.DataFrame(
        [row for result in results for row in result["timings"]]
    )
    correctness = pd.DataFrame(
        [row for result in results for row in result["correctness"]]
    )
    keys = [
        "architecture",
        "vocab_size",
        "hidden_size",
        "n_hidden_states",
        "gemm_n",
        "variant",
    ]
    case_summary = (
        timings.groupby(keys, as_index=False)
        .agg(
            repetitions=("latency_ms", "count"),
            median_ms=("latency_ms", "median"),
            p10_ms=("latency_ms", lambda values: values.quantile(0.1)),
            p90_ms=("latency_ms", lambda values: values.quantile(0.9)),
            std_ms=("latency_ms", "std"),
        )
    )
    config_keys = keys[:-1]
    cublas = (
        case_summary.query("variant == 'cublas'")[
            config_keys + ["median_ms"]
        ]
        .rename(columns={"median_ms": "cublas_median_ms"})
    )
    case_summary = case_summary.merge(
        cublas, on=config_keys, how="left", validate="many_to_one"
    )
    case_summary["latency_ratio"] = (
        case_summary["median_ms"] / case_summary["cublas_median_ms"]
    )
    candidates = case_summary.query("variant != 'cublas'").copy()
    candidates["pass"] = (
        candidates["latency_ratio"].le(MAXIMUM_RATIO).astype(int)
    )
    selected = (
        candidates.sort_values("latency_ratio")
        .groupby(config_keys, as_index=False)
        .first()
    )
    selected["pass"] = (
        selected["latency_ratio"].le(MAXIMUM_RATIO).astype(int)
    )

    timings.to_csv(OUTPUT_DIR / "cases.csv", index=False)
    case_summary.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    correctness.to_csv(OUTPUT_DIR / "correctness.csv", index=False)
    selected.to_csv(OUTPUT_DIR / "selected.csv", index=False)
    summary = {
        "gate": "ordinary-gemm-tuning",
        "command": "make modal-cutlass GATE=ordinary-gemm-tuning",
        "status": (
            "pass" if selected["pass"].eq(1).all() else "tuning-required"
        ),
        "variants": list(VARIANTS),
        "maximum_cutlass_to_cublas_ratio": MAXIMUM_RATIO,
        "warmup_repetitions": WARMUP_REPETITIONS,
        "benchmark_repetitions": BENCHMARK_REPETITIONS,
        "selected_passes": int(selected["pass"].sum()),
        "selected_cases": int(len(selected)),
        "worst_selected_ratio": float(selected["latency_ratio"].max()),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# Ordinary GEMM retained-candidate tuning

Review `summary.json`, then `selected.csv`, `case-summary.csv`,
`correctness.csv`, and the raw repetitions in `cases.csv`.

Every candidate uses matched BF16 inputs and outputs, preallocated buffers,
identical N padding, and cold-L2 timing.
The selected variant must remain within 5% of cuBLAS for every case before
any schedule is moved into the fused candidate epilogue.
"""
    )


@app.local_entrypoint()
def main() -> None:
    handles = [record_sm90.spawn(), record_sm100.spawn()]
    _write_packet([handle.get() for handle in handles])

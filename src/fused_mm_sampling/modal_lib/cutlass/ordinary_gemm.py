"""Dtype- and padding-matched ordinary CUTLASS versus cuBLAS GEMM gate."""

import json
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .ordinary_gemm_common import (
    HIDDEN_STATES,
    MAXIMUM_RATIO,
    MODEL_SHAPES,
    benchmark,
)
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(make_cutlass_provider_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/13-ordinary-gemm")
ARCHITECTURES = ("sm90", "sm100")
PROVIDERS = ("cutlass", "cublas")


@app.function(gpu="H100", image=image, timeout=60 * 60)
def record_sm90() -> dict:
    return _run("sm90")


@app.function(gpu="B200", image=image, timeout=60 * 60)
def record_sm100() -> dict:
    return _run("sm100")


def _run(architecture: str) -> dict:
    import torch

    from fused_mm_sampling.cutlass_impl import (
        cutlass_launch_plain_gemm,
        cutlass_launch_small_n_gemv,
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
            if n_hidden_states <= 2:
                padded = hidden_states
                gemm_n = n_hidden_states
                cutlass_output = torch.empty(
                    (vocab_size, n_hidden_states),
                    dtype=torch.bfloat16,
                    device="cuda",
                )

                def cutlass():
                    cutlass_launch_small_n_gemv(
                        weights, hidden_states, cutlass_output
                    )
            else:
                padded, cutlass_output, gemm_n = (
                    cutlass_make_plain_gemm_buffers(weights, hidden_states)
                )

                def cutlass():
                    cutlass_launch_plain_gemm(
                        weights, padded, cutlass_output
                    )
            cublas_output = torch.empty_like(cutlass_output)

            def cublas():
                torch.mm(weights, padded.T, out=cublas_output)

            cutlass()
            cublas()
            torch.cuda.synchronize()
            difference = (
                cutlass_output.float() - cublas_output.float()
            ).abs()
            correctness.append(
                {
                    "architecture": architecture,
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "n_hidden_states": n_hidden_states,
                    "gemm_n": gemm_n,
                    "dtype": str(cutlass_output.dtype),
                    "max_abs_difference": float(difference.max()),
                    "mean_abs_difference": float(difference.mean()),
                    "finite": int(
                        torch.isfinite(cutlass_output).all()
                        and torch.isfinite(cublas_output).all()
                    ),
                }
            )
            for provider, function in {
                "cutlass": cutlass,
                "cublas": cublas,
            }.items():
                for repetition, latency_ms in enumerate(
                    benchmark(function)
                ):
                    rows.append(
                        {
                            "architecture": architecture,
                            "vocab_size": vocab_size,
                            "hidden_size": hidden_size,
                            "n_hidden_states": n_hidden_states,
                            "gemm_n": gemm_n,
                            "provider": provider,
                            "repetition": repetition,
                            "latency_ms": latency_ms,
                        }
                    )
        del weights
    return {"timings": rows, "correctness": correctness}


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
        "provider",
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
        case_summary.query("provider == 'cublas'")[
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
    cutlass = case_summary.query("provider == 'cutlass'").copy()
    cutlass["pass"] = cutlass["latency_ratio"].le(MAXIMUM_RATIO).astype(int)
    status = "pass" if cutlass["pass"].eq(1).all() else "tuning-required"
    worst = cutlass.sort_values("latency_ratio", ascending=False).iloc[0]

    timings.to_csv(OUTPUT_DIR / "cases.csv", index=False)
    case_summary.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    correctness.to_csv(OUTPUT_DIR / "correctness.csv", index=False)
    summary = {
        "gate": "ordinary-gemm-prerequisite",
        "command": "make modal-cutlass GATE=ordinary-gemm",
        "status": status,
        "maximum_cutlass_to_cublas_ratio": MAXIMUM_RATIO,
        "architectures": list(ARCHITECTURES),
        "model_shapes": [list(shape) for shape in MODEL_SHAPES],
        "hidden_state_sweep": list(HIDDEN_STATES),
        "input_dtype": "torch.bfloat16",
        "output_dtype": "torch.bfloat16",
        "padding_policy": (
            "Logical N for the H=1 and H=2 specialization; N rounded up "
            "to a multiple of eight for both GEMM providers"
        ),
        "cold_l2": True,
        "worst_ratio": float(worst["latency_ratio"]),
        "worst_configuration": {
            "architecture": worst["architecture"],
            "vocab_size": int(worst["vocab_size"]),
            "hidden_size": int(worst["hidden_size"]),
            "n_hidden_states": int(worst["n_hidden_states"]),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# Ordinary GEMM prerequisite

Review `summary.json`, then `case-summary.csv` and `correctness.csv`.

CUTLASS and cuBLAS use the same BF16 weights, hidden states, logical problem
shape, preallocated BF16 output shape, and cold-L2 timing.
H=1 and H=2 use the specialized BF16 small-N kernel without padding.
The prerequisite passes only if CUTLASS is within 5% of cuBLAS everywhere.
If it does not pass, tune the ordinary CUTLASS schedules before modifying the
fused epilogue.
"""
    )


@app.local_entrypoint()
def main() -> None:
    handles = [record_sm90.spawn(), record_sm100.spawn()]
    _write_packet([handle.get() for handle in handles])

"""Gate 2b greedy CUTLASS performance feasibility decision."""

import json
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(make_cutlass_provider_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/10-greedy-performance")
ARCHITECTURES = ("sm90", "sm100")
MODEL_SHAPES = ((151_936, 4_096), (128_256, 8_192))
HIDDEN_STATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
PROVIDERS = (
    "cutlass-fmms",
    "cutlass-gemm",
    "cutlass-gemm-argmax",
    "triton-fmms",
    "cublas-argmax",
)
REFERENCE_PROVIDER = "cutlass-gemm-argmax"
FEASIBILITY_THRESHOLD = 1.05
WARMUP_REPETITIONS = 25
BENCHMARK_REPETITIONS = 100
TIMING_COLUMNS = [
    "architecture",
    "vocab_size",
    "hidden_size",
    "n_hidden_states",
    "provider",
    "repetition",
    "latency_ms",
]


@app.function(gpu="H100", image=image, timeout=60 * 60)
def record_sm90() -> dict:
    return _run("sm90")


@app.function(gpu="B200", image=image, timeout=60 * 60)
def record_sm100() -> dict:
    return _run("sm100")


def _run(architecture: str) -> dict:
    import torch

    from fused_mm_sampling.alg_names import ShortNames
    from fused_mm_sampling.core import get_sampler
    from fused_mm_sampling.cutlass_impl import (
        cutlass_greedy_kernel_attributes,
        cutlass_plain_gemm,
    )

    temperature = torch.empty((), device="cuda")
    rows = []
    correctness_rows = []
    for vocab_size, hidden_size in MODEL_SHAPES:
        weights = torch.randn(
            (vocab_size, hidden_size), dtype=torch.bfloat16, device="cuda"
        )
        for n_hidden_states in HIDDEN_STATES:
            hidden_states = torch.randn(
                (n_hidden_states, hidden_size),
                dtype=torch.bfloat16,
                device="cuda",
            )
            functions = _make_functions(
                weights,
                hidden_states,
                temperature,
                get_sampler,
                ShortNames,
                cutlass_plain_gemm,
            )
            reference = _indices(
                "cutlass-gemm", functions["cutlass-gemm"]()
            )
            for provider, function in functions.items():
                actual = _indices(provider, function())
                agrees_with_cutlass_gemm = torch.equal(actual, reference)
                passed = (
                    tuple(actual.shape) == (n_hidden_states, 1)
                    and actual.dtype == torch.int64
                    and bool(actual.ge(0).all())
                    and bool(actual.lt(vocab_size).all())
                )
                correctness_rows.append(
                    {
                        "architecture": architecture,
                        "vocab_size": vocab_size,
                        "hidden_size": hidden_size,
                        "n_hidden_states": n_hidden_states,
                        "provider": provider,
                        "agrees_with_cutlass_gemm": int(
                            agrees_with_cutlass_gemm
                        ),
                        "pass": int(passed),
                    }
                )
                if not passed:
                    raise RuntimeError(
                        "Invalid greedy performance baseline output for "
                        f"{architecture}, V={vocab_size}, D={hidden_size}, "
                        f"H={n_hidden_states}, provider={provider}"
                    )
                for repetition, latency_ms in enumerate(
                    _benchmark(function, WARMUP_REPETITIONS, BENCHMARK_REPETITIONS)
                ):
                    rows.append(
                        {
                            "architecture": architecture,
                            "vocab_size": vocab_size,
                            "hidden_size": hidden_size,
                            "n_hidden_states": n_hidden_states,
                            "provider": provider,
                            "repetition": repetition,
                            "latency_ms": latency_ms,
                        }
                    )
        del weights
    return {
        "timings": rows,
        "correctness": correctness_rows,
        "kernel_attributes": {
            "architecture": architecture,
            **cutlass_greedy_kernel_attributes(),
        },
    }


def _make_functions(
    weights,
    hidden_states,
    temperature,
    get_sampler,
    short_names,
    cutlass_plain_gemm,
):
    cutlass_sampler = get_sampler("fused-cutlass-greedy", weights=weights)
    triton_sampler = get_sampler(short_names.fused_triton_greedy, weights=weights)
    sample_kwargs = {
        "weights": weights,
        "hidden_states": hidden_states,
        "num_samples": 1,
        "temperature": temperature,
    }

    def cutlass_fmms():
        return cutlass_sampler.sample(**sample_kwargs)

    def cutlass_gemm():
        return cutlass_plain_gemm(weights, hidden_states)

    def cutlass_gemm_argmax():
        return cutlass_plain_gemm(weights, hidden_states).argmax(
            dim=0, keepdim=False
        )[:, None]

    def triton_fmms():
        return triton_sampler.sample(**sample_kwargs)

    def cublas_argmax():
        return (hidden_states @ weights.T).argmax(dim=-1, keepdim=True)

    return {
        "cutlass-fmms": cutlass_fmms,
        "cutlass-gemm": cutlass_gemm,
        "cutlass-gemm-argmax": cutlass_gemm_argmax,
        "triton-fmms": triton_fmms,
        "cublas-argmax": cublas_argmax,
    }


def _indices(provider, output):
    if provider == "cutlass-gemm":
        return output.argmax(dim=0, keepdim=False)[:, None]
    return output


def _benchmark(function, warmup_repetitions, benchmark_repetitions):
    import torch

    cache = torch.empty(
        256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda"
    )
    for _ in range(warmup_repetitions):
        function()
    torch.cuda.synchronize()
    start_events = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(benchmark_repetitions)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True)
        for _ in range(benchmark_repetitions)
    ]
    for start, end in zip(start_events, end_events):
        cache.zero_()
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [
        start.elapsed_time(end)
        for start, end in zip(start_events, end_events)
    ]


def summarize_timings(timings: pd.DataFrame) -> pd.DataFrame:
    """Build the compact per-configuration decision report."""
    keys = [
        "architecture",
        "vocab_size",
        "hidden_size",
        "n_hidden_states",
        "provider",
    ]
    summary = (
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
    reference = (
        summary.query("provider == @REFERENCE_PROVIDER")[
            config_keys + ["median_ms"]
        ]
        .rename(columns={"median_ms": "reference_median_ms"})
    )
    summary = summary.merge(
        reference, on=config_keys, how="left", validate="many_to_one"
    )
    summary["latency_ratio"] = (
        summary["median_ms"] / summary["reference_median_ms"]
    )
    summary["threshold"] = FEASIBILITY_THRESHOLD
    summary["threshold_applicable"] = summary["provider"].eq("cutlass-fmms")
    summary["pass"] = (
        ~summary["threshold_applicable"]
        | summary["latency_ratio"].le(summary["threshold"])
    ).astype(int)
    return summary


def _write_packet(results: list[dict]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timings = pd.DataFrame(
        [row for result in results for row in result["timings"]],
        columns=TIMING_COLUMNS,
    )
    correctness = pd.DataFrame(
        [row for result in results for row in result["correctness"]]
    )
    kernel_attributes = pd.DataFrame(
        [result["kernel_attributes"] for result in results]
    )
    _validate_coverage(timings, correctness)
    case_summary = summarize_timings(timings)
    fmms_rows = case_summary.query("provider == 'cutlass-fmms'")
    status = "pass" if fmms_rows["pass"].eq(1).all() else "no-go"
    worst = fmms_rows.sort_values("latency_ratio", ascending=False).iloc[0]

    timings.to_csv(OUTPUT_DIR / "cases.csv", index=False)
    case_summary.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    correctness.to_csv(OUTPUT_DIR / "correctness.csv", index=False)
    kernel_attributes.to_csv(
        OUTPUT_DIR / "kernel-attributes.csv", index=False
    )
    summary = {
        "gate": "2b",
        "command": "make modal-cutlass GATE=greedy-performance",
        "status": status,
        "threshold": {
            "metric": "median(cutlass-fmms) / median(cutlass-gemm-argmax)",
            "maximum": FEASIBILITY_THRESHOLD,
            "set_before_measurement": True,
        },
        "architectures": list(ARCHITECTURES),
        "providers": list(PROVIDERS),
        "model_shapes": [
            {"vocab_size": vocab_size, "hidden_size": hidden_size}
            for vocab_size, hidden_size in MODEL_SHAPES
        ],
        "hidden_state_sweep": list(HIDDEN_STATES),
        "warmup_repetitions": WARMUP_REPETITIONS,
        "benchmark_repetitions": BENCHMARK_REPETITIONS,
        "worst_shape": {
            "architecture": worst["architecture"],
            "vocab_size": int(worst["vocab_size"]),
            "hidden_size": int(worst["hidden_size"]),
            "n_hidden_states": int(worst["n_hidden_states"]),
            "latency_ratio": float(worst["latency_ratio"]),
            "pass": bool(worst["pass"]),
        },
        "correctness_pass": bool(correctness["pass"].eq(1).all()),
        "decision": (
            "Continue to Gate 3."
            if status == "pass"
            else "Stop before Gate 3 and rework or abandon the epilogue design."
        ),
        "profile_metrics": {
            "static_kernel_resources": True,
            "hbm_writes": False,
            "measured_occupancy": False,
            "component_durations": False,
            "reason": (
                "The timing gate records end-to-end feasibility and static "
                "kernel resources. NCU traffic and component profiling remain "
                "diagnostic follow-up if the epilogue is reworked; the "
                "predeclared end-to-end threshold already determines no-go."
            ),
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# Gate 2b greedy performance verification

Review `summary.json`, then `case-summary.csv`, `kernel-attributes.csv`, and
`correctness.csv`. Use `cases.csv` only to audit raw repetitions.

Expected:

- SM90 and SM100 each contain both primary model shapes and H=1 through H=256.
- Every provider has 100 raw repetitions for every configuration.
- Every provider returns one in-range int64 index per hidden state.
- Agreement with plain CUTLASS argmax is recorded but is not required for random near-ties.
- `cutlass-fmms / cutlass-gemm-argmax <= 1.05` for every configuration.
- The worst configuration and its decision are explicit in `summary.json`.

For a passing timing result, NCU must confirm HBM writes and measured occupancy
and a component profile must separate GEMM from Stage 2 before Gate 2b approval.
For a no-go result, those measurements are diagnostic follow-up and do not
override the predeclared end-to-end threshold.
"""
    )


def _validate_coverage(
    timings: pd.DataFrame, correctness: pd.DataFrame
) -> None:
    expected_configs = (
        len(ARCHITECTURES) * len(MODEL_SHAPES) * len(HIDDEN_STATES)
    )
    expected_timing_rows = (
        expected_configs * len(PROVIDERS) * BENCHMARK_REPETITIONS
    )
    if len(timings) != expected_timing_rows:
        raise RuntimeError(
            f"Expected {expected_timing_rows} timing rows, got {len(timings)}"
        )
    if set(timings["architecture"]) != set(ARCHITECTURES):
        raise RuntimeError("One or more architectures are absent")
    counts = timings.groupby(
        [
            "architecture",
            "vocab_size",
            "hidden_size",
            "n_hidden_states",
            "provider",
        ]
    ).size()
    if not counts.eq(BENCHMARK_REPETITIONS).all():
        raise RuntimeError("One or more configurations lack raw repetitions")
    if len(correctness) != expected_configs * len(PROVIDERS):
        raise RuntimeError("Correctness coverage is incomplete")
    if not correctness["pass"].eq(1).all():
        raise RuntimeError("One or more greedy baselines returned wrong indices")


@app.local_entrypoint()
def main() -> None:
    handles = [record_sm90.spawn(), record_sm100.spawn()]
    _write_packet([handle.get() for handle in handles])

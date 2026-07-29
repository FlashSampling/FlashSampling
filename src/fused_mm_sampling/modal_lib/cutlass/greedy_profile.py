"""Diagnostic component timings for the no-go CUTLASS greedy provider."""

import json
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(make_cutlass_provider_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/11-greedy-profile")
MODEL_SHAPES = ((151_936, 4_096), (128_256, 8_192))
HIDDEN_STATES = (1, 128)
WARMUP_REPETITIONS = 25
BENCHMARK_REPETITIONS = 100


@app.function(gpu="H100", image=image, timeout=60 * 60)
def record_sm90() -> dict:
    return _run("sm90")


@app.function(gpu="B200", image=image, timeout=60 * 60)
def record_sm100() -> dict:
    return _run("sm100")


def _run(architecture: str) -> dict:
    import torch

    from fused_mm_sampling.cutlass_impl import (
        cutlass_launch_greedy_gemm,
        cutlass_launch_greedy_stage2,
        cutlass_make_greedy_buffers,
        cutlass_plain_gemm,
        fused_mm_sample_cutlass_greedy,
    )

    rows = []
    correctness = []
    temperature = torch.empty((), device="cuda")
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
            buffers = cutlass_make_greedy_buffers(weights, hidden_states)
            padded, candidates, output, gemm_n, rounded_n, m_tiles = buffers

            def fused_gemm():
                cutlass_launch_greedy_gemm(
                    weights, padded, candidates, gemm_n, rounded_n
                )

            def stage2():
                cutlass_launch_greedy_stage2(
                    candidates,
                    output,
                    m_tiles,
                    rounded_n,
                    n_hidden_states,
                )

            def preallocated_pipeline():
                fused_gemm()
                stage2()
                return output

            def full_wrapper():
                return fused_mm_sample_cutlass_greedy(
                    weights,
                    hidden_states,
                    num_samples=1,
                    temperature=temperature,
                )

            def plain_gemm():
                return cutlass_plain_gemm(weights, hidden_states)

            fused_gemm()
            stage2()
            expected = full_wrapper()
            torch.cuda.synchronize()
            correctness.append(
                {
                    "architecture": architecture,
                    "vocab_size": vocab_size,
                    "hidden_size": hidden_size,
                    "n_hidden_states": n_hidden_states,
                    "pass": int(torch.equal(output, expected)),
                }
            )
            functions = {
                "fused-gemm-preallocated": fused_gemm,
                "stage2-preallocated": stage2,
                "pipeline-preallocated": preallocated_pipeline,
                "full-wrapper": full_wrapper,
                "plain-gemm-wrapper": plain_gemm,
            }
            for component, function in functions.items():
                latencies = _benchmark_cuda(function)
                rows.extend(
                    {
                        "architecture": architecture,
                        "vocab_size": vocab_size,
                        "hidden_size": hidden_size,
                        "n_hidden_states": n_hidden_states,
                        "component": component,
                        "repetition": repetition,
                        "latency_ms": latency,
                    }
                    for repetition, latency in enumerate(latencies)
                )
        del weights
    return {"timings": rows, "correctness": correctness}


def _benchmark_cuda(function) -> list[float]:
    from fused_mm_sampling.bench.triton_benchmark_lib import bench_cupti

    return bench_cupti(
        function,
        warmup_iters=WARMUP_REPETITIONS,
        rep_iters=BENCHMARK_REPETITIONS,
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
        "component",
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
    pivot = summary.pivot(
        index=keys[:-1], columns="component", values="median_ms"
    ).reset_index()
    pivot["wrapper_overhead_ms"] = (
        pivot["full-wrapper"] - pivot["pipeline-preallocated"]
    )
    pivot["epilogue_gemm_delta_ms"] = (
        pivot["fused-gemm-preallocated"] - pivot["plain-gemm-wrapper"]
    )
    pivot["stage2_share"] = (
        pivot["stage2-preallocated"] / pivot["pipeline-preallocated"]
    )

    timings.to_csv(OUTPUT_DIR / "cases.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    pivot.to_csv(OUTPUT_DIR / "diagnosis.csv", index=False)
    correctness.to_csv(OUTPUT_DIR / "correctness.csv", index=False)
    result_summary = {
        "status": (
            "pass" if correctness["pass"].eq(1).all() else "correctness-fail"
        ),
        "architectures": ["sm90", "sm100"],
        "model_shapes": [list(shape) for shape in MODEL_SHAPES],
        "hidden_state_sweep": list(HIDDEN_STATES),
        "warmup_repetitions": WARMUP_REPETITIONS,
        "benchmark_repetitions": BENCHMARK_REPETITIONS,
        "purpose": (
            "Separate fused GEMM, Stage 2, input preparation and allocation "
            "effects before collecting targeted hardware counters."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(result_summary, indent=2) + "\n"
    )
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# CUTLASS greedy component profile

Review `diagnosis.csv` first, then `case-summary.csv`.

The timings use the project's existing FlashInfer CUPTI benchmark helper with
cold L2 cache behavior.
The preallocated pipeline uses the same fused GEMM and Stage 2 kernels as the
production wrapper.
`wrapper_overhead_ms` is the full wrapper minus that preallocated pipeline.
`epilogue_gemm_delta_ms` compares the fused GEMM with the matching plain
CUTLASS GEMM wrapper.
`stage2_share` reports Stage 2 as a fraction of the preallocated pipeline.

These CUDA-event timings locate the expensive component.
They do not establish a hardware-level cause inside the fused GEMM.
Use NCU on the representative failing and passing points selected from this
packet before changing the kernel.
"""
    )


@app.local_entrypoint()
def main() -> None:
    handles = [record_sm90.spawn(), record_sm100.spawn()]
    _write_packet([handle.get() for handle in handles])

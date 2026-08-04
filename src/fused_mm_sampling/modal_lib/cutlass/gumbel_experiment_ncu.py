"""Shared matched NCU runner for CUTLASS sampling experiments and Triton."""

import io
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ...cutlass_experiments import get_cutlass_sampling_experiment
from ...dev_metrics import emit_dev_event
from ..utils import make_app, make_ncu_image, make_volumes, set_volume_caches
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(
    make_cutlass_provider_image(base_image=make_ncu_image(include_library_code=False))
).add_local_file(
    "benchmarking/cutlass_gumbel_profile_target.py",
    remote_path="/opt/fmms/cutlass_gumbel_profile_target.py",
    copy=False,
)

HIDDEN_STATES = (128, 256)
MODEL_SHAPES = {4_096: 151_936, 8_192: 128_256}
REQUIRED_METRICS = (
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
)
OPTIONAL_METRICS = (
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
    "smsp__sass_inst_executed_op_shfl.sum",
    "smsp__inst_executed.sum",
    "smsp__inst_executed_pipe_alu.sum",
    "smsp__inst_executed_pipe_fmaheavy.sum",
    "smsp__inst_executed_pipe_uniform.sum",
    "smsp__inst_executed_pipe_xu.sum",
    "smsp__inst_executed_pipe_xu.sum.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
    "smsp__sass_thread_inst_executed_op_conversion_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fp32_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_integer_pred_on.sum",
)


@app.function(gpu="B200", image=image, volumes=make_volumes(), timeout=60 * 60)
def record_sm100(hidden_size: int, variant: str) -> dict:
    set_volume_caches()
    emit_dev_event("remote_start", hidden_size=hidden_size, variant=variant)
    vocab_size = MODEL_SHAPES[hidden_size]
    query = subprocess.run(
        ["ncu", "--query-metrics", "--query-metrics-mode", "all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    metrics = [
        *REQUIRED_METRICS,
        *(metric for metric in OPTIONAL_METRICS if metric in query),
    ]
    rows = []
    components = (variant, "triton")
    for hidden_count in HIDDEN_STATES:
        for component in components:
            rows.extend(
                _profile(
                    component,
                    vocab_size,
                    hidden_size,
                    hidden_count,
                    metrics,
                )
            )
    return {"rows": rows, "metrics": metrics, "components": components}


def _profile(
    component: str,
    vocab_size: int,
    hidden_size: int,
    hidden_count: int,
    metrics: list[str],
) -> list[dict]:
    kernel_pattern = (
        "regex:.*fused_mm_sample_triton_kernel.*"
        if component == "triton"
        else "regex:.*device_kernel.*"
    )
    with tempfile.TemporaryDirectory() as directory:
        report_base = Path(directory) / "report"
        report_path = report_base.with_suffix(".ncu-rep")
        subprocess.run(
            [
                "ncu",
                "--metrics",
                ",".join(metrics),
                "--nvtx",
                "--nvtx-include",
                "profile/",
                "--kernel-name",
                kernel_pattern,
                "-f",
                "-o",
                str(report_base),
                "python",
                "/opt/fmms/cutlass_gumbel_profile_target.py",
                "--component",
                component,
                "--vocab-size",
                str(vocab_size),
                "--hidden-size",
                str(hidden_size),
                "--n-hidden-states",
                str(hidden_count),
            ],
            check=True,
        )
        exported = subprocess.run(
            ["ncu", "--import", str(report_path), "--csv", "--page", "raw"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    csv_lines = [line for line in exported.splitlines() if line.startswith('"')]
    frame = pd.read_csv(io.StringIO("\n".join(csv_lines)))
    units = frame.iloc[0]
    frame = frame.dropna(subset=["Kernel Name"]).copy()
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame.insert(0, "component", component)
    frame.insert(1, "vocab_size", vocab_size)
    frame.insert(2, "hidden_size", hidden_size)
    frame.insert(3, "n_hidden_states", hidden_count)
    frame["duration_unit"] = units["gpu__time_duration.sum"]
    frame["dram_read_unit"] = units["dram__bytes_read.sum"]
    frame["dram_write_unit"] = units["dram__bytes_write.sum"]
    return frame.to_dict(orient="records")


@app.local_entrypoint()
def main(hidden_size: int, variant: str, output_dir: str) -> None:
    if hidden_size not in MODEL_SHAPES:
        raise ValueError(f"hidden_size must be one of {tuple(MODEL_SHAPES)}")
    get_cutlass_sampling_experiment(variant)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = record_sm100.remote(hidden_size, variant)
    kernels = pd.DataFrame(result["rows"])
    output = f"ncu-d{hidden_size}-kernels.csv"
    kernels.to_csv(output_dir / output, index=False)
    summary = {
        "gate": f"4-{variant}-ncu",
        "status": "complete",
        "gpu": "B200",
        "model_shape": {
            "vocab_size": MODEL_SHAPES[hidden_size],
            "hidden_size": hidden_size,
        },
        "hidden_states": list(HIDDEN_STATES),
        "components": list(result["components"]),
        "metrics": result["metrics"],
        "output": output,
    }
    (output_dir / f"ncu-d{hidden_size}-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

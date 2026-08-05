"""Shared matched NCU runner for CUTLASS sampling experiments and Triton."""

import io
import json
import os
import subprocess
from pathlib import Path

import pandas as pd

from ...cutlass_experiments import (
    CUTLASS_PROFILE_CONFIG_MENU,
    get_cutlass_sampling_experiment,
)
from ...dev_metrics import emit_dev_event, timed_dev_stage
from ..utils import (
    commit_shared_volume,
    make_app,
    make_ncu_image,
    make_volumes,
    reload_shared_volume,
    set_volume_caches,
    volume_path,
)
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(
    make_cutlass_provider_image(base_image=make_ncu_image(include_library_code=False))
).add_local_file(
    "benchmarking/cutlass_gumbel_profile_target.py",
    remote_path="/opt/fmms/cutlass_gumbel_profile_target.py",
    copy=False,
)

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
def record_sm100(
    hidden_size: int, n_hidden_states: int, variant: str, run_id: str
) -> dict:
    reload_shared_volume()
    set_volume_caches()
    emit_dev_event(
        "remote_start",
        hidden_size=hidden_size,
        n_hidden_states=n_hidden_states,
        run_id=run_id,
        variant=variant,
    )
    vocab_size = MODEL_SHAPES[hidden_size]
    with timed_dev_stage("ncu_metric_query", hidden_size=hidden_size, variant=variant):
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
    report_dir = (
        Path(volume_path)
        / "cutlass-profiler"
        / "experiments"
        / variant
        / run_id
        / f"d{hidden_size}"
        / f"h{n_hidden_states}"
    )
    report_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    reports = []
    components = (variant, "triton")
    for component in components:
        with timed_dev_stage(
            "ncu_profile",
            accounting=False,
            component=component,
            hidden_size=hidden_size,
            n_hidden_states=n_hidden_states,
            variant=variant,
        ):
            profile_rows, report_path = _profile(
                component=component,
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                hidden_count=n_hidden_states,
                metrics=metrics,
                report_dir=report_dir,
            )
        rows.extend(profile_rows)
        reports.append(str(report_path))
    with timed_dev_stage("volume_commit", hidden_size=hidden_size, variant=variant):
        commit_shared_volume()
    return {
        "rows": rows,
        "metrics": metrics,
        "components": components,
        "reports": reports,
    }


def _profile(
    component: str,
    vocab_size: int,
    hidden_size: int,
    hidden_count: int,
    metrics: list[str],
    report_dir: Path,
) -> tuple[list[dict], Path]:
    kernel_pattern = (
        "regex:.*fused_mm_sample_triton_kernel.*"
        if component == "triton"
        else "regex:.*device_kernel.*"
    )
    report_base = report_dir / f"h{hidden_count}-{component}"
    report_path = report_base.with_suffix(".ncu-rep")
    emit_dev_event(
        "ncu_child_start",
        component=component,
        hidden_size=hidden_size,
        n_hidden_states=hidden_count,
        report_path=str(report_path),
    )
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
    with timed_dev_stage(
        "ncu_export",
        accounting=False,
        component=component,
        hidden_size=hidden_size,
        n_hidden_states=hidden_count,
    ):
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
    return frame.to_dict(orient="records"), report_path


@app.local_entrypoint()
def main(
    hidden_size: int, n_hidden_states: int, variant: str, output_dir: str
) -> None:
    if (hidden_size, n_hidden_states) not in CUTLASS_PROFILE_CONFIG_MENU:
        menu = ",".join(f"{d}:{h}" for d, h in CUTLASS_PROFILE_CONFIG_MENU)
        raise ValueError(
            f"Unknown profile config {hidden_size}:{n_hidden_states}; choose from {menu}"
        )
    get_cutlass_sampling_experiment(variant)
    run_id = os.environ.get("FMMS_DEV_RUN_ID")
    if not run_id:
        raise RuntimeError("FMMS_DEV_RUN_ID is required for durable NCU artifacts")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = record_sm100.remote(hidden_size, n_hidden_states, variant, run_id)
    kernels = pd.DataFrame(result["rows"])
    output = f"ncu-d{hidden_size}-h{n_hidden_states}-kernels.csv"
    kernels.to_csv(output_dir / output, index=False)
    summary = {
        "gate": f"4-{variant}-ncu",
        "status": "complete",
        "gpu": "B200",
        "model_shape": {
            "vocab_size": MODEL_SHAPES[hidden_size],
            "hidden_size": hidden_size,
        },
        "hidden_states": [n_hidden_states],
        "components": list(result["components"]),
        "metrics": result["metrics"],
        "output": output,
        "raw_reports": result["reports"],
        "run_id": run_id,
    }
    (output_dir / f"ncu-d{hidden_size}-h{n_hidden_states}-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

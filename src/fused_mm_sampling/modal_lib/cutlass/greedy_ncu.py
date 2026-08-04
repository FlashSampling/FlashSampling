"""Nsight Compute diagnostics for the current B200 CUTLASS donors."""

import io
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ..utils import make_app, make_ncu_image
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = (
    add_cutlass_greedy_provider(
        make_cutlass_provider_image(
            base_image=make_ncu_image(include_library_code=False)
        )
    )
    .add_local_file(
        "benchmarking/cutlass_greedy_profile_target.py",
        remote_path="/opt/fmms/cutlass_greedy_profile_target.py",
        copy=False,
    )
)

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/12-greedy-ncu")
MODEL_SHAPES = ((151_936, 4_096), (128_256, 8_192))
# Profile only the currently slow, actively changing production shape.
# Keep earlier H=128 packets as the control until that dispatch changes.
HIDDEN_STATES = (256,)
COMPONENTS = ("production-fused", "matching-plain")
REPORT_COLUMNS = [
    "Kernel Name",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "launch__waves_per_multiprocessor",
    "sm__maximum_warps_per_active_cycle_pct",
]
METRICS = REPORT_COLUMNS[1:]
DIAGNOSTIC_METRICS = (
    "l1tex__t_sectors_pipe_lsu_mem_global_op_atom.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "lts__t_sectors_op_atom.sum",
    "smsp__inst_executed_op_global_atom.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
)
BYTE_SCALES = {
    "byte": 1,
    "Kbyte": 10**3,
    "Mbyte": 10**6,
    "Gbyte": 10**9,
}
TIME_SCALES = {
    "ns": 1,
    "us": 10**3,
    "ms": 10**6,
    "s": 10**9,
}


@app.function(gpu="B200", image=image, timeout=60 * 60)
def record_sm100() -> dict:
    return _run()


def _run() -> dict:
    metrics = _available_metrics()
    rows = []
    for vocab_size, hidden_size in MODEL_SHAPES:
        for n_hidden_states in HIDDEN_STATES:
            for component in COMPONENTS:
                rows.extend(
                    _profile_component(
                        component,
                        vocab_size,
                        hidden_size,
                        n_hidden_states,
                        metrics,
                    )
                )
    return {"rows": rows, "metrics": metrics}


def _available_metrics() -> list[str]:
    query = subprocess.run(
        ["ncu", "--query-metrics", "--query-metrics-mode", "all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    optional = [metric for metric in DIAGNOSTIC_METRICS if metric in query]
    return [*METRICS, *optional]


def _profile_component(
    component: str,
    vocab_size: int,
    hidden_size: int,
    n_hidden_states: int,
    metrics: list[str],
) -> list[dict]:
    with tempfile.TemporaryDirectory() as directory:
        report_base = Path(directory) / "report"
        report_path = report_base.with_suffix(".ncu-rep")
        command = [
            "ncu",
            "--metrics",
            ",".join(metrics),
            "--nvtx",
            "--nvtx-include",
            "profile/",
            "--kernel-name",
            "regex:.*device_kernel.*",
            "--target-processes",
            "all",
            "--import-source",
            "yes",
            "-f",
            "-o",
            str(report_base),
            "python",
            "/opt/fmms/cutlass_greedy_profile_target.py",
            "--component",
            component,
            "--vocab-size",
            str(vocab_size),
            "--hidden-size",
            str(hidden_size),
            "--n-hidden-states",
            str(n_hidden_states),
        ]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, check=True)
        export = subprocess.run(
            [
                "ncu",
                "--import",
                str(report_path),
                "--csv",
                "--page",
                "raw",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    csv_lines = [
        line for line in export.stdout.splitlines() if line.startswith('"')
    ]
    frame = pd.read_csv(io.StringIO("\n".join(csv_lines)))
    units = frame.iloc[0]
    available = [
        column for column in ["Kernel Name", *metrics] if column in frame
    ]
    frame = frame.loc[:, available].dropna(subset=["Kernel Name"]).copy()
    for column in ("dram__bytes_read.sum", "dram__bytes_write.sum"):
        frame[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            * BYTE_SCALES[units[column]]
        )
    duration_column = "gpu__time_duration.sum"
    frame[duration_column] = (
        pd.to_numeric(frame[duration_column], errors="coerce")
        * TIME_SCALES[units[duration_column]]
    )
    for column in (
        "launch__registers_per_thread",
        "launch__waves_per_multiprocessor",
        "sm__maximum_warps_per_active_cycle_pct",
    ):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in metrics:
        if column not in REPORT_COLUMNS:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.rename(
        columns={
            "dram__bytes_read.sum": "dram_read_bytes",
            "dram__bytes_write.sum": "dram_write_bytes",
            "gpu__time_duration.sum": "gpu_duration_ns",
            "launch__registers_per_thread": "registers_per_thread",
            "launch__waves_per_multiprocessor": "waves_per_sm",
            "sm__maximum_warps_per_active_cycle_pct": "active_warps_pct",
        }
    )
    frame.insert(0, "architecture", "sm100")
    frame.insert(1, "component", component)
    frame.insert(2, "vocab_size", vocab_size)
    frame.insert(3, "hidden_size", hidden_size)
    frame.insert(4, "n_hidden_states", n_hidden_states)
    return frame.to_dict(orient="records")


def _write_packet(result: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kernels = pd.DataFrame(result["rows"])
    kernels.to_csv(OUTPUT_DIR / "kernels.csv", index=False)
    selected = kernels.assign(
        kernel_name=kernels["Kernel Name"].str.slice(0, 160)
    ).drop(columns=["Kernel Name"])
    selected.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    summary = {
        "status": "complete",
        "profiler": "Nsight Compute",
        "metrics": result["metrics"],
        "architectures": ["sm100"],
        "model_shapes": [
            {"vocab_size": vocab_size, "hidden_size": hidden_size}
            for vocab_size, hidden_size in MODEL_SHAPES
        ],
        "hidden_state_sweep": list(HIDDEN_STATES),
        "components": list(COMPONENTS),
        "nvtx_range": "profile",
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# CUTLASS greedy Nsight Compute profile

Review `case-summary.csv`.

Each row comes from the project's standard Nsight Compute image and raw CSV
export path.
Only the operation inside the `profile` NVTX range is captured.
The ranges invoke the exact production fused donor and matching plain Gate 2c
donor, including their wrapper kernels and allocations.
Identify the CUTLASS GEMMs by kernel name.

Compare the changed H=256 fused GEMM with its matching plain donor for both
primary B200 shapes.
Inspect atomic instructions and sectors, scheduler stalls, tensor-core
utilization, duration, DRAM traffic, registers, waves, and active warps.
"""
    )


@app.local_entrypoint()
def main() -> None:
    _write_packet(record_sm100.remote())

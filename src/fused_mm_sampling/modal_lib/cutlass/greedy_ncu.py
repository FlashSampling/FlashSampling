"""Nsight Compute diagnostics for representative CUTLASS greedy kernels."""

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
        make_cutlass_provider_image(base_image=make_ncu_image())
    )
    .add_local_file(
        "benchmarking/cutlass_greedy_profile_target.py",
        remote_path="/opt/fmms/cutlass_greedy_profile_target.py",
        copy=True,
    )
)

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/12-greedy-ncu")
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 4_096
HIDDEN_STATES = (1, 128)
COMPONENTS = ("fused-gemm", "stage2", "plain-gemm")
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


@app.function(gpu="H100", image=image, timeout=60 * 60)
def record_sm90() -> list[dict]:
    return _run("sm90")


@app.function(gpu="B200", image=image, timeout=60 * 60)
def record_sm100() -> list[dict]:
    return _run("sm100")


def _run(architecture: str) -> list[dict]:
    rows = []
    for n_hidden_states in HIDDEN_STATES:
        for component in COMPONENTS:
            rows.extend(
                _profile_component(
                    architecture, component, n_hidden_states
                )
            )
    return rows


def _profile_component(
    architecture: str, component: str, n_hidden_states: int
) -> list[dict]:
    with tempfile.TemporaryDirectory() as directory:
        report_base = Path(directory) / "report"
        report_path = report_base.with_suffix(".ncu-rep")
        command = [
            "ncu",
            "--metrics",
            ",".join(METRICS),
            "--nvtx",
            "--nvtx-include",
            "profile/",
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
            str(VOCAB_SIZE),
            "--hidden-size",
            str(HIDDEN_SIZE),
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
    available = [column for column in REPORT_COLUMNS if column in frame]
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
    frame.insert(0, "architecture", architecture)
    frame.insert(1, "component", component)
    frame.insert(2, "n_hidden_states", n_hidden_states)
    return frame.to_dict(orient="records")


def _write_packet(results: list[list[dict]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    kernels = pd.DataFrame(
        [row for architecture_rows in results for row in architecture_rows]
    )
    kernels.to_csv(OUTPUT_DIR / "kernels.csv", index=False)
    selected = kernels.assign(
        kernel_name=kernels["Kernel Name"].str.slice(0, 160)
    ).drop(columns=["Kernel Name"])
    selected.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    summary = {
        "status": "complete",
        "profiler": "Nsight Compute",
        "metrics": METRICS,
        "architectures": ["sm90", "sm100"],
        "vocab_size": VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
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
The target uses preallocated buffers for the fused GEMM and Stage 2.
The plain GEMM range may contain allocation-related CUDA kernels in addition
to its CUTLASS GEMM, so identify it by kernel name.

Compare fused and plain CUTLASS GEMM duration, DRAM traffic, registers, waves,
and achieved active warps at H=1 and H=128.
Compare Stage 2 duration and traffic against the complete component timing
packet in `../11-greedy-profile/`.
"""
    )


@app.local_entrypoint()
def main() -> None:
    handles = [record_sm90.spawn(), record_sm100.spawn()]
    _write_packet([handle.get() for handle in handles])

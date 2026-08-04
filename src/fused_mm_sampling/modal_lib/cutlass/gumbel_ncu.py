"""Matched NCU profiles for Gate 4 greedy and Gumbel-Max kernels."""

import io
import json
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ..utils import make_app, make_ncu_image, make_volumes, set_volume_caches
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(
    make_cutlass_provider_image(
        base_image=make_ncu_image(include_library_code=False)
    )
).add_local_file(
    "benchmarking/cutlass_gumbel_profile_target.py",
    remote_path="/opt/fmms/cutlass_gumbel_profile_target.py",
    copy=False,
)

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/18-gumbel-provider")
HIDDEN_STATES = (64, 128, 256)
COMPONENTS = ("greedy", "gumbel")
REQUIRED_METRICS = (
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
)
OPTIONAL_METRICS = (
    "smsp__inst_executed_pipe_xu.sum.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
)


@app.function(
    gpu="B200", image=image, volumes=make_volumes(), timeout=60 * 60
)
def record_sm100() -> dict:
    set_volume_caches()
    query = subprocess.run(
        ["ncu", "--query-metrics", "--query-metrics-mode", "all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    optional = [metric for metric in OPTIONAL_METRICS if metric in query]
    metrics = [*REQUIRED_METRICS, *optional]
    rows = []
    for hidden_count in HIDDEN_STATES:
        for component in COMPONENTS:
            rows.extend(_profile(component, hidden_count, metrics))
    return {"rows": rows, "metrics": metrics}


def _profile(component: str, hidden_count: int, metrics: list[str]) -> list[dict]:
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
                "regex:.*device_kernel.*",
                "-f",
                "-o",
                str(report_base),
                "python",
                "/opt/fmms/cutlass_gumbel_profile_target.py",
                "--component",
                component,
                "--vocab-size",
                "151936",
                "--hidden-size",
                "4096",
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
    frame.insert(1, "n_hidden_states", hidden_count)
    frame["duration_unit"] = units["gpu__time_duration.sum"]
    frame["dram_read_unit"] = units["dram__bytes_read.sum"]
    frame["dram_write_unit"] = units["dram__bytes_write.sum"]
    return frame.to_dict(orient="records")


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = record_sm100.remote()
    kernels = pd.DataFrame(result["rows"])
    kernels.to_csv(OUTPUT_DIR / "ncu-kernels.csv", index=False)
    summary = {
        "gate": "4-phase-3-ncu",
        "status": "complete",
        "gpu": "B200",
        "model_shape": {"vocab_size": 151_936, "hidden_size": 4_096},
        "hidden_states": list(HIDDEN_STATES),
        "components": list(COMPONENTS),
        "metrics": result["metrics"],
        "output": "ncu-kernels.csv",
    }
    (OUTPUT_DIR / "ncu-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

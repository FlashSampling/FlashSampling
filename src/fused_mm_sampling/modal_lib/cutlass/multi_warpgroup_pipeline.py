"""Validate the two-warpgroup SM100 epilogue pipelines and barriers."""

import json
import subprocess
from pathlib import Path

from ..utils import make_app
from .gate_common import sanitizer_pass
from .utils import add_cutlass_multi_warpgroup_pipeline, make_cutlass_image

app = make_app()
image = add_cutlass_multi_warpgroup_pipeline(make_cutlass_image())

OUTPUT_DIR = Path(
    "benchmarking/modal-results/cutlass/28-multi-warpgroup-pipeline"
)
EXECUTABLE = "/opt/fmms/cutlass_multi_warpgroup_pipeline"


@app.function(gpu="B200", image=image, timeout=10 * 60)
def verify() -> tuple[dict, dict[str, str]]:
    repeated = subprocess.run(
        [EXECUTABLE, "100"],
        check=True,
        capture_output=True,
        text=True,
    )
    repeated_summary = json.loads(repeated.stdout)
    reports = {}
    for tool in ("memcheck", "racecheck"):
        result = subprocess.run(
            [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                EXECUTABLE,
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        reports[tool] = result.stdout + result.stderr
        if result.returncode != 0:
            raise RuntimeError(
                f"{tool} failed with exit code {result.returncode}:\n"
                f"{reports[tool][-12_000:]}"
            )
    sanitizer_results = sanitizer_pass(reports)
    summary = {
        **repeated_summary,
        "repeated_runs": 100,
        "sanitizer_runs": 2,
        **sanitizer_results,
    }
    if not all(sanitizer_results.values()):
        raise RuntimeError(f"Sanitizer summary did not pass: {summary}")
    print(json.dumps(summary, indent=2))
    return summary, reports


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary, reports = verify.remote()
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    for tool, report in reports.items():
        (OUTPUT_DIR / f"{tool}.txt").write_text(report)

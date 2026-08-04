"""Gate 2d fused-EVT experiments for the winning B200 2-SM schedules."""

import subprocess
from io import StringIO

from ..utils import make_app
from .gate_common import csv_from_sanitizer_stdout, result_dir
from .utils import add_cutlass_winning_schedule_evt, make_cutlass_image

app = make_app()
image = add_cutlass_winning_schedule_evt(make_cutlass_image())

OUTPUT_DIR = result_dir("16-winning-schedule-evt")
EXECUTABLES = {
    "128x64x128-c2": "/opt/fmms/cutlass_winning_evt_128x64x128-c2",
    "256x128x64-c2": "/opt/fmms/cutlass_winning_evt_256x128x64-c2",
    "256x128x64-c4": "/opt/fmms/cutlass_winning_evt_256x128x64-c4",
    "256x128x128-c2": "/opt/fmms/cutlass_winning_evt_256x128x128-c2",
    "256x256x64-c2": "/opt/fmms/cutlass_winning_evt_256x256x64-c2",
}
EXPECTED_COLUMNS = [
    "architecture", "family", "case", "m", "n", "k", "row_type", "m_tile",
    "column", "tile_begin", "tile_end", "expected_value_bits",
    "actual_value_bits", "expected_index", "actual_index", "pass",
]
CSV_HEADER = ",".join(EXPECTED_COLUMNS)


@app.function(gpu="B200", image=image, timeout=15 * 60)
def record() -> dict:
    import pandas as pd

    reports = {}
    case_frames = []
    for family, executable in EXECUTABLES.items():
        for tool in ("memcheck", "racecheck"):
            result = subprocess.run(
                [
                    "compute-sanitizer", "--tool", tool,
                    "--error-exitcode", "99", executable,
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            reports[f"{tool}-{family}"] = result.stdout + result.stderr
            if result.returncode != 0:
                return {"returncode": result.returncode, **reports}
            if tool == "racecheck":
                csv_text = csv_from_sanitizer_stdout(result.stdout, CSV_HEADER)
                frame = pd.read_csv(StringIO(csv_text))
                frame.insert(1, "schedule", family)
                case_frames.append(frame)
    if not case_frames:
        raise RuntimeError("Fused-EVT CSV is absent")
    cases = pd.concat(case_frames, ignore_index=True)
    if not cases["pass"].eq(1).all():
        failures = cases.query("`pass` != 1")
        raise RuntimeError(f"{len(failures)} fused-EVT comparisons failed")
    final_cases = cases.query("row_type == 'final'")
    expected_rows = final_cases[["case", "n"]].drop_duplicates().rename(
        columns={"n": "expected"}
    )
    expected_rows["expected"] *= len(EXECUTABLES)
    actual_rows = final_cases.groupby("case", as_index=False).agg(
        actual=("column", "count")
    )
    coverage = expected_rows.merge(actual_rows, on="case", validate="one_to_one")
    if not coverage["actual"].eq(coverage["expected"]).all():
        raise RuntimeError("Incomplete fused-EVT candidate coverage")
    print(f"passed {len(cases)} exact candidate comparisons")
    return {"returncode": 0, "csv": cases.to_csv(index=False), **reports}


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = record.remote()
    for report, text in result.items():
        if report.startswith(("memcheck-", "racecheck-")):
            (OUTPUT_DIR / f"{report}-sm100.txt").write_text(text)
    if result["returncode"] != 0:
        raise RuntimeError(
            f"fused-EVT executable failed with {result['returncode']}; "
            f"see {OUTPUT_DIR}"
        )
    (OUTPUT_DIR / "cases.csv").write_text(result["csv"])

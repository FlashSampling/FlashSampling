"""Gate 2d fused-EVT experiment for the B200 128x64x128 2-SM schedule."""

import subprocess
from io import StringIO
from pathlib import Path

from ..utils import make_app
from .utils import add_cutlass_winning_schedule_evt, make_cutlass_image

app = make_app()
image = add_cutlass_winning_schedule_evt(make_cutlass_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/16-winning-schedule-evt")
EXECUTABLE = "/opt/fmms/cutlass_winning_evt_128x64x128_c2"
EXPECTED_COLUMNS = [
    "architecture", "family", "case", "m", "n", "k", "m_tile",
    "column", "tile_begin", "tile_end", "expected_value_bits",
    "actual_value_bits", "expected_index", "actual_index", "pass",
]


@app.function(gpu="B200", image=image, timeout=15 * 60)
def record() -> dict:
    import pandas as pd

    reports = {}
    csv_text = None
    for tool in ("memcheck", "racecheck"):
        result = subprocess.run(
            [
                "compute-sanitizer", "--tool", tool,
                "--error-exitcode", "99", EXECUTABLE,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        reports[tool] = result.stdout + result.stderr
        if result.returncode != 0:
            return {"returncode": result.returncode, **reports}
        if tool == "racecheck":
            csv_text = _extract_csv(result.stdout)
    if csv_text is None:
        raise RuntimeError("Fused-EVT CSV is absent")
    cases = pd.read_csv(StringIO(csv_text))
    if not cases["pass"].eq(1).all():
        failures = cases.query("`pass` != 1")
        raise RuntimeError(f"{len(failures)} fused-EVT comparisons failed")
    expected_rows = cases[["case", "n"]].drop_duplicates().rename(
        columns={"n": "expected"}
    )
    actual_rows = cases.groupby("case", as_index=False).agg(
        actual=("column", "count")
    )
    coverage = expected_rows.merge(actual_rows, on="case", validate="one_to_one")
    if not coverage["actual"].eq(coverage["expected"]).all():
        raise RuntimeError("Incomplete fused-EVT candidate coverage")
    print(f"passed {len(cases)} exact candidate comparisons")
    return {"returncode": 0, "csv": csv_text, **reports}


def _extract_csv(stdout: str) -> str:
    lines = stdout.splitlines()
    header = ",".join(EXPECTED_COLUMNS)
    csv_start = next(
        (position for position, line in enumerate(lines) if line == header),
        None,
    )
    if csv_start is None:
        raise RuntimeError("Fused-EVT CSV is absent from sanitizer output")
    csv_lines = []
    for line in lines[csv_start:]:
        if line.startswith("========="):
            break
        csv_lines.append(line)
    return "\n".join(csv_lines) + "\n"


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = record.remote()
    for tool in ("memcheck", "racecheck"):
        if tool in result:
            (OUTPUT_DIR / f"{tool}-sm100.txt").write_text(result[tool])
    if result["returncode"] != 0:
        raise RuntimeError(
            f"fused-EVT executable failed with {result['returncode']}; "
            f"see {OUTPUT_DIR}"
        )
    (OUTPUT_DIR / "cases.csv").write_text(result["csv"])

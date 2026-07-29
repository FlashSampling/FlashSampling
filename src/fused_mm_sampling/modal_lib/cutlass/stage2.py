"""Validate the GPU Stage 2 merge of real CUTLASS GEMM EVT candidates."""

import json
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .utils import add_cutlass_stage2, make_cutlass_image

app = make_app()
image = add_cutlass_stage2(make_cutlass_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/08-stage2")
ARCHITECTURES = ("sm90", "sm100")
FAMILIES = ("winner_tiles", "boundaries", "negative_ties", "cross_tile_ties")
EXPECTED_COLUMNS = [
    "architecture",
    "family",
    "case",
    "m",
    "n",
    "k",
    "row_type",
    "m_tile",
    "column",
    "tile_begin",
    "tile_end",
    "expected_value_bits",
    "actual_value_bits",
    "expected_index",
    "actual_index",
    "pass",
]


@app.function(gpu="H100", image=image, timeout=15 * 60)
def record_sm90() -> dict[str, str]:
    return _run("/opt/fmms/cutlass_stage2_sm90")


@app.function(gpu="B200", image=image, timeout=15 * 60)
def record_sm100() -> dict[str, str]:
    return _run("/opt/fmms/cutlass_stage2_sm100")


def _run(executable: str) -> dict[str, str]:
    reports = {}
    csv_text = None
    for tool in ("memcheck", "racecheck"):
        result = subprocess.run(
            [
                "compute-sanitizer",
                "--tool",
                tool,
                "--error-exitcode",
                "99",
                executable,
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
        if tool == "racecheck":
            csv_text = _extract_csv(result.stdout)
    if csv_text is None:
        raise RuntimeError("Stage 2 CSV is absent")
    return {"csv": csv_text, **reports}


def _extract_csv(stdout: str) -> str:
    lines = stdout.splitlines()
    header = ",".join(EXPECTED_COLUMNS)
    csv_start = next(
        (position for position, line in enumerate(lines) if line == header),
        None,
    )
    if csv_start is None:
        raise RuntimeError("Stage 2 CSV is absent from sanitizer output")
    csv_lines = []
    for line in lines[csv_start:]:
        if line.startswith("========="):
            break
        csv_lines.append(line)
    return "\n".join(csv_lines) + "\n"


def _validate(csv_text: str, architecture: str) -> pd.DataFrame:
    cases = pd.read_csv(StringIO(csv_text))
    if cases.columns.tolist() != EXPECTED_COLUMNS:
        raise RuntimeError(f"Unexpected columns: {cases.columns.tolist()}")
    if set(cases["architecture"]) != {architecture}:
        raise RuntimeError(f"Unexpected architecture rows for {architecture}")
    if set(cases["family"]) != set(FAMILIES):
        raise RuntimeError(f"Incomplete test families for {architecture}")
    if set(cases["row_type"]) != {"candidate", "final"}:
        raise RuntimeError(f"Intermediate or final rows are absent for {architecture}")
    if not cases["pass"].eq(1).all():
        failures = cases.query("`pass` != 1")
        raise RuntimeError(f"{len(failures)} Stage 2 comparisons failed")

    candidates = cases.query("row_type == 'candidate'")
    finals = cases.query("row_type == 'final'")
    expected_candidates = (
        candidates[["case", "m", "n"]]
        .drop_duplicates()
        .assign(expected=lambda frame: ((frame["m"] + 127) // 128) * frame["n"])
    )
    actual_candidates = (
        candidates.groupby("case", as_index=False)
        .agg(actual=("column", "count"))
    )
    coverage = expected_candidates.merge(
        actual_candidates, on="case", validate="one_to_one"
    )
    if not coverage["expected"].eq(coverage["actual"]).all():
        raise RuntimeError(f"Incomplete candidate coordinates for {architecture}")
    final_coverage = (
        finals.groupby("case", as_index=False)
        .agg(actual=("column", "count"), expected=("n", "first"))
    )
    if not final_coverage["expected"].eq(final_coverage["actual"]).all():
        raise RuntimeError(f"Incomplete final outputs for {architecture}")

    winner_cases = finals.query("family == 'winner_tiles'")[
        ["case", "actual_index"]
    ].drop_duplicates()
    expected_winners = {
        "winner_first_tile": 0,
        "winner_middle_tile": 128,
        "winner_last_tile": 256,
    }
    actual_winners = dict(
        zip(winner_cases["case"], winner_cases["actual_index"], strict=True)
    )
    if actual_winners != expected_winners:
        raise RuntimeError(f"Winner-tile coverage is incomplete: {actual_winners}")
    tie_indices = finals.query("family == 'cross_tile_ties'")["actual_index"]
    if tie_indices.empty or not tie_indices.eq(7).all():
        raise RuntimeError("Cross-tile ties did not select the lowest global index")
    return cases


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    sanitizer_pass = {}
    for architecture, remote_function in (
        ("sm90", record_sm90),
        ("sm100", record_sm100),
    ):
        result = remote_function.remote()
        frames.append(_validate(result["csv"], architecture))
        sanitizer_pass[architecture] = {
            "memcheck": "ERROR SUMMARY: 0 errors" in result["memcheck"],
            "racecheck": (
                "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
                in result["racecheck"]
            ),
        }
        for tool in ("memcheck", "racecheck"):
            (OUTPUT_DIR / f"{tool}-{architecture}.txt").write_text(result[tool])
        if not all(sanitizer_pass[architecture].values()):
            raise RuntimeError(f"Sanitizer did not pass for {architecture}")

    cases = pd.concat(frames, ignore_index=True)
    cases.to_csv(OUTPUT_DIR / "cases.csv", index=False)
    case_summary = (
        cases.groupby(
            ["architecture", "family", "case", "row_type"], as_index=False
        )
        .agg(
            m=("m", "first"),
            n=("n", "first"),
            row_count=("column", "count"),
            value_mismatch_count=(
                "expected_value_bits",
                lambda values: values.ne(
                    cases.loc[values.index, "actual_value_bits"]
                ).sum(),
            ),
            index_mismatch_count=(
                "expected_index",
                lambda values: values.ne(
                    cases.loc[values.index, "actual_index"]
                ).sum(),
            ),
            pass_status=("pass", "min"),
        )
        .rename(columns={"pass_status": "pass"})
    )
    case_summary.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    failures = cases.query("`pass` != 1")
    candidate_count = len(cases.query("row_type == 'candidate'"))
    final_count = len(cases.query("row_type == 'final'"))
    summary = {
        "gate": "1h",
        "command": "make modal-cutlass GATE=stage2",
        "expected_result": (
            "Every intermediate EVT candidate and GPU Stage 2 output matches "
            "the deterministic real-GEMM reference exactly."
        ),
        "actual_result": (
            f"All {candidate_count:,} candidates and {final_count:,} final "
            "outputs matched exactly."
        ),
        "status": "pass" if failures.empty else "fail",
        "architectures": list(ARCHITECTURES),
        "test_families": list(FAMILIES),
        "candidate_count": candidate_count,
        "final_output_count": final_count,
        "failure_count": len(failures),
        "final_reduction": True,
        "exact_fp32_bit_comparison": True,
        "sanitizer_pass": sanitizer_pass,
        "raw_measurements": {
            "applicable": False,
            "reason": "Gate 1h is a deterministic correctness gate.",
        },
        "statistics": {
            "applicable": False,
            "reason": "Gate 1h performs deterministic exact comparisons.",
        },
    }
    if summary["status"] != "pass":
        raise RuntimeError("Gate 1h verification packet is incomplete")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# Gate 1h verification

Expected:

- Both `sm90` and `sm100` are present.
- Every intermediate `(m_tile, column)` candidate matches exactly.
- Every final Stage 2 value bit pattern and global index matches exactly.
- Winners cover the first, middle, and last M tiles.
- Cross-tile ties choose the lowest global index.
- Memcheck and racecheck pass on both architectures.

Actual:

- `case-summary.csv` separates intermediate candidate and final rows.
- `cases.csv` retains the complete evidence used by Stage 2.
- `summary.json` records coverage and sanitizer status.

Inspect `case-summary.csv` first and require zero value and index mismatches.
Then inspect final rows for the three `winner_tiles` cases and the
`cross_tile_ties` case.
Search `log.txt` and sanitizer reports for errors, skips, and fallbacks.
"""
    )
    print(json.dumps(summary, indent=2))

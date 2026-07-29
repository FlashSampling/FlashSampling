"""Validate per-tile max-with-index candidates from a real CUTLASS GEMM EVT."""

import json
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .utils import add_cutlass_evt_candidates, make_cutlass_image

app = make_app()
image = add_cutlass_evt_candidates(make_cutlass_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/07-evt-candidates")
ARCHITECTURES = ("sm90", "sm100")
FAMILIES = ("tile_offsets", "boundaries", "negative_ties", "cross_tile_ties")
EXPECTED_COLUMNS = [
    "architecture",
    "family",
    "case",
    "m",
    "n",
    "k",
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
    return _run("/opt/fmms/cutlass_evt_candidates_sm90")


@app.function(gpu="B200", image=image, timeout=15 * 60)
def record_sm100() -> dict[str, str]:
    return _run("/opt/fmms/cutlass_evt_candidates_sm100")


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
        raise RuntimeError("EVT candidate CSV is absent")
    return {"csv": csv_text, **reports}


def _extract_csv(stdout: str) -> str:
    lines = stdout.splitlines()
    header = ",".join(EXPECTED_COLUMNS)
    csv_start = next(
        (position for position, line in enumerate(lines) if line == header),
        None,
    )
    if csv_start is None:
        raise RuntimeError("EVT candidate CSV is absent from sanitizer output")
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
    if not cases["pass"].eq(1).all():
        failures = cases.query("`pass` != 1")
        raise RuntimeError(f"{len(failures)} EVT candidate comparisons failed")

    expected_rows = (
        cases[["case", "m", "n"]]
        .drop_duplicates()
        .assign(m_tiles=lambda frame: (frame["m"] + 127) // 128)
        .assign(expected=lambda frame: frame["m_tiles"] * frame["n"])
    )
    actual_rows = (
        cases.groupby("case", as_index=False)
        .agg(actual=("column", "count"))
    )
    coverage = expected_rows.merge(actual_rows, on="case", validate="one_to_one")
    if not coverage["actual"].eq(coverage["expected"]).all():
        raise RuntimeError(f"Incomplete candidate coordinates for {architecture}")
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
        cases.assign(
            value_match=cases["expected_value_bits"].eq(
                cases["actual_value_bits"]
            ),
            index_match=cases["expected_index"].eq(cases["actual_index"]),
            nonzero_tile=cases["m_tile"].gt(0),
        )
        .groupby(["architecture", "family", "case"], as_index=False)
        .agg(
            m=("m", "first"),
            n=("n", "first"),
            k=("k", "first"),
            candidate_count=("column", "count"),
            m_tile_count=("m_tile", "nunique"),
            nonzero_tile_count=("nonzero_tile", "sum"),
            value_mismatch_count=("value_match", lambda values: values.ne(1).sum()),
            index_mismatch_count=("index_match", lambda values: values.ne(1).sum()),
            pass_status=("pass", "min"),
        )
        .rename(columns={"pass_status": "pass"})
    )
    case_summary.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    failures = cases.query("`pass` != 1")
    summary = {
        "gate": "1g",
        "command": "make modal-cutlass GATE=evt-candidates",
        "expected_result": (
            "Every packed FP32-value/i32-index candidate emitted by the "
            "CUTLASS EVT matches the corresponding real GEMM M-tile slice."
        ),
        "actual_result": (
            f"All {len(cases):,} per-tile candidates matched exactly across "
            f"{len(case_summary)} architecture-case combinations."
        ),
        "status": (
            "pass"
            if failures.empty
            and set(cases["architecture"]) == set(ARCHITECTURES)
            and set(cases["family"]) == set(FAMILIES)
            and all(
                all(tool_results.values())
                for tool_results in sanitizer_pass.values()
            )
            else "fail"
        ),
        "architectures": list(ARCHITECTURES),
        "test_families": list(FAMILIES),
        "candidate_count": len(cases),
        "failure_count": len(failures),
        "real_cutlass_gemm": True,
        "warp_specialized_cutlass_kernel": True,
        "final_reduction": False,
        "candidate_representation": "packed FP32 value and i32 global M index",
        "exact_fp32_bit_comparison": True,
        "sanitizer_pass": sanitizer_pass,
        "raw_measurements": {
            "applicable": False,
            "reason": "Gate 1g is a deterministic correctness gate.",
        },
        "statistics": {
            "applicable": False,
            "reason": "Gate 1g performs deterministic exact comparisons.",
        },
    }
    if summary["status"] != "pass":
        raise RuntimeError("Gate 1g verification packet is incomplete")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# Gate 1g verification

Expected:

- Both `sm90` and `sm100` are present.
- All four declared test families are present on both architectures.
- Every `(m_tile, column)` candidate is recorded, including losing tiles.
- Candidate value bits and global indices match exactly.
- Nonzero M-tile offsets, complete and partial M/N shapes, negative values,
  within-tile ties, and equal maxima in different tiles are covered.
- Memcheck reports zero errors on both architectures.
- Racecheck reports zero hazards, errors, and warnings on both architectures.

Actual:

- `summary.json` records overall coverage and sanitizer status.
- `case-summary.csv` contains one compact row per architecture and case.
- `cases.csv` retains every candidate before Stage 2.
- Four sanitizer reports retain the complete memory and race evidence.

Inspect `case-summary.csv` first.
Confirm zero value and index mismatches and nonzero tile coverage.
Then inspect all `cross_tile_ties` rows in `cases.csv`; equal values from
different tiles must retain their own lowest global index.
Search `log.txt` and all sanitizer reports for errors, exceptions, skipped
tests, NaNs, and fallbacks.
Gate 1g deliberately keeps `FinalReduction=false` and does not run Stage 2.
"""
    )
    print(json.dumps(summary, indent=2))

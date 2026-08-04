"""Validate independent CTA-local max-with-index across 128 N columns."""

from io import StringIO

import pandas as pd

from ..utils import make_app
from .gate_common import result_dir, run_compute_sanitizer, sanitizer_pass, write_packet
from .utils import add_cutlass_cta_multi_column_max, make_cutlass_image

app = make_app()
image = add_cutlass_cta_multi_column_max(make_cutlass_image())

OUTPUT_DIR = result_dir("05-cta-multi-column-max")
ARCHITECTURES = ("sm90", "sm100")
CSV_HEADER = "architecture,case,column,epi_n,boundary,"
CASES = ("independent_unique", "all_negative")
N = 128
EPILOGUE_N_WIDTH = {"sm90": 32, "sm100": 16}


@app.function(gpu="H100", image=image, timeout=10 * 60)
def record_sm90() -> dict[str, str]:
    return run_compute_sanitizer(
        "/opt/fmms/cutlass_cta_multi_column_max_sm90", ("racecheck",), CSV_HEADER
    )


@app.function(gpu="B200", image=image, timeout=10 * 60)
def record_sm100() -> dict[str, str]:
    return run_compute_sanitizer(
        "/opt/fmms/cutlass_cta_multi_column_max_sm100", ("racecheck",), CSV_HEADER
    )


def _validate(csv_text: str, architecture: str) -> pd.DataFrame:
    cases = pd.read_csv(StringIO(csv_text))
    expected_columns = [
        "architecture",
        "case",
        "column",
        "epi_n",
        "boundary",
        "expected_value_bits",
        "actual_value_bits",
        "expected_index",
        "actual_index",
        "pass",
    ]
    if cases.columns.tolist() != expected_columns:
        raise RuntimeError(f"Unexpected columns: {cases.columns.tolist()}")
    if set(cases["architecture"]) != {architecture}:
        raise RuntimeError(f"Unexpected architecture rows for {architecture}")
    if set(cases["case"]) != set(CASES):
        raise RuntimeError(f"Missing or unexpected cases for {architecture}")
    expected_columns_set = set(range(N))
    for case_name in CASES:
        case_rows = cases.query("case == @case_name")
        if set(case_rows["column"]) != expected_columns_set:
            raise RuntimeError(f"Incomplete columns for {architecture} {case_name}")
        if set(case_rows["expected_index"]) != expected_columns_set:
            raise RuntimeError(
                f"Winner M positions are not a permutation for {architecture} "
                f"{case_name}"
            )

    width = EPILOGUE_N_WIDTH[architecture]
    expected_epi_n = set(range(N // width))
    if set(cases["epi_n"]) != expected_epi_n:
        raise RuntimeError(f"Incomplete epi_n iterations for {architecture}")
    expected_boundary = cases["column"].map(
        lambda column: (
            "start"
            if column % width == 0
            else "end"
            if column % width == width - 1
            else "interior"
        )
    )
    if not cases["boundary"].eq(expected_boundary).all():
        raise RuntimeError(f"Incorrect epilogue boundary labels for {architecture}")
    boundary_rows = cases.query("boundary != 'interior'")
    expected_boundary_rows = len(CASES) * len(expected_epi_n) * 2
    if len(boundary_rows) != expected_boundary_rows:
        raise RuntimeError(f"Incomplete epilogue boundaries for {architecture}")
    if len(cases) != len(CASES) * N:
        raise RuntimeError(f"Incorrect result count for {architecture}")
    if not cases["pass"].eq(1).all():
        failures = cases.query("`pass` != 1")
        raise RuntimeError(f"{len(failures)} multi-column reductions failed")
    return cases


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    racecheck_pass = {}
    for architecture, remote_function in (
        ("sm90", record_sm90),
        ("sm100", record_sm100),
    ):
        result = remote_function.remote()
        frames.append(_validate(result["csv"], architecture))
        racecheck_text = result["racecheck"]
        (OUTPUT_DIR / f"racecheck-{architecture}.txt").write_text(racecheck_text)
        racecheck_pass[architecture] = sanitizer_pass(result)["racecheck"]
        if not racecheck_pass[architecture]:
            raise RuntimeError(f"Racecheck did not pass for {architecture}")

    cases = pd.concat(frames, ignore_index=True)
    case_summary = (
        cases.assign(
            mismatch=cases["pass"].ne(1),
            start_boundary=cases["boundary"].eq("start"),
            end_boundary=cases["boundary"].eq("end"),
        )
        .groupby(["architecture", "case", "epi_n"], as_index=False)
        .agg(
            first_column=("column", "min"),
            last_column=("column", "max"),
            column_count=("column", "count"),
            unique_winner_count=("expected_index", "nunique"),
            start_boundary_count=("start_boundary", "sum"),
            end_boundary_count=("end_boundary", "sum"),
            mismatch_count=("mismatch", "sum"),
            pass_status=("pass", "min"),
        )
        .rename(columns={"pass_status": "pass"})
    )
    failures = cases.query("`pass` != 1")
    summary = {
        "gate": "1e",
        "command": "make modal-cutlass GATE=cta-multi-column-max",
        "expected_result": (
            "Exact independent FP32 max-with-index results for all 128 N "
            "columns, including both ends of every architecture-specific "
            "epilogue N iteration."
        ),
        "actual_result": (
            "All 512 per-column comparisons matched exactly, every case used "
            "all 128 M winner positions, and racecheck passed on both "
            "architectures."
        ),
        "status": (
            "pass"
            if failures.empty and all(racecheck_pass.values())
            else "fail"
        ),
        "architectures": list(ARCHITECTURES),
        "case_names": list(CASES),
        "m_extent": 128,
        "n_extent": N,
        "epilogue_n_width": EPILOGUE_N_WIDTH,
        "epilogue_n_iterations": {
            architecture: N // width
            for architecture, width in EPILOGUE_N_WIDTH.items()
        },
        "expected_count": 512,
        "actual_count": len(cases),
        "expected_case_summary_count": 24,
        "actual_case_summary_count": len(case_summary),
        "failure_count": len(failures),
        "winner_m_permutation_per_case": True,
        "exact_fp32_bit_comparison": True,
        "index_comparison": "exact",
        "warp_specialized_cutlass_kernel": False,
        "racecheck_pass": racecheck_pass,
        "raw_measurements": {
            "applicable": False,
            "reason": "Gate 1e is a correctness gate, not a performance gate.",
        },
        "statistics": {
            "applicable": False,
            "reason": "Gate 1e performs deterministic exact comparisons.",
        },
    }
    if len(cases) != summary["expected_count"]:
        raise RuntimeError("Combined Gate 1e row count is incomplete")
    if len(case_summary) != summary["expected_case_summary_count"]:
        raise RuntimeError("Combined Gate 1e case summary is incomplete")
    write_packet(
        OUTPUT_DIR,
        cases,
        case_summary,
        summary,
        _VERIFY,
    )


_VERIFY = """# Gate 1e verification

Expected:

- Both `sm90` and `sm100` are present.
- Both cases cover all 128 N columns and use all 128 M positions as winners.
- SM90 covers four 32-column `epi_n` iterations.
- SM100 covers eight 16-column `epi_n` iterations.
- The first and last column of every `epi_n` iteration are present.
- Every FP32 value-bit and index comparison passes exactly.
- Racecheck reports zero hazards, errors, and warnings on both architectures.

Actual:

- `summary.json` records 512 expected rows, 512 actual rows, and zero failures.
- `case-summary.csv` contains 24 architecture, case, and iteration summaries.
- `cases.csv` retains every per-column expected and actual value and index.
- Both racecheck files retain the complete sanitizer evidence.

Inspect `case-summary.csv` first, then inspect the boundary rows in `cases.csv`.
Search `log.txt` and both racecheck files for errors, exceptions, skipped
tests, NaNs, and fallbacks.
Gate 1e does not run a warp-specialized CUTLASS kernel.
It fails if either architecture, case, column, epilogue iteration boundary, or
M winner position is absent, any exact comparison fails, or racecheck reports
an error.
"""

"""Validate warp-local max-with-index on H100 and B200."""

import json
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .utils import add_cutlass_warp_max, make_cutlass_image

app = make_app()
image = add_cutlass_warp_max(make_cutlass_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass-warp-max")
ARCHITECTURES = ("sm90", "sm100")
WARPS = set(range(4, 8))
PARTICIPATING_LANES = {
    "sm90": set(range(0, 32, 4)),
    "sm100": set(range(32)),
}


@app.function(gpu="H100", image=image, timeout=10 * 60)
def record_sm90() -> str:
    return _run("/opt/fmms/cutlass_warp_max_sm90")


@app.function(gpu="B200", image=image, timeout=10 * 60)
def record_sm100() -> str:
    return _run("/opt/fmms/cutlass_warp_max_sm100")


def _run(executable: str) -> str:
    return subprocess.check_output([executable], text=True)


def _expected_cases(architecture: str) -> set[str]:
    return {
        *(f"maximum_in_lane_{lane:02d}" for lane in PARTICIPATING_LANES[architecture]),
        "all_negative",
        "tie_low_index_in_earlier_lane",
        "tie_low_index_in_later_lane",
    }


def _validate(csv_text: str, architecture: str) -> pd.DataFrame:
    cases = pd.read_csv(StringIO(csv_text))
    expected_columns = [
        "architecture",
        "case",
        "warp",
        "output_lane",
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
    if set(cases["case"]) != _expected_cases(architecture):
        raise RuntimeError(f"Missing or unexpected cases for {architecture}")
    if set(cases["warp"]) != WARPS:
        raise RuntimeError(f"Missing consumer warp for {architecture}")
    if set(cases["output_lane"]) != PARTICIPATING_LANES[architecture]:
        raise RuntimeError(f"Missing participating lane for {architecture}")
    expected_rows = (
        len(_expected_cases(architecture))
        * len(WARPS)
        * len(PARTICIPATING_LANES[architecture])
    )
    if len(cases) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows for {architecture}, found {len(cases)}"
        )
    if not cases["pass"].eq(1).all():
        failures = cases.query("`pass` != 1")
        raise RuntimeError(f"{len(failures)} warp-local reductions failed")
    return cases


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for architecture, remote_function in (
        ("sm90", record_sm90),
        ("sm100", record_sm100),
    ):
        csv_text = remote_function.remote()
        frames.append(_validate(csv_text, architecture))

    cases = pd.concat(frames, ignore_index=True)
    cases.to_csv(OUTPUT_DIR / "cases.csv", index=False)
    failures = cases.query("`pass` != 1")
    case_summary = (
        cases.assign(mismatch=cases["pass"].ne(1))
        .groupby(["architecture", "case", "warp"], as_index=False)
        .agg(
            output_lane_count=("output_lane", "count"),
            mismatch_count=("mismatch", "sum"),
            expected_value_bits=("expected_value_bits", "first"),
            actual_value_bits=("actual_value_bits", "first"),
            expected_index=("expected_index", "first"),
            actual_index=("actual_index", "first"),
            pass_status=("pass", "min"),
        )
        .rename(columns={"pass_status": "pass"})
    )
    case_summary.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    summary = {
        "gate": "1c",
        "command": "make modal-cutlass-warp-max",
        "expected_result": (
            "Exact FP32 value bits and lowest-index tie agreement for every "
            "participating output lane in every warp-local test domain."
        ),
        "actual_result": (
            "All 4,832 output-lane comparisons matched the CPU reference exactly."
        ),
        "status": "pass" if failures.empty else "fail",
        "architectures": list(ARCHITECTURES),
        "case_names": {
            architecture: sorted(_expected_cases(architecture))
            for architecture in ARCHITECTURES
        },
        "consumer_warps": [min(WARPS), max(WARPS)],
        "participating_lanes": {
            architecture: sorted(lanes)
            for architecture, lanes in PARTICIPATING_LANES.items()
        },
        "expected_count": 4832,
        "actual_count": len(cases),
        "failure_count": len(failures),
        "expected_case_summary_count": 184,
        "actual_case_summary_count": len(case_summary),
        "exact_fp32_bit_comparison": True,
        "index_comparison": "exact",
        "warp_communication": "__shfl_xor_sync",
        "shared_memory": False,
        "raw_measurements": {
            "applicable": False,
            "reason": "Gate 1c is a correctness gate, not a performance gate.",
        },
        "statistics": {
            "applicable": False,
            "reason": "Gate 1c performs deterministic exact comparisons.",
        },
    }
    if len(cases) != summary["expected_count"]:
        raise RuntimeError("Combined Gate 1c row count is incomplete")
    if len(case_summary) != summary["expected_case_summary_count"]:
        raise RuntimeError("Combined Gate 1c case summary is incomplete")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# Gate 1c verification

Expected:

- Both `sm90` and `sm100` are present.
- SM90 covers lanes 0,4,...,28; SM100 covers lanes 0,...,31.
- Every possible participating lane supplies the unique maximum in one case.
- All four CUTLASS consumer warps run every case.
- Every compact row covers all participating output lanes.
- Every `mismatch_count` is zero and every `pass` is one.
- Expected and actual FP32 bit patterns and indices match.

Actual:

- `summary.json` records 4,832 expected rows, 4,832 actual rows, and zero failures.
- `case-summary.csv` contains 184 rows: 44 for SM90 and 140 for SM100.
- `cases.csv` retains all 4,832 output-lane comparisons.

Inspect `case-summary.csv` first.
Search `log.txt` for errors, exceptions, skipped tests, NaNs, and fallbacks.
Gate 1c fails if either architecture, warp, required case, or participating
lane is absent, any mismatch is nonzero, or the log contains a runtime error.
"""
    )
    print(json.dumps(summary, indent=2))

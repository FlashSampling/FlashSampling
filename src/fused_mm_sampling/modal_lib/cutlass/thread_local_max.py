"""Validate thread-local max-with-index on H100 and B200."""

import json
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .utils import add_cutlass_thread_local_max, make_cutlass_image

app = make_app()
image = add_cutlass_thread_local_max(make_cutlass_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass-thread-local-max")
ARCHITECTURES = ("sm90", "sm100")
EXPECTED_CASES = {
    *(f"maximum_in_slot_{slot:02d}" for slot in range(16)),
    "all_negative",
    "tie_low_index_in_earlier_slot",
    "tie_low_index_in_later_slot",
}
EXPECTED_ORDERS = {"ascending", "descending"}
EXPECTED_THREADS = set(range(128, 256))


@app.function(gpu="H100", image=image, timeout=10 * 60)
def record_sm90() -> str:
    return _run("/opt/fmms/cutlass_thread_local_max_sm90")


@app.function(gpu="B200", image=image, timeout=10 * 60)
def record_sm100() -> str:
    return _run("/opt/fmms/cutlass_thread_local_max_sm100")


def _run(executable: str) -> str:
    return subprocess.check_output([executable], text=True)


def _validate(csv_text: str, architecture: str) -> pd.DataFrame:
    cases = pd.read_csv(StringIO(csv_text))
    expected_columns = [
        "architecture",
        "case",
        "visit_order",
        "thread",
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
    if set(cases["case"]) != EXPECTED_CASES:
        raise RuntimeError(f"Missing or unexpected cases for {architecture}")
    if set(cases["visit_order"]) != EXPECTED_ORDERS:
        raise RuntimeError(f"Missing visit order for {architecture}")
    if set(cases["thread"]) != EXPECTED_THREADS:
        raise RuntimeError(f"Missing consumer thread for {architecture}")
    expected_rows = len(EXPECTED_CASES) * len(EXPECTED_ORDERS) * len(EXPECTED_THREADS)
    if len(cases) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows for {architecture}, found {len(cases)}"
        )
    if not cases["pass"].eq(1).all():
        failures = cases.query("`pass` != 1")
        raise RuntimeError(f"{len(failures)} thread-local reductions failed")
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
        .groupby(["architecture", "case", "visit_order"], as_index=False)
        .agg(
            thread_count=("thread", "count"),
            mismatch_count=("mismatch", "sum"),
            expected_value_bits=("expected_value_bits", "first"),
            actual_value_bits=("actual_value_bits", "first"),
            expected_index=("expected_index", "first"),
            actual_index=("actual_index", "first"),
        )
    )
    case_summary.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    summary = {
        "gate": "1b",
        "architectures": list(ARCHITECTURES),
        "case_names": sorted(EXPECTED_CASES),
        "visit_orders": sorted(EXPECTED_ORDERS),
        "consumer_threads": [min(EXPECTED_THREADS), max(EXPECTED_THREADS)],
        "fragment_slots": [0, 15],
        "expected_count": len(cases),
        "actual_count": len(cases),
        "failure_count": len(failures),
        "exact_fp32_bit_comparison": True,
        "index_comparison": "exact",
        "warp_communication": False,
        "shared_memory": False,
        "raw_measurements": {
            "applicable": False,
            "reason": "Gate 1b is a correctness gate, not a performance gate.",
        },
        "statistics": {
            "applicable": False,
            "reason": "Gate 1b performs deterministic exact comparisons.",
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# Gate 1b verification

Expected:

- Both `sm90` and `sm100` are present.
- All 19 cases run in ascending and descending visitation order.
- Every case/order row covers 128 CUTLASS consumer threads.
- Every `mismatch_count` is zero.
- Expected and actual FP32 bit patterns and indices match.

Actual:

- `summary.json` records 9,728 expected rows, 9,728 actual rows, and zero failures.
- `case-summary.csv` contains 76 rows: 2 architectures x 19 cases x 2 orders.
- `cases.csv` retains all 9,728 thread-level comparisons.

Inspect `case-summary.csv` first.
Search `log.txt` for errors, exceptions, skipped tests, NaNs, and fallbacks.
Gate 1b fails if either architecture or any case is absent, any row has fewer
than 128 threads, any mismatch is nonzero, or the log contains a runtime error.
"""
    )
    print(json.dumps(summary, indent=2))

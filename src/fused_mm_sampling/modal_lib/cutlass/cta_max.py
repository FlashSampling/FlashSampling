"""Validate CTA-local max-with-index on H100 and B200."""

import json
import subprocess
from io import StringIO
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .utils import add_cutlass_cta_max, make_cutlass_image

app = make_app()
image = add_cutlass_cta_max(make_cutlass_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/04-cta-max")
ARCHITECTURES = ("sm90", "sm100")
EXPECTED_CASES = {
    *(f"maximum_in_warp_{warp}" for warp in range(4, 8)),
    "all_negative",
    "tie_low_index_in_earlier_warp",
    "tie_low_index_in_later_warp",
}


@app.function(gpu="H100", image=image, timeout=10 * 60)
def record_sm90() -> dict[str, str]:
    return _run("/opt/fmms/cutlass_cta_max_sm90")


@app.function(gpu="B200", image=image, timeout=10 * 60)
def record_sm100() -> dict[str, str]:
    return _run("/opt/fmms/cutlass_cta_max_sm100")


def _run(executable: str) -> dict[str, str]:
    result = subprocess.run(
        [
            "compute-sanitizer",
            "--tool",
            "racecheck",
            "--error-exitcode",
            "99",
            executable,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    lines = result.stdout.splitlines()
    header = "architecture,case,expected_value_bits,actual_value_bits,"
    csv_start = next(
        (
            position
            for position, line in enumerate(lines)
            if line.startswith(header)
        ),
        None,
    )
    if csv_start is None:
        raise RuntimeError("CTA result CSV is absent from racecheck output")
    csv_lines = []
    for line in lines[csv_start:]:
        if line.startswith("========="):
            break
        csv_lines.append(line)
    return {"csv": "\n".join(csv_lines) + "\n", "racecheck": output}


def _validate(csv_text: str, architecture: str) -> pd.DataFrame:
    cases = pd.read_csv(StringIO(csv_text))
    expected_columns = [
        "architecture",
        "case",
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
    if len(cases) != len(EXPECTED_CASES):
        raise RuntimeError(f"Expected one result per case for {architecture}")
    if not cases["pass"].eq(1).all():
        failures = cases.query("`pass` != 1")
        raise RuntimeError(f"{len(failures)} CTA-local reductions failed")
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
        racecheck_pass[architecture] = (
            "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
            in racecheck_text
        )
        if not racecheck_pass[architecture]:
            raise RuntimeError(f"Racecheck did not report zero errors for {architecture}")

    cases = pd.concat(frames, ignore_index=True)
    cases.to_csv(OUTPUT_DIR / "cases.csv", index=False)
    case_summary = cases.assign(
        value_bits_match=cases["expected_value_bits"].eq(cases["actual_value_bits"]),
        index_match=cases["expected_index"].eq(cases["actual_index"]),
    )
    case_summary.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    failures = cases.query("`pass` != 1")
    summary = {
        "gate": "1d",
        "command": "make modal-cutlass GATE=cta-max",
        "expected_result": (
            "Exact FP32 value bits and lowest-index tie agreement across all "
            "four simulated CUTLASS consumer-warp roles, with zero racecheck errors."
        ),
        "actual_result": (
            "All 14 full-CTA comparisons matched the CPU reference exactly, "
            "and racecheck reported zero errors on both architectures."
        ),
        "status": (
            "pass"
            if failures.empty and all(racecheck_pass.values())
            else "fail"
        ),
        "architectures": list(ARCHITECTURES),
        "case_names": sorted(EXPECTED_CASES),
        "simulated_cutlass_consumer_warps": [4, 7],
        "actual_harness_warps": [0, 3],
        "warp_specialized_cutlass_kernel": False,
        "n_columns": 1,
        "complete_m_tile": 128,
        "expected_count": 14,
        "actual_count": len(cases),
        "failure_count": len(failures),
        "exact_fp32_bit_comparison": True,
        "index_comparison": "exact",
        "warp_communication": "__shfl_xor_sync",
        "cross_warp_communication": "shared memory",
        "racecheck_pass": racecheck_pass,
        "raw_measurements": {
            "applicable": False,
            "reason": "Gate 1d is a correctness gate, not a performance gate.",
        },
        "statistics": {
            "applicable": False,
            "reason": "Gate 1d performs deterministic exact comparisons.",
        },
    }
    if len(cases) != summary["expected_count"]:
        raise RuntimeError("Combined Gate 1d row count is incomplete")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# Gate 1d verification

Expected:

- Both `sm90` and `sm100` are present.
- Harness warps 0 through 3 simulate CUTLASS consumer-warp roles 4 through 7.
- Each simulated consumer-warp role supplies the unique winner once.
- Both inter-warp tie orders select the lower global index.
- The all-negative case selects the least-negative value.
- Every value-bit and index comparison passes exactly.
- Racecheck reports zero hazards, errors, and warnings on both architectures.

Actual:

- `summary.json` records 14 expected rows, 14 actual rows, and zero failures.
- `case-summary.csv` contains one compact row for every architecture and case.
- `racecheck-sm90.txt` and `racecheck-sm100.txt` retain sanitizer evidence.

Inspect `case-summary.csv` first.
Search `log.txt` and both racecheck files for errors, exceptions, skipped
tests, NaNs, and fallbacks.
Gate 1d does not run a warp-specialized CUTLASS kernel.
Gate 1d fails if either architecture, simulated contributing warp role, tie
order, or all-negative case is absent, any exact comparison fails, or
racecheck reports an error.
"""
    )
    print(json.dumps(summary, indent=2))

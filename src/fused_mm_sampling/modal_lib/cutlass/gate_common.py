"""Shared helpers for the CUTLASS correctness-gate runners.

These are the stable, verbatim-repeated boilerplate that the max-with-index,
EVT-candidate, and Stage 2 gate runners used to inline. Gate-specific
validation and orchestration stays in each runner; only the concrete,
unchanging bits live here.
"""

import json
import os
import subprocess
from pathlib import Path

import pandas as pd


def result_dir(slug: str) -> Path:
    """The standard cutlass result directory for `slug`.

    Appends `CUTLASS_RESULT_POSTFIX` (the Makefile `POSTFIX`) when set, so
    parallel or verification runs can write to a distinct packet instead of
    colliding with one that names the same directory.
    """
    postfix = os.environ.get("CUTLASS_RESULT_POSTFIX", "")
    return Path("benchmarking/modal-results/cutlass") / f"{slug}{postfix}"


def csv_from_sanitizer_stdout(
    stdout: str,
    header: str,
    error_msg: str = "result CSV is absent",
    *,
    exact: bool = False,
) -> str:
    """Extract the CSV block a gate executable prints under compute-sanitizer.

    The executable prints `architecture,...,pass` rows to stdout after the
    sanitizer header lines; the block ends at the first `=========` summary
    separator. `exact=True` matches only a line equal to `header` (used by the
    boundary gate), otherwise any line starting with `header`.
    """
    lines = stdout.splitlines()
    predicate = (lambda line: line == header) if exact else (
        lambda line: line.startswith(header)
    )
    csv_start = next(
        (position for position, line in enumerate(lines) if predicate(line)),
        None,
    )
    if csv_start is None:
        raise RuntimeError(f"{error_msg} from sanitizer output")
    csv_lines = []
    for line in lines[csv_start:]:
        if line.startswith("========="):
            break
        csv_lines.append(line)
    return "\n".join(csv_lines) + "\n"


def run_compute_sanitizer(
    executable: str,
    tools: tuple[str, ...],
    csv_header: str,
    *,
    csv_from: str = "racecheck",
    error_msg: str = "result CSV is absent",
    exact_csv_header: bool = False,
) -> dict[str, str]:
    """Run a gate executable under one or more compute-sanitizer tools.

    Returns `{"csv": <extracted CSV>, <tool>: <combined stdout+stderr>, ...}`
    for every tool in `tools`. A nonzero sanitizer exit aborts with a tailed
    error. `csv` is extracted from the `csv_from` tool's stdout.
    """
    reports: dict[str, str] = {}
    csv_text = None
    for tool in tools:
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
        if tool == csv_from:
            csv_text = csv_from_sanitizer_stdout(
                result.stdout,
                csv_header,
                error_msg,
                exact=exact_csv_header,
            )
    if csv_text is None:
        raise RuntimeError(f"CSV requested from a tool that did not run: {csv_from}")
    return {"csv": csv_text, **reports}


def sanitizer_pass(reports: dict[str, str]) -> dict[str, bool]:
    """Whether each present sanitizer report reports zero problems."""
    flags: dict[str, bool] = {}
    if "memcheck" in reports:
        flags["memcheck"] = "ERROR SUMMARY: 0 errors" in reports["memcheck"]
    if "racecheck" in reports:
        flags["racecheck"] = (
            "RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)"
            in reports["racecheck"]
        )
    return flags


def write_packet(
    output_dir,
    cases: pd.DataFrame,
    case_summary: pd.DataFrame,
    summary: dict,
    verify_text: str,
) -> int:
    """Write the standard per-gate verification packet and echo the summary.

    Writes `cases.csv`, `case-summary.csv`, `summary.json`, and `VERIFY.md`
    under `output_dir`, then prints the summary JSON. Returns the number of
    failure rows in `cases` (rows where the `pass` column is not 1).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output_dir / "cases.csv", index=False)
    case_summary.to_csv(output_dir / "case-summary.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "VERIFY.md").write_text(verify_text)
    print(json.dumps(summary, indent=2))
    return int(cases["pass"].ne(1).sum())

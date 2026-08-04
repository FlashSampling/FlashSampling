"""Validate CTA max-with-index predication on partial M and N tiles."""

from io import StringIO

import pandas as pd

from ..utils import make_app
from .gate_common import result_dir, run_compute_sanitizer, sanitizer_pass, write_packet
from .utils import add_cutlass_cta_boundary_max, make_cutlass_image

app = make_app()
image = add_cutlass_cta_boundary_max(make_cutlass_image())

OUTPUT_DIR = result_dir("06-cta-boundary-max")
ARCHITECTURES = ("sm90", "sm100")
M_EXTENTS = (100, 127, 128, 129, 255, 256, 257)
N_EXTENTS = (1, 2, 63, 64, 65, 127, 128, 129)
TILE_EXTENT = 128
EXPECTED_COLUMNS = [
    "architecture",
    "m_extent",
    "n_extent",
    "m_tile",
    "n_tile",
    "valid_m",
    "valid_n",
    "column",
    "valid_column",
    "final_valid_m",
    "padded_m_count",
    "padded_m_sentinel_bits",
    "expected_value_bits",
    "actual_value_bits",
    "expected_index",
    "actual_index",
    "pass",
]
CSV_HEADER = ",".join(EXPECTED_COLUMNS)


@app.function(gpu="H100", image=image, timeout=10 * 60)
def record_sm90() -> dict[str, str]:
    return run_compute_sanitizer(
        "/opt/fmms/cutlass_cta_boundary_max_sm90",
        ("memcheck", "racecheck"),
        CSV_HEADER,
        exact_csv_header=True,
    )


@app.function(gpu="B200", image=image, timeout=10 * 60)
def record_sm100() -> dict[str, str]:
    return run_compute_sanitizer(
        "/opt/fmms/cutlass_cta_boundary_max_sm100",
        ("memcheck", "racecheck"),
        CSV_HEADER,
        exact_csv_header=True,
    )


def _validate(csv_text: str, architecture: str) -> pd.DataFrame:
    cases = pd.read_csv(StringIO(csv_text))
    if cases.columns.tolist() != EXPECTED_COLUMNS:
        raise RuntimeError(f"Unexpected columns: {cases.columns.tolist()}")
    if set(cases["architecture"]) != {architecture}:
        raise RuntimeError(f"Unexpected architecture rows for {architecture}")

    shape_summary = (
        cases.groupby(["m_extent", "n_extent"], as_index=False)
        .agg(
            row_count=("column", "count"),
            column_count=("column", "nunique"),
            valid_m=("valid_m", "first"),
            valid_n=("valid_n", "first"),
            valid_column_count=("valid_column", "sum"),
            final_valid_m=("final_valid_m", "first"),
            padded_m_count=("padded_m_count", "first"),
            padded_m_sentinel_bits=("padded_m_sentinel_bits", "first"),
            mismatch_count=("pass", lambda values: values.ne(1).sum()),
        )
    )
    expected_shapes = pd.MultiIndex.from_product(
        [M_EXTENTS, N_EXTENTS], names=["m_extent", "n_extent"]
    ).to_frame(index=False)
    shape_coverage = expected_shapes.merge(
        shape_summary,
        on=["m_extent", "n_extent"],
        how="outer",
        indicator=True,
    )
    if not shape_coverage["_merge"].eq("both").all():
        raise RuntimeError(f"Incomplete shape Cartesian product for {architecture}")

    expected_valid_m = shape_summary["m_extent"].map(_boundary_extent)
    expected_valid_n = shape_summary["n_extent"].map(_boundary_extent)
    checks = (
        shape_summary["row_count"].eq(TILE_EXTENT)
        & shape_summary["column_count"].eq(TILE_EXTENT)
        & shape_summary["valid_m"].eq(expected_valid_m)
        & shape_summary["valid_n"].eq(expected_valid_n)
        & shape_summary["valid_column_count"].eq(expected_valid_n)
        & shape_summary["final_valid_m"].eq(expected_valid_m - 1)
        & shape_summary["padded_m_count"].eq(TILE_EXTENT - expected_valid_m)
        & shape_summary["mismatch_count"].eq(0)
    )
    partial_m = shape_summary.query("valid_m < @TILE_EXTENT")
    if not checks.all():
        raise RuntimeError(f"Boundary shape evidence is incomplete for {architecture}")
    if partial_m.empty or partial_m["padded_m_sentinel_bits"].eq(0).any():
        raise RuntimeError(f"Padded M sentinels are absent for {architecture}")
    if not cases["pass"].eq(1).all():
        failures = cases.query("`pass` != 1")
        raise RuntimeError(f"{len(failures)} boundary comparisons failed")
    return cases


def _boundary_extent(extent: int) -> int:
    remainder = extent % TILE_EXTENT
    return TILE_EXTENT if remainder == 0 else remainder


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    sanitizer_pass_flags = {}
    for architecture, remote_function in (
        ("sm90", record_sm90),
        ("sm100", record_sm100),
    ):
        result = remote_function.remote()
        frames.append(_validate(result["csv"], architecture))
        sanitizer_pass_flags[architecture] = sanitizer_pass(result)
        for tool in ("memcheck", "racecheck"):
            (OUTPUT_DIR / f"{tool}-{architecture}.txt").write_text(result[tool])
        if not all(sanitizer_pass_flags[architecture].values()):
            raise RuntimeError(f"Sanitizer did not pass for {architecture}")

    cases = pd.concat(frames, ignore_index=True)
    case_summary = (
        cases.assign(
            mismatch=cases["pass"].ne(1),
            padded_column=cases["valid_column"].eq(0),
        )
        .groupby(["architecture", "m_extent", "n_extent"], as_index=False)
        .agg(
            m_tile=("m_tile", "first"),
            n_tile=("n_tile", "first"),
            valid_m=("valid_m", "first"),
            valid_n=("valid_n", "first"),
            final_valid_m=("final_valid_m", "first"),
            padded_m_count=("padded_m_count", "first"),
            padded_m_sentinel_bits=("padded_m_sentinel_bits", "first"),
            valid_column_count=("valid_column", "sum"),
            padded_column_count=("padded_column", "sum"),
            mismatch_count=("mismatch", "sum"),
            pass_status=("pass", "min"),
        )
        .rename(columns={"pass_status": "pass"})
    )
    failures = cases.query("`pass` != 1")
    expected_count = len(ARCHITECTURES) * len(M_EXTENTS) * len(N_EXTENTS) * 128
    expected_summary_count = len(ARCHITECTURES) * len(M_EXTENTS) * len(N_EXTENTS)
    summary = {
        "gate": "1f",
        "command": "make modal-cutlass GATE=cta-boundary-max",
        "expected_result": (
            "Exact tile-local FP32 max-with-index results for the full M and N "
            "boundary-shape Cartesian product, with padded M sentinels excluded "
            "and padded N outputs left unchanged."
        ),
        "actual_result": (
            f"All {len(cases):,} comparisons matched exactly across "
            f"{len(case_summary)} architecture-shape combinations."
        ),
        "status": (
            "pass"
            if failures.empty
            and len(cases) == expected_count
            and len(case_summary) == expected_summary_count
            and all(
                all(tool_results.values())
                for tool_results in sanitizer_pass_flags.values()
            )
            else "fail"
        ),
        "architectures": list(ARCHITECTURES),
        "m_extents": list(M_EXTENTS),
        "n_extents": list(N_EXTENTS),
        "tile_extent": TILE_EXTENT,
        "expected_count": expected_count,
        "actual_count": len(cases),
        "expected_case_summary_count": expected_summary_count,
        "actual_case_summary_count": len(case_summary),
        "failure_count": len(failures),
        "shape_cartesian_product_complete": True,
        "final_valid_m_winner_per_shape": True,
        "padded_m_sentinel_larger_than_valid_winner": True,
        "padded_n_output_canary_preserved": True,
        "index_scope": "tile-local",
        "exact_fp32_bit_comparison": True,
        "sanitizer_pass": sanitizer_pass_flags,
        "warp_specialized_cutlass_kernel": False,
        "raw_measurements": {
            "applicable": False,
            "reason": "Gate 1f is a correctness gate, not a performance gate.",
        },
        "statistics": {
            "applicable": False,
            "reason": "Gate 1f performs deterministic exact comparisons.",
        },
    }
    if summary["status"] != "pass":
        raise RuntimeError("Gate 1f verification packet is incomplete")
    write_packet(
        OUTPUT_DIR,
        cases,
        case_summary,
        summary,
        _VERIFY,
    )


_VERIFY = """# Gate 1f verification

Expected:

- Both `sm90` and `sm100` are present.
- Every declared M and N extent appears in the full Cartesian product.
- Every shape records all 128 output columns.
- Valid columns select the final valid tile-local M coordinate.
- A larger maximum in every padded M row never wins.
- Every padded N output retains its initialization canary.
- Memcheck reports zero errors on both architectures.
- Racecheck reports zero hazards, errors, and warnings on both architectures.

Actual:

- `summary.json` records the expected and actual row and shape counts.
- `case-summary.csv` contains one compact row per architecture and shape.
- `cases.csv` retains all valid-result and padded-output comparisons.
- Four sanitizer reports retain the complete memory and race evidence.

Inspect `case-summary.csv` first.
Confirm that partial M shapes have nonzero `padded_m_count` and
`padded_m_sentinel_bits`, and partial N shapes have nonzero
`padded_column_count`.
Search `log.txt` and all sanitizer reports for errors, exceptions, skipped
tests, NaNs, and fallbacks.
Gate 1f intentionally retains tile-local indices.
Gate 1g adds global vocabulary indices and cross-tile tie handling.
Gate 1f does not run a warp-specialized CUTLASS kernel.
"""

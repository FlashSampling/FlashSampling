"""Validate accumulator ownership with partitioned SM100 TMEM loads."""

import json
import subprocess
from io import StringIO
from pathlib import Path

from ..utils import make_app
from .utils import add_cutlass_multi_warpgroup_layout, make_cutlass_image

app = make_app()
image = add_cutlass_multi_warpgroup_layout(make_cutlass_image())

OUTPUT_DIR = Path(
    "benchmarking/modal-results/cutlass/27-multi-warpgroup-layout"
)
TILE_M = 256
TILE_N = 128


@app.function(gpu="B200", image=image, timeout=10 * 60)
def record() -> tuple[str, dict]:
    import pandas as pd

    process = subprocess.run(
        ["/opt/fmms/cutlass_multi_warpgroup_layout"],
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        print(process.stdout)
        print(process.stderr)
        process.check_returncode()
    csv_text = process.stdout
    mapping = pd.read_csv(StringIO(csv_text))
    expected_columns = [
        "m",
        "n",
        "thread",
        "fragment",
        "epi_v",
        "epi_m",
        "epi_n",
        "cta",
        "owner_count",
        "owner_group_mask",
    ]
    if mapping.columns.tolist() != expected_columns:
        raise RuntimeError(f"Unexpected columns: {mapping.columns.tolist()}")
    if len(mapping) != TILE_M * TILE_N:
        raise RuntimeError(
            f"Expected {TILE_M * TILE_N} coordinates, found {len(mapping)}"
        )
    owner_counts = mapping["owner_count"]
    failures = []
    if not owner_counts.eq(1).all():
        histogram = owner_counts.value_counts().sort_index().to_dict()
        failures.append(f"Ownership is not unique and complete: {histogram}")
    owner_group_masks = sorted(mapping["owner_group_mask"].unique().tolist())
    if owner_group_masks != [1, 2, 4, 8]:
        failures.append(
            f"Expected all four callback-owner groups, found {owner_group_masks}"
        )
    threads = sorted(mapping["thread"].unique().tolist())

    summary = {
        "passed": not failures,
        "failures": failures,
        "tile_m": TILE_M,
        "tile_n": TILE_N,
        "coordinates": len(mapping),
        "callback_visits": int(owner_counts.sum()),
        "owner_count_histogram": {
            str(int(count)): int(coordinates)
            for count, coordinates in owner_counts.value_counts()
            .sort_index()
            .items()
        },
        "owner_group_masks": owner_group_masks,
        "threads": threads,
        "epilogue_iterations_m": sorted(mapping["epi_m"].unique().tolist()),
        "epilogue_iterations_n": sorted(mapping["epi_n"].unique().tolist()),
        "epilogue_vectors": sorted(mapping["epi_v"].unique().tolist()),
        "fragment_slots": sorted(mapping["fragment"].unique().tolist()),
        "ctas": sorted(mapping["cta"].unique().tolist()),
    }
    print(json.dumps(summary, indent=2))
    return csv_text, summary


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_text, summary = record.remote()
    (OUTPUT_DIR / "ownership.csv").write_text(csv_text)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if not summary["passed"]:
        raise RuntimeError("; ".join(summary["failures"]))

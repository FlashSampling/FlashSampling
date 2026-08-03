"""Gate 2d accumulator ownership for the winning B200 GEMM schedules."""

import json
import subprocess
from io import StringIO
from pathlib import Path

from ..utils import make_app
from .utils import add_cutlass_winning_schedule_layout, make_cutlass_image

app = make_app()
image = add_cutlass_winning_schedule_layout(make_cutlass_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/15-winning-schedule-layout")
VARIANTS = {
    "128x64x128-c2": (128, 64, "m128_n64"),
    "256x128x64-c2": (256, 128, "m256_n128"),
    "256x128x64-c4": (256, 128, "m256_n128"),
    "256x128x128-c2": (256, 128, "m256_n128"),
    "256x256x64-c2": (256, 256, "m256_n256"),
}


@app.function(gpu="B200", image=image, timeout=10 * 60)
def record() -> dict[str, tuple[str, dict]]:
    import pandas as pd

    results = {}
    for name, (tile_m, tile_n, formula) in VARIANTS.items():
        executable = f"/opt/fmms/cutlass_winning_layout_{name}"
        csv_text = subprocess.check_output([executable], text=True)
        mapping = pd.read_csv(StringIO(csv_text))
        expected_columns = [
            "m", "n", "thread", "fragment", "epi_v", "epi_m", "epi_n", "cta"
        ]
        if mapping.columns.tolist() != expected_columns:
            raise RuntimeError(f"{name}: unexpected columns {mapping.columns.tolist()}")
        if len(mapping) != tile_m * tile_n:
            raise RuntimeError(
                f"{name}: expected {tile_m * tile_n} coordinates, found {len(mapping)}"
            )
        if mapping[["m", "n"]].duplicated().any():
            raise RuntimeError(f"{name}: at least one coordinate has multiple owners")
        expected = pd.MultiIndex.from_product(
            [range(tile_m), range(tile_n)], names=["m", "n"]
        )
        actual = pd.MultiIndex.from_frame(mapping[["m", "n"]])
        if not actual.sort_values().equals(expected):
            raise RuntimeError(f"{name}: diagnostic did not cover every coordinate")
        _verify_formula(mapping, formula, name)
        summary = {
            "variant": name,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "coordinates": len(mapping),
            "ownership_formula": formula,
            "epilogue_iterations_m": sorted(mapping["epi_m"].unique().tolist()),
            "epilogue_iterations_n": sorted(mapping["epi_n"].unique().tolist()),
            "epilogue_vectors": sorted(mapping["epi_v"].unique().tolist()),
            "fragment_slots": sorted(mapping["fragment"].unique().tolist()),
            "threads": sorted(mapping["thread"].unique().tolist()),
            "ctas": sorted(mapping["cta"].unique().tolist()),
        }
        print(json.dumps(summary, indent=2))
        results[name] = (csv_text, summary)
    return results


def _verify_formula(mapping, formula: str, name: str) -> None:
    thread = mapping["thread"] - 128
    if formula == "m128_n64":
        expected_m = thread.mod(64) + 64 * mapping["cta"]
        expected_n = 32 * thread.floordiv(64) + 16 * mapping["epi_n"] + mapping["fragment"]
    elif formula == "m256_n128":
        expected_m = thread + 128 * mapping["cta"]
        expected_n = 16 * mapping["epi_n"] + mapping["fragment"]
    elif formula == "m256_n256":
        expected_m = thread + 128 * mapping["cta"]
        expected_n = 32 * mapping["epi_n"] + mapping["fragment"]
    else:
        raise RuntimeError(f"{name}: unknown ownership formula {formula}")
    if not mapping["m"].equals(expected_m) or not mapping["n"].equals(expected_n):
        raise RuntimeError(f"{name}: observed coordinates do not match {formula}")


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, (csv_text, summary) in record.remote().items():
        (OUTPUT_DIR / f"{name}.csv").write_text(csv_text)
        (OUTPUT_DIR / f"{name}-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

"""Record CUTLASS accumulator-fragment ownership on H100 and B200."""

import json
import subprocess
from io import StringIO
from pathlib import Path

from ..utils import make_app
from .utils import add_cutlass_accumulator_layout, make_cutlass_image

app = make_app()
image = add_cutlass_accumulator_layout(make_cutlass_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/01-accumulator-layout")
M = 128
N = 128


@app.function(gpu="H100", image=image, timeout=10 * 60)
def record_sm90() -> tuple[str, dict]:
    return _record("/opt/fmms/cutlass_accumulator_layout_sm90", "sm90")


@app.function(gpu="B200", image=image, timeout=10 * 60)
def record_sm100() -> tuple[str, dict]:
    return _record("/opt/fmms/cutlass_accumulator_layout_sm100", "sm100")


def _record(executable: str, architecture: str) -> tuple[str, dict]:
    import pandas as pd

    csv_text = subprocess.check_output([executable], text=True)
    mapping = pd.read_csv(StringIO(csv_text))
    expected_columns = [
        "m", "n", "thread", "fragment", "epi_v", "epi_m", "epi_n", "cta"
    ]
    if mapping.columns.tolist() != expected_columns:
        raise RuntimeError(f"Unexpected columns: {mapping.columns.tolist()}")
    if len(mapping) != M * N:
        raise RuntimeError(f"Expected {M * N} coordinates, found {len(mapping)}")
    if mapping[["m", "n"]].duplicated().any():
        raise RuntimeError("At least one output coordinate has multiple owners")
    expected_coordinates = pd.MultiIndex.from_product([range(M), range(N)], names=["m", "n"])
    actual_coordinates = pd.MultiIndex.from_frame(mapping[["m", "n"]])
    if not actual_coordinates.sort_values().equals(expected_coordinates):
        raise RuntimeError("The diagnostic did not cover every output coordinate")

    summary = {
        "architecture": architecture,
        "coordinates": len(mapping),
        "epilogue_iterations_m": sorted(mapping["epi_m"].unique().tolist()),
        "epilogue_iterations_n": sorted(mapping["epi_n"].unique().tolist()),
        "epilogue_vectors": sorted(mapping["epi_v"].unique().tolist()),
        "fragment_slots": sorted(mapping["fragment"].unique().tolist()),
        "threads": sorted(mapping["thread"].unique().tolist()),
        "ctas": sorted(mapping["cta"].unique().tolist()),
    }
    print(json.dumps(summary, indent=2))
    return csv_text, summary


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for architecture, remote_function in (("sm90", record_sm90), ("sm100", record_sm100)):
        csv_text, summary = remote_function.remote()
        (OUTPUT_DIR / f"{architecture}.csv").write_text(csv_text)
        (OUTPUT_DIR / f"{architecture}-summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

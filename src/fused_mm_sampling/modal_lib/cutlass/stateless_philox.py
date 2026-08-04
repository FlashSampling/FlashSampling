"""Gate 3: validate and profile stateless Philox streams on B200."""

import io
import json
import math
import re
import subprocess
import tempfile
from pathlib import Path

import modal
import numpy as np
import pandas as pd

from ..utils import make_app, make_ncu_image
from .utils import add_cutlass_stateless_philox

app = make_app()
image = add_cutlass_stateless_philox(make_ncu_image(include_library_code=False))

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/17-stateless-philox")
SEEDS = (0, 1, 0x0123456789ABCDEF, 0xFFFFFFFFFFFFFFFF)
LAUNCHES = ((128, 64), (256, 128), (512, 256))
COUNT = 4 * 4 * 65536
UNIFORM_BINS = 256
FAMILY_ALPHA = 0.01
PER_TEST_ALPHA = FAMILY_ALPHA / len(SEEDS)
REQUIRED_PROFILE_METRICS = (
    "gpu__time_duration.sum",
    "launch__registers_per_thread",
    "smsp__inst_executed.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
)
OPTIONAL_PROFILE_METRICS = (
    "smsp__inst_executed_pipe_xu.sum.pct_of_peak_sustained_active",
    "smsp__issue_active.avg.pct_of_peak_sustained_active",
)


@app.function(gpu="B200", image=image, timeout=20 * 60)
def validate_b200() -> str:
    rows = []
    reference_rows = []
    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)
        for seed in SEEDS:
            canonical = None
            for threads, tile_vocab in LAUNCHES:
                output_path = directory_path / f"{seed}-{threads}-{tile_vocab}.bin"
                subprocess.run(
                    [
                        "/opt/fmms/cutlass_stateless_philox",
                        str(threads),
                        str(tile_vocab),
                        str(seed),
                        str(output_path),
                    ],
                    check=True,
                )
                values = np.fromfile(output_path, dtype=np.uint32).reshape(-1, 4)
                if values.shape != (COUNT, 4):
                    raise RuntimeError(f"Unexpected Philox output shape {values.shape}")
                if canonical is None:
                    canonical = values.copy()
                    sorted_blocks = np.unique(canonical, axis=0)
                    duplicate_blocks = len(canonical) - len(sorted_blocks)
                    counts = np.bincount(
                        (canonical[:, 0] >> 24).astype(np.int64),
                        minlength=UNIFORM_BINS,
                    )
                    expected = COUNT / UNIFORM_BINS
                    chi_square = float(np.square(counts - expected).sum() / expected)
                    p_value = _chi_square_survival(chi_square, UNIFORM_BINS - 1)
                    reference_rows.extend(
                        {
                            "seed": seed,
                            "linear": linear,
                            "x": int(value[0]),
                            "y": int(value[1]),
                            "z": int(value[2]),
                            "w": int(value[3]),
                        }
                        for linear, value in enumerate(canonical[:8])
                    )
                else:
                    duplicate_blocks = 0
                    chi_square = math.nan
                    p_value = math.nan
                mismatch_count = int(np.count_nonzero(values != canonical))
                rows.append(
                    {
                        "seed": seed,
                        "threads": threads,
                        "tile_vocab": tile_vocab,
                        "output_count": len(values),
                        "word_mismatch_count": mismatch_count,
                        "duplicate_block_count": duplicate_blocks,
                        "chi_square_255": chi_square,
                        "p_value": p_value,
                        "uniformity_pass": bool(
                            math.isnan(p_value) or p_value >= PER_TEST_ALPHA
                        ),
                    }
                )
    profile = _profile()
    return json.dumps({"cases": rows, "reference": reference_rows, **profile})


def _profile() -> dict:
    query = subprocess.run(
        ["ncu", "--query-metrics", "--query-metrics-mode", "all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    optional_metrics = [
        metric for metric in OPTIONAL_PROFILE_METRICS if metric in query
    ]
    metrics = [*REQUIRED_PROFILE_METRICS, *optional_metrics]
    with tempfile.TemporaryDirectory() as directory:
        report_base = Path(directory) / "philox"
        report_path = report_base.with_suffix(".ncu-rep")
        subprocess.run(
            [
                "ncu",
                "--metrics",
                ",".join(metrics),
                "--kernel-name",
                "regex:.*generate.*",
                "-f",
                "-o",
                str(report_base),
                "/opt/fmms/cutlass_stateless_philox",
                "256",
                "128",
                "0",
                str(Path(directory) / "unused.bin"),
                "--profile",
            ],
            check=True,
        )
        export = subprocess.run(
            ["ncu", "--import", str(report_path), "--csv", "--page", "raw"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    csv_lines = [line for line in export.splitlines() if line.startswith('"')]
    frame = pd.read_csv(io.StringIO("\n".join(csv_lines)))
    units = frame.iloc[0]
    frame = frame.dropna(subset=["Kernel Name"]).copy()
    for metric in metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    frame["duration_unit"] = units["gpu__time_duration.sum"]
    sass = subprocess.run(
        ["cuobjdump", "--dump-sass", "/opt/fmms/cutlass_stateless_philox"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    generate_section = re.search(
        r"Function\s*:\s*[^\n]*generate[^\n]*\n(.*?)(?=\n\s*Function\s*:|\Z)",
        sass,
        flags=re.DOTALL,
    )
    if generate_section is None:
        raise RuntimeError("Could not isolate the generate kernel in SASS")
    generate_sass = generate_section.group(1)
    return {
        "profile": frame[["Kernel Name", *metrics, "duration_unit"]].to_dict(
            orient="records"
        ),
        "static_mufu_instruction_count": generate_sass.count("MUFU"),
    }


def _chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    """Regularized upper incomplete gamma Q(k/2, x/2)."""
    a = degrees_of_freedom / 2
    x = statistic / 2
    if x < a + 1:
        term = 1 / a
        total = term
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        lower = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, min(1.0, 1 - lower))
    b = x + 1 - a
    c = 1 / 1e-300
    d = 1 / b
    h = d
    for iteration in range(1, 1001):
        an = -iteration * (iteration - a)
        b += 2
        d = an * d + b
        if abs(d) < 1e-300:
            d = 1e-300
        c = b + an / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-14:
            break
    return max(0.0, min(1.0, math.exp(-x + a * math.log(x) - math.lgamma(a)) * h))


def _philox_reference(seed: int, sample: int, hidden: int, vocab: int) -> tuple[int, ...]:
    mask = 0xFFFFFFFF
    value = [vocab & mask, vocab >> 32, hidden, sample]
    key0, key1 = seed & mask, seed >> 32
    for _ in range(10):
        product0 = 0xD2511F53 * value[0]
        product1 = 0xCD9E8D57 * value[2]
        value = [
            ((product1 >> 32) ^ value[1] ^ key0) & mask,
            product1 & mask,
            ((product0 >> 32) ^ value[3] ^ key1) & mask,
            product0 & mask,
        ]
        key0 = (key0 + 0x9E3779B9) & mask
        key1 = (key1 + 0xBB67AE85) & mask
    return tuple(value)


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    packet = json.loads(validate_b200.remote())
    cases = pd.DataFrame(packet["cases"])
    references = pd.DataFrame(packet["reference"])
    decoded = references.assign(
        sample=references["linear"] // (4 * 65536),
        hidden=(references["linear"] % (4 * 65536)) // 65536,
        vocab=references["linear"] % 65536,
    )
    expected = pd.DataFrame.from_records(
        [
            _philox_reference(int(seed), int(sample), int(hidden), int(vocab))
            for seed, sample, hidden, vocab in decoded[
                ["seed", "sample", "hidden", "vocab"]
            ].itertuples(index=False, name=None)
        ]
    )
    expected.columns = ["expected_x", "expected_y", "expected_z", "expected_w"]
    references = pd.concat([decoded, expected], axis=1)
    reference_pass = references[["x", "y", "z", "w"]].to_numpy().tolist() == (
        references[["expected_x", "expected_y", "expected_z", "expected_w"]]
        .to_numpy().tolist()
    )
    status = (
        reference_pass
        and cases["word_mismatch_count"].eq(0).all()
        and cases["duplicate_block_count"].eq(0).all()
        and cases["uniformity_pass"].all()
    )
    cases.to_csv(OUTPUT_DIR / "cases.csv", index=False)
    references.to_csv(OUTPUT_DIR / "reference-vectors.csv", index=False)
    profile = pd.DataFrame(packet["profile"])
    profile.to_csv(OUTPUT_DIR / "profile.csv", index=False)
    profile_row = profile.iloc[0]
    registers = int(profile_row["launch__registers_per_thread"])
    local_loads = int(profile_row["smsp__sass_inst_executed_op_local_ld.sum"])
    local_stores = int(profile_row["smsp__sass_inst_executed_op_local_st.sum"])
    static_mufu_instructions = int(packet["static_mufu_instruction_count"])
    executed_instructions = int(profile_row["smsp__inst_executed.sum"])
    instructions_per_output = executed_instructions / COUNT
    phase_b_pass = (
        registers <= 32
        and local_loads == 0
        and local_stores == 0
        and static_mufu_instructions == 0
        and instructions_per_output <= 6
    )
    summary = {
        "gate": "3",
        "command": "make modal-cutlass GATE=stateless-philox",
        "status": "pass" if status and phase_b_pass else "fail",
        "gpu": "B200",
        "coordinate": "(seed, sample_idx, hidden_idx, global_vocab_idx)",
        "seeds": list(SEEDS),
        "launches": [list(launch) for launch in LAUNCHES],
        "outputs_per_seed": COUNT,
        "full_block_collision_count": int(cases["duplicate_block_count"].sum()),
        "launch_word_mismatch_count": int(cases["word_mismatch_count"].sum()),
        "reference_vector_count": len(references),
        "reference_vectors_pass": reference_pass,
        "uniformity": {
            "tested_word": "high 8 bits of Philox x",
            "bins": UNIFORM_BINS,
            "degrees_of_freedom": UNIFORM_BINS - 1,
            "family_alpha": FAMILY_ALPHA,
            "multiple_test_policy": "Bonferroni across four seeds",
            "per_test_alpha": PER_TEST_ALPHA,
            "minimum_p_value": float(cases["p_value"].min()),
        },
        "phase_b": {
            "profile_file": "profile.csv",
            "outputs": COUNT,
            "registers_per_thread": registers,
            "register_threshold": 32,
            "local_load_instructions": local_loads,
            "local_store_instructions": local_stores,
            "static_mufu_instructions": static_mufu_instructions,
            "xu_sfu_pipe_utilization_pct": float(
                profile_row[
                    "smsp__inst_executed_pipe_xu.sum.pct_of_peak_sustained_active"
                ]
            ),
            "issue_active_pct": float(
                profile_row["smsp__issue_active.avg.pct_of_peak_sustained_active"]
            ),
            "executed_warp_instructions": executed_instructions,
            "warp_instructions_per_output": instructions_per_output,
            "warp_instructions_per_output_threshold": 6,
            "status": "pass" if phase_b_pass else "fail",
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# Gate 3 verification

Expected: all launch layouts produce identical words, all full 128-bit blocks
are unique, CPU reference vectors match exactly, and all four byte-frequency
chi-squared tests pass the Bonferroni-corrected threshold.

Inspect `summary.json`, then `cases.csv` and `reference-vectors.csv`.
`profile.csv` records instruction count, registers, issue-slot utilization,
local-memory instructions, and SFU-pipe instructions for the same kernel.
"""
    )
    if not status or not phase_b_pass:
        raise RuntimeError("Gate 3 failed")

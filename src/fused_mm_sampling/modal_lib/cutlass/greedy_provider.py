"""Gate 2a correctness matrix for the production CUTLASS greedy provider."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from ..utils import make_app
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(make_cutlass_provider_image())

OUTPUT_DIR = Path("benchmarking/modal-results/cutlass/09-greedy-provider")
ARCHITECTURES = ("sm90", "sm100")
HIDDEN_STATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
MODEL_SHAPES = ((151_936, 4_096), (128_256, 8_192))
BOUNDARY_VOCABS = (100, 127, 128, 129, 255, 256, 257)
COLUMNS = [
    "architecture",
    "family",
    "case",
    "vocab_size",
    "hidden_size",
    "n_hidden_states",
    "expected_indices",
    "actual_indices",
    "pass",
]


@app.function(gpu="H100", image=image, timeout=30 * 60)
def record_sm90() -> dict:
    return _run("sm90")


@app.function(gpu="B200", image=image, timeout=30 * 60)
def record_sm100() -> dict:
    return _run("sm100")


def _run(architecture: str) -> dict:
    import torch

    from fused_mm_sampling.core import get_sampler

    rows = []
    temperature = torch.empty((), device="cuda")
    for vocab_size in BOUNDARY_VOCABS:
        rows.append(
            _run_case(
                architecture,
                "boundaries",
                f"vocab_{vocab_size}",
                vocab_size,
                64,
                2,
                temperature,
                get_sampler,
            )
        )
    rows.append(
        _run_tie_case(architecture, temperature, get_sampler)
    )
    for vocab_size, hidden_size in MODEL_SHAPES:
        for n_hidden_states in HIDDEN_STATES:
            rows.append(
                _run_case(
                    architecture,
                    "model_shapes",
                    f"v{vocab_size}_d{hidden_size}_h{n_hidden_states}",
                    vocab_size,
                    hidden_size,
                    n_hidden_states,
                    temperature,
                    get_sampler,
                )
            )
    pytest_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_core.py::test_greedy_sampling",
            "-k",
            "cutlass",
        ],
        cwd="/opt/fmms/repo",
        check=False,
        capture_output=True,
        text=True,
    )
    pytest_log = pytest_result.stdout + pytest_result.stderr
    if pytest_result.returncode != 0:
        raise RuntimeError(
            f"CUTLASS provider pytest failed on {architecture}:\n{pytest_log[-12_000:]}"
        )
    return {"rows": rows, "pytest_log": pytest_log}


def _run_case(
    architecture,
    family,
    case,
    vocab_size,
    hidden_size,
    n_hidden_states,
    temperature,
    get_sampler,
):
    import torch

    weights = torch.zeros(
        (vocab_size, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    hidden_states = torch.zeros(
        (n_hidden_states, hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    columns = torch.arange(n_hidden_states, device="cuda")
    winners = (
        torch.arange(n_hidden_states, device="cuda", dtype=torch.long) * 997 + 13
    ) % vocab_size
    hidden_states[columns, columns] = 1
    weights[winners, columns] = 2
    sampler = get_sampler("fused-cutlass-greedy", weights=weights)
    sampler.prepare()
    actual = sampler.sample(
        weights=weights,
        hidden_states=hidden_states,
        num_samples=1,
        temperature=temperature,
    )[:, 0]
    passed = torch.equal(actual, winners)
    row = {
        "architecture": architecture,
        "family": family,
        "case": case,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "n_hidden_states": n_hidden_states,
        "expected_indices": ";".join(map(str, winners.tolist())),
        "actual_indices": ";".join(map(str, actual.tolist())),
        "pass": int(passed),
    }
    del actual, sampler, hidden_states, weights
    if not passed:
        raise RuntimeError(f"CUTLASS greedy mismatch: {row}")
    return row


def _run_tie_case(architecture, temperature, get_sampler):
    import torch

    weights = torch.zeros((257, 64), dtype=torch.bfloat16, device="cuda")
    hidden_states = torch.zeros((1, 64), dtype=torch.bfloat16, device="cuda")
    hidden_states[0, 0] = 1
    weights[7, 0] = 3
    weights[135, 0] = 3
    sampler = get_sampler("fused-cutlass-greedy", weights=weights)
    actual = sampler.sample(
        weights=weights,
        hidden_states=hidden_states,
        num_samples=1,
        temperature=temperature,
    )[:, 0]
    row = {
        "architecture": architecture,
        "family": "ties",
        "case": "cross_tile_lowest_index",
        "vocab_size": 257,
        "hidden_size": 64,
        "n_hidden_states": 1,
        "expected_indices": "7",
        "actual_indices": str(actual.item()),
        "pass": int(actual.item() == 7),
    }
    if not row["pass"]:
        raise RuntimeError(f"CUTLASS greedy tie mismatch: {row}")
    return row


@app.local_entrypoint()
def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    pytest_pass = {}
    for architecture, remote_function in (
        ("sm90", record_sm90),
        ("sm100", record_sm100),
    ):
        result = remote_function.remote()
        rows.extend(result["rows"])
        pytest_pass[architecture] = "passed" in result["pytest_log"]
        (OUTPUT_DIR / f"pytest-{architecture}.txt").write_text(
            result["pytest_log"]
        )
    if not all(pytest_pass.values()):
        raise RuntimeError(f"Provider pytest evidence is incomplete: {pytest_pass}")
    cases = pd.DataFrame(rows, columns=COLUMNS)
    expected_rows = 2 * (len(BOUNDARY_VOCABS) + 1 + len(MODEL_SHAPES) * len(HIDDEN_STATES))
    if len(cases) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, got {len(cases)}")
    if set(cases["architecture"]) != set(ARCHITECTURES):
        raise RuntimeError("One or more architectures are absent")
    if not cases["pass"].eq(1).all():
        raise RuntimeError("Gate 2a has output mismatches")
    cases.to_csv(OUTPUT_DIR / "cases.csv", index=False)
    case_summary = (
        cases.groupby(["architecture", "family"], as_index=False)
        .agg(case_count=("case", "count"), pass_count=("pass", "sum"), passed=("pass", "min"))
    )
    case_summary.to_csv(OUTPUT_DIR / "case-summary.csv", index=False)
    summary = {
        "gate": "2a",
        "command": "make modal-cutlass GATE=greedy-provider",
        "status": "pass",
        "architectures": list(ARCHITECTURES),
        "provider": "fused-cutlass-greedy",
        "case_count": len(cases),
        "failure_count": 0,
        "pytest_pass": pytest_pass,
        "pytest_test": "tests/test_core.py::test_greedy_sampling -k cutlass",
        "model_shapes": [
            {"vocab_size": vocab_size, "hidden_size": hidden_size}
            for vocab_size, hidden_size in MODEL_SHAPES
        ],
        "hidden_state_sweep": list(HIDDEN_STATES),
        "limitations": ["BF16", "TP1", "greedy", "SM90/SM100"],
        "raw_measurements": {
            "applicable": False,
            "reason": "Gate 2a is a deterministic correctness gate.",
        },
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT_DIR / "VERIFY.md").write_text(
        """# Gate 2a verification

Expected:

- Both SM90 and SM100 are present.
- Every boundary, tie, and primary model-shape row passes.
- The full H=1 through H=256 power-of-two sweep is present for both model shapes.
- The provider name is `fused-cutlass-greedy`.
- `pytest-sm90.txt` and `pytest-sm100.txt` each report 18 passing cases from the shared greedy test.

Review `summary.json`, then `case-summary.csv`, then `cases.csv`.
"""
    )
    print(cases.to_csv(index=False), end="")

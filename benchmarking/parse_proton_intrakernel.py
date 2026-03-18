"""Parse Proton chrome traces produced by the TTGIR override workflow.

Scope names in the trace: "kernel", "setup", "mask", "tile-mgmt", "sample",
"store". Matmul time is derived as:
kernel - setup - mask - tile-mgmt - sample - store.

Usage:
    python parse_proton_intrakernel.py kernel.chrome_trace
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def parse_chrome_trace(path: Path) -> dict | None:
    """Parse a chrome trace and return aggregate scope durations (in cycles)."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.load(path.open())
    events = data["traceEvents"]
    if not events:
        return None

    scope_dur: dict[str, float] = defaultdict(float)
    scope_cnt: dict[str, int] = defaultdict(int)
    for e in events:
        scope_dur[e["name"]] += e["dur"]
        scope_cnt[e["name"]] += 1

    total = sum(scope_dur.values())
    if total == 0:
        return None
    return {"scopes": dict(scope_dur), "counts": dict(scope_cnt), "total_cycles": total}


def trace_phase_pcts(trace: dict) -> dict[str, float]:
    """Return {"matmul": pct, "sampling": pct} from a chrome trace.

    Matmul is derived as kernel - setup - mask - tile-mgmt - sample - store.
    Sampling aggregates mask + sample + store (tile-mgmt excluded).
    """
    scopes = trace["scopes"]
    if "kernel" not in scopes:
        return None
    kernel = scopes["kernel"]
    setup = scopes.get("setup", 0)
    mask = scopes.get("mask", 0)
    tile_mgmt = scopes.get("tile-mgmt", 0)
    sample = scopes.get("sample", 0)
    store = scopes.get("store", 0)
    matmul = kernel - setup - mask - tile_mgmt - sample - store
    return {
        "matmul": matmul / kernel * 100,
        "sampling": (mask + sample + store) / kernel * 100,
    }


EXPECTED_SCOPES = ["kernel", "setup", "mask", "tile-mgmt", "sample", "store"]


def print_breakdown(trace: dict) -> None:
    """Print per-scope runtime breakdown as percentages of kernel time."""
    scopes = trace["scopes"]
    counts = trace["counts"]
    missing = [s for s in EXPECTED_SCOPES if s not in scopes]
    if missing:
        print(
            f"WARNING: missing scopes (buffer overflow?): {missing}. Showing available scopes only."
        )

    kernel = scopes.get("kernel")
    if kernel is None:
        # kernel/setup dropped by circular buffer overflow at high bsz.
        # Show raw epilogue scopes without matmul derivation.
        rows = []
        for scope in EXPECTED_SCOPES:
            if scope in scopes:
                rows.append((scope, scopes[scope], counts[scope]))
        df = pd.DataFrame(rows, columns=["phase", "cycles", "count"])
        print(df.to_markdown(index=False))
        return

    setup = scopes.get("setup", 0)
    mask = scopes.get("mask", 0)
    tile_mgmt = scopes.get("tile-mgmt", 0)
    sample = scopes.get("sample", 0)
    store = scopes.get("store", 0)
    matmul = kernel - setup - mask - tile_mgmt - sample - store

    rows = [
        ("kernel", kernel, counts.get("kernel")),
        ("setup", setup, counts.get("setup")),
        ("matmul*", matmul, None),
        ("mask", mask, counts.get("mask")),
        ("tile-mgmt", tile_mgmt, counts.get("tile-mgmt")),
        ("sample", sample, counts.get("sample")),
        ("store", store, counts.get("store")),
    ]
    df = pd.DataFrame(rows, columns=["phase", "cycles", "count"])
    df["runtime_%"] = (df["cycles"] / kernel * 100).round(1)
    print(df.to_markdown(index=False))
    print()
    print("* matmul = kernel - setup - mask - tile-mgmt - sample - store")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Path to kernel.chrome_trace")
    args = parser.parse_args()
    trace = parse_chrome_trace(args.trace)
    if trace is None:
        print(f"No events in {args.trace}")
        raise SystemExit(1)
    print_breakdown(trace)

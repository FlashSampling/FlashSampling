"""Extract KV-cache occupancy, running reqs, and waiting reqs from vLLM sweep logs.

Usage:
    python parse_engine_stats.py <sweep.log> [--by-concurrency]

Parses the periodic engine stats lines:
    Engine 000: ... Running: 32 reqs, Waiting: 0 reqs, GPU KV cache usage: 14.2%, ...

The sweep log structure per benchmark run is:
    1. Namespace(..., max_concurrency=N, ...)   <- bench client args
    2. "Starting main benchmark run..."
    3. Engine stats interleaved with progress bars
    4. "Serving Benchmark Result" summary
    5. [BEGIN BENCHMARK] / [END BENCHMARK] post-hoc metadata

Engine stats are emitted every 10s by the vLLM server, so short runs
(low concurrency) may have 0-2 samples.
"""

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class BenchRun:
    max_concurrency: int = 0
    run_number: int = 0
    stats: list = field(default_factory=list)


_ENGINE_RE = re.compile(
    r"Running:\s*(\d+)\s*reqs.*?"
    r"Waiting:\s*(\d+)\s*reqs.*?"
    r"GPU KV cache usage:\s*([\d.]+)%"
)
# The bench client prints its Namespace with max_concurrency=N
_NAMESPACE_CONC_RE = re.compile(r"max_concurrency=(\d+)")
_RUN_NUMBER_RE = re.compile(r"Run Number:\s*(\d+)")


def parse_sweep_log(path: Path) -> list[BenchRun]:
    """Parse engine stats grouped by benchmark run.

    Uses "Starting main benchmark run..." as the start delimiter and
    "Serving Benchmark Result" as the end delimiter for each run.
    """
    runs: list[BenchRun] = []
    current: BenchRun | None = None
    pending_concurrency = 0
    pending_run = 0

    for line in path.read_text().splitlines():
        # Track concurrency and run number from lines that precede the
        # benchmark start.
        m = _RUN_NUMBER_RE.search(line)
        if m:
            pending_run = int(m.group(1))

        if "Namespace(" in line:
            m = _NAMESPACE_CONC_RE.search(line)
            if m:
                pending_concurrency = int(m.group(1))

        if "Starting main benchmark run..." in line:
            current = BenchRun(
                max_concurrency=pending_concurrency,
                run_number=pending_run,
            )
            continue

        if "Serving Benchmark Result" in line:
            if current:
                runs.append(current)
            current = None
            continue

        if current is None:
            continue

        m = _ENGINE_RE.search(line)
        if m:
            current.stats.append(
                {
                    "running": int(m.group(1)),
                    "waiting": int(m.group(2)),
                    "kv_cache_pct": float(m.group(3)),
                }
            )

    return runs


def summarize(runs: list[BenchRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        s = pd.DataFrame(run.stats) if run.stats else pd.DataFrame(
            {"running": [], "waiting": [], "kv_cache_pct": []}
        )
        rows.append(
            {
                "concurrency": run.max_concurrency,
                "run": run.run_number,
                "n_logs": len(s),
                "kv_mean": s["kv_cache_pct"].mean() if len(s) else 0,
                "kv_max": s["kv_cache_pct"].max() if len(s) else 0,
                "running_mean": s["running"].mean() if len(s) else 0,
                "running_max": s["running"].max() if len(s) else 0,
                "waiting_mean": s["waiting"].mean() if len(s) else 0,
                "waiting_max": s["waiting"].max() if len(s) else 0,
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, nargs="+", help="Path(s) to sweep.log")
    parser.add_argument(
        "--by-concurrency",
        action="store_true",
        help="Aggregate across runs per concurrency level",
    )
    args = parser.parse_args()

    all_runs = []
    for log_path in args.log:
        runs = parse_sweep_log(log_path)
        all_runs.extend(runs)

    if not all_runs:
        print("No benchmark runs found in log(s).", file=sys.stderr)
        sys.exit(1)

    df = summarize(all_runs)

    if args.by_concurrency:
        agg = (
            df.groupby("concurrency")
            .agg(
                n_logs=("n_logs", "sum"),
                kv_mean=("kv_mean", "mean"),
                kv_max=("kv_max", "max"),
                running_mean=("running_mean", "mean"),
                running_max=("running_max", "max"),
                waiting_mean=("waiting_mean", "mean"),
                waiting_max=("waiting_max", "max"),
            )
            .round(1)
        )
        print(agg.to_markdown())
    else:
        print(df.round(1).to_markdown(index=False))


if __name__ == "__main__":
    main()

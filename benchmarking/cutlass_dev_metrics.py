#!/usr/bin/env python3
"""Summarize structured CUTLASS development-loop measurements."""

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    args = _parse_args()
    metrics_dir = Path(args.metrics_dir)
    runs, builds = load_metrics(metrics_dir)
    if args.label:
        runs = runs.query("label == @args.label")
        builds = builds.query("label == @args.label")
    run_summary, build_summary = summarize_metrics(runs, builds)
    output_dir = Path(args.output_dir or metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_summary.to_csv(output_dir / "run-summary.csv", index=False)
    build_summary.to_csv(output_dir / "build-summary.csv", index=False)
    report = render_report(run_summary, build_summary)
    (output_dir / "summary.md").write_text(report)
    print(report, end="")


def load_metrics(metrics_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load run records and normalize nested extension-load events."""
    records = [json.loads(path.read_text()) for path in sorted(metrics_dir.glob("*.json"))]
    if not records:
        raise RuntimeError(f"No CUTLASS development metrics found under {metrics_dir}")
    runs = pd.DataFrame(
        {key: value for key, value in record.items() if key not in {"events", "git", "command"}}
        | {
            "git_commit": record.get("git", {}).get("commit"),
            "git_branch": record.get("git", {}).get("branch"),
            "git_dirty": record.get("git", {}).get("dirty"),
        }
        for record in records
    )
    build_frames = []
    for record in records:
        frame = pd.DataFrame(
            event for event in record.get("events", []) if event.get("event") == "extension_load"
        )
        if frame.empty:
            continue
        frame.insert(0, "run_id", record["run_id"])
        frame.insert(1, "gate", record["gate"])
        frame.insert(2, "phase", record["phase"])
        frame.insert(3, "label", record.get("label", ""))
        build_frames.append(frame)
    builds = pd.concat(build_frames, ignore_index=True) if build_frames else _empty_builds()
    return runs, builds


def summarize_metrics(
    runs: pd.DataFrame, builds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-gate run and per-extension cache summaries."""
    run_groups = ["gate", "phase", "label"]
    runs = runs.assign(
        successful_wall_seconds=runs["wall_seconds"].where(
            runs["status"].eq("success")
        ),
        observed_build_seconds=runs["build_seconds"].where(
            runs["build_seconds"].gt(0)
        ),
        successful_unattributed_seconds=runs["unattributed_seconds"].where(
            runs["status"].eq("success")
        ),
    )
    run_summary = (
        runs.groupby(run_groups, as_index=False, dropna=False)
        .agg(
            runs=("run_id", "count"),
            failures=("status", lambda values: values.ne("success").sum()),
            success_rate=("status", lambda values: values.eq("success").mean()),
            median_wall_seconds=("wall_seconds", "median"),
            p90_wall_seconds=("wall_seconds", lambda values: values.quantile(0.9)),
            median_success_wall_seconds=("successful_wall_seconds", "median"),
            median_startup_seconds=("remote_start_seconds", "median"),
            median_observed_build_seconds=("observed_build_seconds", "median"),
            median_success_unattributed_seconds=(
                "successful_unattributed_seconds",
                "median",
            ),
            median_log_bytes=("log_bytes", "median"),
        )
        .sort_values(run_groups)
    )
    if builds.empty:
        return run_summary, _empty_build_summary()
    builds = builds.assign(
        cache_state=builds["cache_hit"]
        .map({True: "hit", False: "miss"})
        .fillna("unknown")
    )
    build_summary = (
        builds.groupby(
            ["gate", "phase", "label", "prefix", "cache_state"],
            as_index=False,
            dropna=False,
        )
        .agg(
            loads=("event", "count"),
            median_load_seconds=("duration_seconds", "median"),
            p90_load_seconds=("duration_seconds", lambda values: values.quantile(0.9)),
            median_dependency_count=("dependency_count", "median"),
        )
        .sort_values(["gate", "phase", "label", "prefix", "cache_state"])
    )
    return run_summary, build_summary


def render_report(run_summary: pd.DataFrame, build_summary: pd.DataFrame) -> str:
    """Render a compact reviewer-first Markdown report."""
    sections = ["# CUTLASS development-loop metrics", "", "## Runs", ""]
    sections.extend(
        [
            "```text",
            run_summary.to_string(
                index=False, float_format=lambda value: f"{value:.3f}"
            ),
            "```",
        ]
    )
    sections.extend(["", "## Extension loads", ""])
    if build_summary.empty:
        sections.append("No extension-load events were recorded.")
    else:
        sections.extend(
            [
                "```text",
                build_summary.to_string(
                    index=False, float_format=lambda value: f"{value:.3f}"
                ),
                "```",
            ]
        )
    sections.extend(
        [
            "",
            "Startup is measured from local invocation to receipt of an explicit remote-start event.",
            "Time to the first event is retained in each raw record when a gate lacks that marker.",
            "Unattributed time is a residual and is not assigned a causal explanation.",
            "",
        ]
    )
    return "\n".join(sections)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metrics-dir",
        default="benchmarking/modal-results/cutlass/dev-metrics",
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--label")
    return parser.parse_args()


def _empty_builds() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "run_id",
            "gate",
            "phase",
            "label",
            "event",
            "prefix",
            "cache_hit",
            "duration_seconds",
            "dependency_count",
        )
    )


def _empty_build_summary() -> pd.DataFrame:
    return pd.DataFrame(
        columns=(
            "gate",
            "phase",
            "label",
            "prefix",
            "cache_state",
            "loads",
            "median_load_seconds",
            "p90_load_seconds",
            "median_dependency_count",
        )
    )


if __name__ == "__main__":
    main()

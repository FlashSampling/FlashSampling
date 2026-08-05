#!/usr/bin/env python3
"""Build one CUTLASS experiment once, then fan out its consumers."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    args = _parse_args()
    raise SystemExit(run_workflow(args.variant, args.label))


def run_workflow(variant: str, label: str) -> int:
    """Run the cache-fill barrier and launch consumers only after it passes."""
    workflow_id = _workflow_id(label)
    commands = workflow_commands(variant, workflow_id)
    result_dir = Path("benchmarking/modal-results/cutlass/experiments") / variant
    result_dir.mkdir(parents=True, exist_ok=True)
    record_path = result_dir / f"workflow-{workflow_id}.json"
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    record = {
        "schema_version": 1,
        "workflow_id": workflow_id,
        "variant": variant,
        "started_at": started_at.isoformat(),
        "commands": commands,
        "build_exit_code": None,
        "consumer_exit_codes": {},
    }
    build = subprocess.run(commands["build"], check=False)
    record["build_exit_code"] = build.returncode
    if build.returncode != 0:
        return _finish_record(record, record_path, start, build.returncode)

    processes = {
        name: subprocess.Popen(command, start_new_session=True)
        for name, command in commands.items()
        if name != "build"
    }
    try:
        for name, process in processes.items():
            record["consumer_exit_codes"][name] = process.wait()
    except KeyboardInterrupt:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for name, process in processes.items():
            record["consumer_exit_codes"][name] = process.wait()
        return _finish_record(record, record_path, start, 130)
    exit_code = next(
        (
            code
            for code in record["consumer_exit_codes"].values()
            if code != 0
        ),
        0,
    )
    return _finish_record(record, record_path, start, exit_code)


def workflow_commands(variant: str, workflow_id: str) -> dict[str, list[str]]:
    """Return the exact single-writer and consumer commands for one workflow."""
    common = [f"CUTLASS_VARIANT={variant}"]
    return {
        "build": [
            "make",
            "modal-cutlass",
            "GATE=gumbel-experiment-build",
            *common,
            f"CUTLASS_DEV_LABEL={workflow_id}-build",
        ],
        "timing": [
            "make",
            "modal-cutlass",
            "GATE=gumbel-experiment",
            *common,
            f"CUTLASS_DEV_LABEL={workflow_id}-timing",
        ],
        "ncu-d4096": [
            "make",
            "modal-cutlass",
            "GATE=gumbel-experiment-ncu",
            *common,
            "CUTLASS_HIDDEN_SIZE=4096",
            f"CUTLASS_DEV_LABEL={workflow_id}-ncu-d4096",
        ],
        "ncu-d8192": [
            "make",
            "modal-cutlass",
            "GATE=gumbel-experiment-ncu",
            *common,
            "CUTLASS_HIDDEN_SIZE=8192",
            f"CUTLASS_DEV_LABEL={workflow_id}-ncu-d8192",
        ],
    }


def _finish_record(record: dict, path: Path, start: float, exit_code: int) -> int:
    record["ended_at"] = datetime.now(timezone.utc).isoformat()
    record["wall_seconds"] = time.perf_counter() - start
    record["status"] = "success" if exit_code == 0 else "failed"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"CUTLASS experiment workflow: {path}", flush=True)
    return exit_code


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--label", default="")
    return parser.parse_args()


def _workflow_id(label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    parts = [timestamp]
    if label:
        parts.append(_slug(label))
    parts.append(suffix)
    return "-".join(parts)


def _slug(value: str) -> str:
    return "".join(
        character if character.isalnum() else "-" for character in value
    ).strip("-")


if __name__ == "__main__":
    main()

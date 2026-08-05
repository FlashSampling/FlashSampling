#!/usr/bin/env python3
"""Build one CUTLASS experiment once, then fan out its consumers."""

import argparse
import json
import os
import signal
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fused_mm_sampling.cutlass_experiments import (
    CUTLASS_PROFILE_CONFIG_MENU as PROFILE_CONFIG_MENU,
)


def main() -> None:
    args = _parse_args()
    try:
        profile_configs = parse_profile_configs(args.profile_configs)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    raise SystemExit(run_workflow(args.variant, args.label, profile_configs))


def run_workflow(
    variant: str,
    label: str,
    profile_configs: tuple[tuple[int, int], ...] = (),
) -> int:
    """Run the cache-fill barrier and launch consumers only after it passes."""
    workflow_id = _workflow_id(label)
    commands = workflow_commands(variant, workflow_id, profile_configs)
    result_dir = Path("benchmarking/modal-results/cutlass/experiments") / variant
    result_dir.mkdir(parents=True, exist_ok=True)
    record_path = result_dir / f"workflow-{workflow_id}.json"
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    record = {
        "schema_version": 1,
        "workflow_id": workflow_id,
        "variant": variant,
        "profile_configs": [
            {"hidden_size": hidden_size, "n_hidden_states": n_hidden_states}
            for hidden_size, n_hidden_states in profile_configs
        ],
        "started_at": started_at.isoformat(),
        "commands": commands,
        "build_exit_code": None,
        "consumer_exit_codes": {},
    }
    build = subprocess.run(commands["build"], check=False)
    record["build_exit_code"] = build.returncode
    if build.returncode != 0:
        return _finish_record(record, record_path, start, build.returncode)

    timing = subprocess.run(commands["timing"], check=False)
    record["consumer_exit_codes"]["timing"] = timing.returncode
    if timing.returncode != 0:
        return _finish_record(record, record_path, start, timing.returncode)

    processes = {}
    try:
        for name, command in commands.items():
            if name not in {"build", "timing"}:
                processes[name] = subprocess.Popen(command, start_new_session=True)
        for name, process in processes.items():
            record["consumer_exit_codes"][name] = process.wait()
    except (KeyboardInterrupt, OSError) as error:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
        for name, process in processes.items():
            record["consumer_exit_codes"][name] = process.wait()
        record["launch_error"] = f"{type(error).__name__}: {error}"
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else 1
        return _finish_record(record, record_path, start, exit_code)
    exit_code = next(
        (
            code
            for code in record["consumer_exit_codes"].values()
            if code != 0
        ),
        0,
    )
    return _finish_record(record, record_path, start, exit_code)


def workflow_commands(
    variant: str,
    workflow_id: str,
    profile_configs: tuple[tuple[int, int], ...] = (),
) -> dict[str, list[str]]:
    """Return the exact single-writer and consumer commands for one workflow."""
    common = [f"CUTLASS_VARIANT={variant}"]
    commands = {
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
    }
    for hidden_size, n_hidden_states in profile_configs:
        name = f"ncu-d{hidden_size}-h{n_hidden_states}"
        commands[name] = [
            "make",
            "modal-cutlass",
            "GATE=gumbel-experiment-ncu",
            *common,
            f"CUTLASS_HIDDEN_SIZE={hidden_size}",
            f"CUTLASS_N_HIDDEN_STATES={n_hidden_states}",
            f"CUTLASS_DEV_LABEL={workflow_id}-{name}",
        ]
    return commands


def parse_profile_configs(value: str) -> tuple[tuple[int, int], ...]:
    """Parse an explicit JSON list of named profile configurations."""
    if not value.strip():
        return ()
    try:
        items = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid profile configuration JSON: {error.msg}") from error
    if not isinstance(items, list):
        raise ValueError("Profile configurations must be a JSON list")
    configs = []
    required_fields = {"hidden_size", "n_hidden_states"}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(
                f"Profile configuration {index} must be a JSON object"
            )
        if set(item) != required_fields:
            raise ValueError(
                f"Profile configuration {index} must contain exactly "
                f"{sorted(required_fields)}"
            )
        if any(type(item[field]) is not int for field in required_fields):
            raise ValueError(
                f"Profile configuration {index} fields must be integers"
            )
        hidden_size = item["hidden_size"]
        n_hidden_states = item["n_hidden_states"]
        config = (hidden_size, n_hidden_states)
        if config not in PROFILE_CONFIG_MENU:
            menu = _profile_config_menu_json()
            raise ValueError(
                f"Unknown profile configuration {item!r}; choose from {menu}"
            )
        if config not in configs:
            configs.append(config)
    return tuple(configs)


def _profile_config_menu_json() -> str:
    return json.dumps(
        [
            {"hidden_size": hidden_size, "n_hidden_states": n_hidden_states}
            for hidden_size, n_hidden_states in PROFILE_CONFIG_MENU
        ],
        separators=(",", ":"),
    )


def _finish_record(record: dict, path: Path, start: float, exit_code: int) -> int:
    ended_at = datetime.now(timezone.utc)
    started_at = datetime.fromisoformat(record["started_at"])
    record["ended_at"] = ended_at.isoformat()
    record["wall_seconds"] = (ended_at - started_at).total_seconds()
    record["observer_active_seconds"] = time.perf_counter() - start
    record["status"] = "success" if exit_code == 0 else "failed"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"CUTLASS experiment workflow: {path}", flush=True)
    return exit_code


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument(
        "--profile-configs",
        default="",
        help="JSON list of named configurations; empty runs no NCU profiles",
    )
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

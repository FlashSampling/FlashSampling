#!/usr/bin/env python3
"""Run one CUTLASS Modal gate while recording development-loop friction."""

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

EVENT_PREFIX = "FMMS_DEV_EVENT "
SCHEMA_VERSION = 2
SENSITIVE_ARGUMENT_FRAGMENTS = ("token", "key", "secret", "password")


def main() -> None:
    args = _parse_args()
    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("A command is required after --")
    raise SystemExit(run_observed_command(args, command))


def run_observed_command(args, command: list[str]) -> int:
    """Stream a subprocess, collect structured events, and persist one record."""
    run_id = _run_id(args.gate, args.label)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    start = time.perf_counter()
    events = []
    first_event_seconds = None
    remote_start_seconds = None
    interrupted = False
    child_env = _child_environment(run_id)
    process = None
    exit_code = 1
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
            start_new_session=True,
        )
        assert process.stdout is not None
        with log_path.open("w") as log_file:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_file.write(line)
                event = _parse_event(line)
                if event is not None:
                    if first_event_seconds is None:
                        first_event_seconds = time.perf_counter() - start
                    if event.get("event") == "remote_start" and remote_start_seconds is None:
                        remote_start_seconds = time.perf_counter() - start
                    events.append(event)
        exit_code = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGINT)
            exit_code = process.wait()
    finally:
        ended_at = datetime.now(timezone.utc)
        observer_active_seconds = time.perf_counter() - start
        wall_seconds = (ended_at - started_at).total_seconds()
        record = _build_record(
            args=args,
            command=command,
            run_id=run_id,
            started_at=started_at,
            ended_at=ended_at,
            wall_seconds=wall_seconds,
            observer_active_seconds=observer_active_seconds,
            exit_code=exit_code,
            interrupted=interrupted,
            first_event_seconds=first_event_seconds,
            remote_start_seconds=remote_start_seconds,
            events=events,
            log_path=log_path,
        )
        metrics_path = metrics_dir / f"{run_id}.json"
        metrics_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"CUTLASS development metrics: {metrics_path}", flush=True)
    return exit_code


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate", required=True)
    parser.add_argument("--phase", default="default")
    parser.add_argument("--label", default="")
    parser.add_argument("--log", required=True)
    parser.add_argument(
        "--metrics-dir",
        default="benchmarking/modal-results/cutlass/dev-metrics",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def _build_record(
    *,
    args,
    command,
    run_id,
    started_at,
    ended_at,
    wall_seconds,
    observer_active_seconds,
    exit_code,
    interrupted,
    first_event_seconds,
    remote_start_seconds,
    events,
    log_path,
) -> dict:
    build_events = [event for event in events if event.get("event") == "extension_load"]
    stage_events = [
        event
        for event in events
        if event.get("event") == "stage_end"
        and event.get("accounting", True)
        and "duration_seconds" in event
    ]
    build_seconds = sum(float(event["duration_seconds"]) for event in build_events)
    stage_seconds = sum(float(event["duration_seconds"]) for event in stage_events)
    startup_seconds = remote_start_seconds or 0.0
    known_seconds = startup_seconds + build_seconds + stage_seconds
    raw_unattributed_seconds = wall_seconds - known_seconds
    git = _git_metadata()
    result_dir = log_path.parent
    artifact_files = [path for path in result_dir.rglob("*") if path.is_file()]
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "gate": args.gate,
        "phase": _effective_phase(args.phase, command),
        "label": args.label,
        "command": _redact_command(command),
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "wall_seconds": wall_seconds,
        "observer_active_seconds": observer_active_seconds,
        "observer_wall_minus_active_seconds": wall_seconds - observer_active_seconds,
        "first_event_seconds": first_event_seconds,
        "remote_start_seconds": remote_start_seconds,
        "build_seconds": build_seconds,
        "stage_seconds": stage_seconds,
        "unattributed_seconds": raw_unattributed_seconds,
        "accounting_consistent": raw_unattributed_seconds >= -0.05,
        "exit_code": exit_code,
        "status": "interrupted" if interrupted else ("success" if exit_code == 0 else "failed"),
        "git": git,
        "log_path": str(log_path),
        "log_bytes": log_path.stat().st_size if log_path.exists() else 0,
        "result_dir_file_count": len(artifact_files),
        "result_dir_bytes": sum(path.stat().st_size for path in artifact_files),
        "events": events,
    }


def _git_metadata() -> dict:
    status = _git_output(["status", "--short"]).splitlines()
    return {
        "commit": _git_output(["rev-parse", "HEAD"]),
        "branch": _git_output(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "changed_path_count": len(status),
    }


def _child_environment(run_id: str) -> dict[str, str]:
    environment = os.environ.copy()
    source_dir = str(Path(__file__).resolve().parents[1] / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        os.pathsep.join((source_dir, existing_pythonpath))
        if existing_pythonpath
        else source_dir
    )
    environment["FMMS_DEV_METRICS"] = "1"
    environment["FMMS_DEV_RUN_ID"] = run_id
    return environment


def _git_output(arguments: list[str]) -> str:
    return subprocess.run(
        ["git", *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _parse_event(line: str) -> dict | None:
    position = line.find(EVENT_PREFIX)
    if position < 0:
        return None
    try:
        event = json.loads(line[position + len(EVENT_PREFIX) :])
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def _effective_phase(default: str, command: list[str]) -> str:
    for index, argument in enumerate(command):
        if argument == "--phase" and index + 1 < len(command):
            return command[index + 1]
        if argument.startswith("--phase="):
            return argument.partition("=")[2]
    return default


def _redact_command(command: list[str]) -> list[str]:
    redacted = []
    redact_next = False
    for argument in command:
        lowered = argument.lower()
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue
        if any(fragment in lowered for fragment in SENSITIVE_ARGUMENT_FRAGMENTS):
            if "=" in argument:
                redacted.append(argument.partition("=")[0] + "=<redacted>")
            else:
                redacted.append(argument)
                redact_next = argument.startswith("-")
            continue
        redacted.append(argument)
    return redacted


def _run_id(gate: str, label: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    parts = [timestamp, _slug(gate)]
    if label:
        parts.append(_slug(label))
    parts.append(suffix)
    return "-".join(parts)


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


if __name__ == "__main__":
    main()

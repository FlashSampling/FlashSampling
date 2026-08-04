import argparse
import json
import os
import sys
from pathlib import Path

import pytest
from fused_mm_sampling.cutlass_build import (
    ExtensionBuildSpec,
    broad_source_fingerprint,
    discover_local_dependencies,
    extension_fingerprint,
)
from fused_mm_sampling.cutlass_experiments import (
    CUTLASS_SAMPLING_EXPERIMENTS,
    get_cutlass_sampling_experiment,
)
from fused_mm_sampling.dev_metrics import EVENT_PREFIX, emit_dev_event, timed_dev_stage

from benchmarking.cutlass_dev_metrics import load_metrics, summarize_metrics
from benchmarking.cutlass_dev_run import (
    _child_environment,
    _redact_command,
    run_observed_command,
)


def test_precise_fingerprint_ignores_unrelated_files(tmp_path):
    root = _source_tree(tmp_path)
    before = extension_fingerprint(_spec(root))
    (root / "unrelated.cu").write_text("changed\n")
    after = extension_fingerprint(_spec(root))
    assert before == after
    assert before.dependencies == ("dependency.cuh", "entry.cu", "toolchain.patch")


def test_legacy_fingerprint_changes_for_unrelated_files(tmp_path):
    root = _source_tree(tmp_path)
    before = broad_source_fingerprint(root)
    (root / "unrelated.cu").write_text("changed\n")
    after = broad_source_fingerprint(root)
    assert before.digest != after.digest


@pytest.mark.parametrize(
    ("mutation", "replacement"),
    (
        ("dependency.cuh", "constexpr int value = 2;\n"),
        ("toolchain.patch", "different patch\n"),
    ),
)
def test_precise_fingerprint_changes_for_file_inputs(tmp_path, mutation, replacement):
    root = _source_tree(tmp_path)
    before = extension_fingerprint(_spec(root))
    (root / mutation).write_text(replacement)
    after = extension_fingerprint(_spec(root))
    assert before.digest != after.digest


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("cuda_flags", ("-O0",)),
        ("architecture", "90"),
        ("toolchain_identity", "cutlass-other"),
        ("python_abi", "cpython-other"),
        ("torch_version", "other"),
        ("cuda_version", "other"),
    ),
)
def test_precise_fingerprint_changes_for_non_file_inputs(tmp_path, field, replacement):
    root = _source_tree(tmp_path)
    spec = _spec(root)
    before = extension_fingerprint(spec)
    values = {name: getattr(spec, name) for name in spec.__dataclass_fields__}
    values[field] = replacement
    after = extension_fingerprint(ExtensionBuildSpec(**values))
    assert before.digest != after.digest


def test_dependency_discovery_rejects_missing_local_include(tmp_path):
    source = tmp_path / "entry.cu"
    source.write_text('#include "missing.cuh"\n')
    with pytest.raises(FileNotFoundError, match="missing.cuh"):
        discover_local_dependencies(tmp_path, (source,))


def test_dependency_discovery_ignores_cutlass_external_includes(tmp_path):
    source = tmp_path / "entry.cu"
    source.write_text('#include "cute/tensor.hpp"\n#include "cutlass/cutlass.h"\n')
    assert discover_local_dependencies(tmp_path, (source,)) == ("entry.cu",)


def test_production_dependency_audit():
    root = Path("src/fused_mm_sampling/csrc/cutlass").resolve()
    sources = (root / "greedy_provider.cu", root / "winning_schedule_provider.cu")
    closure = discover_local_dependencies(root, sources)
    assert closure == (
        "evt_candidates.cu",
        "greedy_provider.cu",
        "stateless_philox.cuh",
        "winning_schedule_provider.cu",
    )
    assert len(broad_source_fingerprint(root).dependencies) == 14


def test_sampling_experiment_registry_is_bounded_and_compositional():
    assert tuple(CUTLASS_SAMPLING_EXPERIMENTS) == (
        "warpgroup-fastlog-smem",
        "warpgroup-fastmath-smem",
        "warpgroup-fastmath",
    )
    spill_free = get_cutlass_sampling_experiment("warpgroup-fastlog-smem")
    combined = get_cutlass_sampling_experiment("warpgroup-fastmath-smem")
    fastmath = get_cutlass_sampling_experiment("warpgroup-fastmath")
    shared_flags = {
        "-DFMMS_WARPGROUP_REDUCTION",
        "-DFMMS_INLINE_GUMBEL",
        "-DFMMS_FAST_LOG",
    }
    assert shared_flags.issubset(spill_free.cuda_flags)
    assert shared_flags.issubset(combined.cuda_flags)
    assert shared_flags.issubset(fastmath.cuda_flags)
    assert "-DFMMS_WARPGROUP_SMEM_STAGE" in spill_free.cuda_flags
    assert "-DFMMS_WARPGROUP_SMEM_STAGE" in combined.cuda_flags
    assert "-DFMMS_FAST_DIV" in combined.cuda_flags
    assert "-DFMMS_FAST_DIV" in fastmath.cuda_flags


def test_sampling_experiment_registry_rejects_unknown_variant():
    with pytest.raises(ValueError, match="Unknown CUTLASS sampling experiment"):
        get_cutlass_sampling_experiment("rejected-one-off")


def test_dev_events_are_opt_in_and_record_failures(monkeypatch, capsys):
    emit_dev_event("disabled")
    assert capsys.readouterr().out == ""
    monkeypatch.setenv("FMMS_DEV_METRICS", "1")
    with pytest.raises(RuntimeError):
        with timed_dev_stage("test"):
            raise RuntimeError("expected")
    lines = capsys.readouterr().out.splitlines()
    events = [json.loads(line.removeprefix(EVENT_PREFIX)) for line in lines]
    assert [event["event"] for event in events] == ["stage_start", "stage_end"]
    assert events[-1]["status"] == "error"
    assert events[-1]["error_type"] == "RuntimeError"


def test_observed_runner_streams_logs_and_records_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _initialize_git_repo(tmp_path)
    log_path = tmp_path / "results" / "log.txt"
    metrics_dir = tmp_path / "metrics"
    args = argparse.Namespace(
        gate="test-gate",
        phase="default",
        label="fixture",
        log=str(log_path),
        metrics_dir=str(metrics_dir),
    )
    event = {
        "event": "extension_load",
        "prefix": "fixture",
        "cache_hit": True,
        "duration_seconds": 0.25,
        "dependency_count": 3,
    }
    command = [
        sys.executable,
        "-c",
        f"print('hello'); print({(EVENT_PREFIX + json.dumps(event))!r})",
        "--phase",
        "smoke",
    ]
    assert run_observed_command(args, command) == 0
    record_path = next(metrics_dir.glob("*.json"))
    record = json.loads(record_path.read_text())
    assert record["status"] == "success"
    assert record["phase"] == "smoke"
    assert record["build_seconds"] == pytest.approx(0.25)
    assert "hello" in log_path.read_text()


def test_observed_runner_preserves_failure_exit_code(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _initialize_git_repo(tmp_path)
    args = argparse.Namespace(
        gate="test-gate",
        phase="default",
        label="failure",
        log=str(tmp_path / "results" / "log.txt"),
        metrics_dir=str(tmp_path / "metrics"),
    )
    assert run_observed_command(args, [sys.executable, "-c", "raise SystemExit(7)"]) == 7
    record = json.loads(next((tmp_path / "metrics").glob("*.json")).read_text())
    assert record["status"] == "failed"
    assert record["exit_code"] == 7


def test_metrics_summary_handles_runs_without_build_events(tmp_path):
    records = (
        _run_record("one", "success", 10.0, []),
        _run_record(
            "two",
            "failed",
            20.0,
            [
                {
                    "event": "extension_load",
                    "prefix": "sampling",
                    "cache_hit": False,
                    "duration_seconds": 5.0,
                    "dependency_count": 6,
                }
            ],
        ),
    )
    for record in records:
        (tmp_path / f"{record['run_id']}.json").write_text(json.dumps(record))
    runs, builds = load_metrics(tmp_path)
    run_summary, build_summary = summarize_metrics(runs, builds)
    assert run_summary.iloc[0]["success_rate"] == pytest.approx(0.5)
    assert build_summary.iloc[0]["cache_state"] == "miss"


def test_command_redaction():
    assert _redact_command(["command", "--token", "private", "--api-key=value", "visible"]) == [
        "command",
        "--token",
        "<redacted>",
        "--api-key=<redacted>",
        "visible",
    ]


def test_child_environment_prefers_current_worktree(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")
    environment = _child_environment("run")
    assert environment["PYTHONPATH"].split(os.pathsep)[0].endswith("/src")
    assert environment["FMMS_DEV_METRICS"] == "1"


def _source_tree(tmp_path: Path) -> Path:
    (tmp_path / "entry.cu").write_text('#include "dependency.cuh"\n#include "cute/tensor.hpp"\n')
    (tmp_path / "dependency.cuh").write_text("constexpr int value = 1;\n")
    (tmp_path / "toolchain.patch").write_text("patch\n")
    (tmp_path / "unrelated.cu").write_text("original\n")
    return tmp_path


def _spec(root: Path) -> ExtensionBuildSpec:
    return ExtensionBuildSpec(
        prefix="test",
        source_root=root,
        sources=(root / "entry.cu",),
        cuda_flags=("-O3",),
        architecture="100",
        toolchain_identity="cutlass-sha",
        python_abi="cpython-test",
        torch_version="torch-test",
        cuda_version="cuda-test",
        supplemental_inputs=(root / "toolchain.patch",),
    )


def _initialize_git_repo(path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "tracked").write_text("fixture")
    subprocess.run(["git", "add", "tracked"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)


def _run_record(run_id: str, status: str, wall_seconds: float, events: list[dict]) -> dict:
    return {
        "run_id": run_id,
        "gate": "gate",
        "phase": "phase",
        "label": "label",
        "status": status,
        "wall_seconds": wall_seconds,
        "first_event_seconds": 1.0,
        "remote_start_seconds": 1.0,
        "build_seconds": 0.0,
        "stage_seconds": 0.0,
        "unattributed_seconds": wall_seconds - 1.0,
        "log_bytes": 10,
        "events": events,
        "git": {"commit": "sha", "branch": "branch", "dirty": False},
        "command": ["command"],
    }

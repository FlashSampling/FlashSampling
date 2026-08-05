"""Build-only CUTLASS compilation study with durable compiler traces."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from ...cutlass_compile_studies import CutlassCompileStudy, get_cutlass_compile_study
from ...cutlass_experiments import get_cutlass_sampling_experiment
from ...dev_metrics import emit_dev_event, timed_dev_stage
from ..utils import (
    commit_shared_volume,
    make_app,
    make_volumes,
    reload_shared_volume,
    set_volume_caches,
    volume_path,
)
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(
    make_cutlass_provider_image().apt_install(
        "cuda-ctadvisor-13-0",
        "cuda-ctadvisor-13-2",
        "cuda-cudart-dev-13-2",
        "cuda-nvcc-13-2",
        "ccache",
        "libcublas-dev-13-2",
        "libcusolver-dev-13-2",
        "libcusparse-dev-13-2",
    )
)
VARIANT = "warpgroup-fastmath-smem"


@app.function(cpu=4, image=image, timeout=10 * 60)
def trace_smoke() -> None:
    """Isolate CUDA device-trace behavior before an expensive CUTLASS build."""
    cases = {
        "sm100a-default": ("-arch=sm_100a",),
        "sm100a-threads1": ("-arch=sm_100a", "--threads=1"),
        "sm100-default": ("-arch=sm_100",),
        "sm100a-explicit": ("-arch=sm_100a",),
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "trace_smoke.cu"
        source.write_text("__global__ void trace_smoke() {}\n")
        for name, flags in cases.items():
            case_dir = root / name
            case_dir.mkdir()
            output = case_dir / "trace_smoke.o"
            trace_flag = "--fdevice-time-trace=-"
            if name == "sm100a-explicit":
                trace_flag = f"--fdevice-time-trace={case_dir / 'explicit.json'}"
            completed = subprocess.run(
                [
                    "nvcc",
                    *flags,
                    trace_flag,
                    "-MD",
                    "-MF",
                    str(case_dir / "trace_smoke.d"),
                    "-c",
                    str(source),
                    "-o",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            traces = []
            for trace in sorted(case_dir.glob("*.json")):
                try:
                    json.loads(trace.read_text())
                    valid_json = True
                except json.JSONDecodeError:
                    valid_json = False
                traces.append(
                    {
                        "name": trace.name,
                        "bytes": trace.stat().st_size,
                        "valid_json": valid_json,
                    }
                )
            print(
                json.dumps(
                    {
                        "case": name,
                        "flags": list(flags),
                        "returncode": completed.returncode,
                        "stderr": completed.stderr,
                        "traces": traces,
                    },
                    sort_keys=True,
                )
            )


@app.function(
    gpu="B200",
    cpu=16,
    memory=32_768,
    image=image,
    volumes=make_volumes(),
    timeout=60 * 60,
)
def record_sm100(study_name: str, run_id: str) -> dict:
    study = get_cutlass_compile_study(study_name)
    experiment = get_cutlass_sampling_experiment(VARIANT)
    reload_shared_volume()
    set_volume_caches()
    report_dir = (
        Path(volume_path) / "cutlass-compile-study" / study_name / run_id
    )
    report_dir.mkdir(parents=True, exist_ok=False)
    trace_dir = Path("/tmp/fmms-nvcc-traces") / run_id
    cuda_root = Path(
        "/usr/local/cuda-13.2" if study.device_trace else "/usr/local/cuda-13.0"
    )
    os.environ["CUDA_HOME"] = str(cuda_root)
    os.environ["PATH"] = f"{cuda_root / 'bin'}:{os.environ['PATH']}"
    if study.device_trace:
        _enable_nvcc_trace_wrapper(trace_dir, cuda_root)
    ccache_before = None
    if study.use_ccache:
        _enable_ccache(study, cuda_root)
        ccache_before = _ccache_stats()

    from fused_mm_sampling import cutlass_impl

    emit_dev_event(
        "remote_start",
        stage="compile_study",
        study=study_name,
        variant=VARIANT,
    )
    tools = _tool_versions()
    prefix = f"{experiment.extension_prefix}_{study.extension_suffix}"
    flags = (*experiment.cuda_flags, *study.cuda_flags)
    os.environ["FMMS_CUTLASS_VERBOSE"] = "1"
    extension_load_start = time.perf_counter()
    module = None
    extension_error = None
    try:
        module = cutlass_impl._load_sampling_extension(
            prefix,
            flags,
            architecture_flags=study.architecture_flags,
        )
    except RuntimeError as error:
        if not study.device_trace:
            raise
        extension_error = f"{type(error).__name__}: {error}"
    extension_load_seconds = time.perf_counter() - extension_load_start
    if module is not None:
        build_dir = Path(module.__file__).parent
    else:
        build_roots = sorted(
            Path(os.environ["TORCH_EXTENSIONS_DIR"]).glob(f"{prefix}_*"),
            key=lambda path: path.stat().st_mtime_ns,
        )
        if not build_roots:
            raise RuntimeError(f"No failed trace build directory found for {prefix}")
        build_dir = build_roots[-1]
    if study.device_trace:
        for trace in trace_dir.glob("*.json"):
            shutil.copy2(trace, build_dir / trace.name)
    ccache_after = _ccache_stats() if study.use_ccache else None
    if ccache_before is not None:
        (report_dir / "ccache-before.json").write_text(
            json.dumps(ccache_before, indent=2, sort_keys=True) + "\n"
        )
        (report_dir / "ccache-after.json").write_text(
            json.dumps(ccache_after, indent=2, sort_keys=True) + "\n"
        )
    trace_files = sorted(build_dir.rglob("*.json"))
    if trace_files:
        with timed_dev_stage("compile_trace_analysis", study=study_name):
            advisor = subprocess.run(
                ["ctadvisor", "-path", str(build_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        advisor_output = advisor.stdout + advisor.stderr
        advisor_status = "complete"
    else:
        advisor_output = (
            "Compile Time Advisor not run because this timing comparison uses "
            "NVCC --time=- instead of device trace JSON.\n"
        )
        advisor_status = "not_applicable"
    (report_dir / "ctadvisor.txt").write_text(advisor_output)
    artifacts = _copy_build_artifacts(build_dir, report_dir)
    summary = {
        "schema_version": 1,
        "study": study_name,
        "variant": VARIANT,
        "run_id": run_id,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "requested_cpu_cores": study.cpu_cores,
        "device_trace": study.device_trace,
        "ccache_enabled": study.use_ccache,
        "ccache_before": ccache_before,
        "ccache_after": ccache_after,
        "visible_cpu_count": os.cpu_count(),
        "cpu_affinity_count": len(os.sched_getaffinity(0)),
        "cuda_flags": list(flags),
        "architecture_flags": list(study.architecture_flags),
        "extension_prefix": prefix,
        "binary_path": str(module.__file__) if module is not None else None,
        "build_dir": str(build_dir),
        "extension_load_seconds": extension_load_seconds,
        "extension_load_status": "success" if module is not None else "trace_only",
        "extension_error": extension_error,
        "ninja_build_steps": _read_ninja_log(build_dir / ".ninja_log"),
        "instrumentation": "NVCC phase CSV on retained process output via --time=-",
        "ctadvisor_status": advisor_status,
        "tools": tools,
        "artifacts": artifacts,
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    with timed_dev_stage("volume_commit", study=study_name):
        commit_shared_volume()
    return summary | {
        "ctadvisor": advisor_output,
        "report_dir": str(report_dir),
    }


@app.local_entrypoint()
def main(study: str, output_dir: str) -> None:
    get_cutlass_compile_study(study)
    run_id = os.environ.get("FMMS_DEV_RUN_ID")
    if not run_id:
        raise RuntimeError("FMMS_DEV_RUN_ID is required for compile-study artifacts")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result = record_sm100.remote(study, run_id)
    advisor = result.pop("ctadvisor")
    (output_path / "ctadvisor.txt").write_text(advisor)
    (output_path / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(advisor, end="" if advisor.endswith("\n") else "\n")
    print(f"CUTLASS compile-study artifacts: {result['report_dir']}")


def _tool_versions() -> dict[str, str]:
    commands = {
        "nvcc": ["nvcc", "--version"],
        "ctadvisor": ["ctadvisor", "--version"],
        "ninja": ["ninja", "--version"],
    }
    versions = {}
    for name, command in commands.items():
        executable = shutil.which(command[0])
        if executable is None:
            raise RuntimeError(f"Required compile-study tool is missing: {command[0]}")
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        versions[name] = (completed.stdout + completed.stderr).strip()
    return versions


def _enable_nvcc_trace_wrapper(trace_dir: Path, real_cuda_root: Path) -> None:
    trace_dir.mkdir(parents=True, exist_ok=False)
    cuda_root = Path("/tmp/fmms-cuda-trace-wrapper")
    wrapper = cuda_root / "bin" / "nvcc"
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).parents[2] / "nvcc_trace_wrapper.py"
    shutil.copy2(source, wrapper)
    wrapper.chmod(0o755)
    for name in ("include", "lib64"):
        destination = cuda_root / name
        if not destination.exists():
            destination.symlink_to(real_cuda_root / name)
    os.environ["CUDA_HOME"] = str(cuda_root)
    os.environ["FMMS_REAL_NVCC"] = str(real_cuda_root / "bin" / "nvcc")
    os.environ["FMMS_NVCC_TRACE_DIR"] = str(trace_dir)
    os.environ["FMMS_NVCC_TRACE_PTX_ONLY"] = "1"


def _enable_ccache(study: CutlassCompileStudy, cuda_root: Path) -> None:
    if not study.build_root_suffix:
        raise ValueError("A ccache study requires a distinct build root suffix")
    os.environ["TORCH_EXTENSIONS_DIR"] = str(
        Path(volume_path)
        / "cache"
        / "torch_extensions-compile-study"
        / study.build_root_suffix
    )
    ccache_dir = (
        Path(volume_path) / "cache" / "ccache" / "cutlass-study-phase3-v1"
    )
    ccache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CCACHE_DIR"] = str(ccache_dir)
    os.environ["CCACHE_NAMESPACE"] = "cutlass-study-phase3-v1"
    os.environ["CCACHE_BASEDIR"] = "/opt/fmms/repo"
    os.environ["CCACHE_COMPILERCHECK"] = "content"
    if study.ccache_probe:
        wrapper_root = Path("/tmp/fmms-ccache-wrapper")
        wrapper = wrapper_root / "bin" / "nvcc"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).parents[2] / "nvcc_ccache_wrapper.py"
        shutil.copy2(source, wrapper)
        wrapper.chmod(0o755)
        header = wrapper_root / "fmms_ccache_probe.h"
        header.write_text(
            "#pragma once\n"
            "#ifndef FMMS_CCACHE_PROBE_VALUE\n"
            "#define FMMS_CCACHE_PROBE_VALUE 1\n"
            "#endif\n"
            "namespace fmms_ccache_probe {\n"
            "inline constexpr int value = FMMS_CCACHE_PROBE_VALUE;\n"
            "}\n"
        )
        os.environ["FMMS_CCACHE_PROBE_MODE"] = study.ccache_probe
        os.environ["FMMS_CCACHE_PROBE_HEADER"] = str(header)
        os.environ["FMMS_REAL_NVCC"] = str(cuda_root / "bin" / "nvcc")
        os.environ["PYTORCH_NVCC"] = str(wrapper)
    else:
        os.environ["PYTORCH_NVCC"] = f"ccache {cuda_root / 'bin' / 'nvcc'}"


def _ccache_stats() -> dict:
    completed = subprocess.run(
        ["ccache", "--print-stats"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        key: int(value)
        for key, value in (
            line.split("\t", maxsplit=1)
            for line in completed.stdout.splitlines()
        )
    }


def _read_ninja_log(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    steps = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        start_ms, end_ms, restat, output, command_hash = line.split("\t")
        steps.append(
            {
                "output": output,
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "duration_seconds": (int(end_ms) - int(start_ms)) / 1_000,
                "restat": int(restat),
                "command_hash": command_hash,
            }
        )
    return steps


def _copy_build_artifacts(build_dir: Path, report_dir: Path) -> list[dict]:
    selected = [build_dir / "build.ninja", build_dir / ".ninja_log"]
    selected.extend(sorted(build_dir.rglob("*.json")))
    artifacts = []
    for source in selected:
        if not source.is_file():
            continue
        relative = source.relative_to(build_dir)
        destination = report_dir / "build" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        artifacts.append(
            {
                "path": str(destination),
                "build_relative_path": str(relative),
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
                "retained_path": str(destination),
            }
        )
    for source in sorted(
        [*build_dir.rglob("*.o"), *build_dir.rglob("*.so")]
    ):
        if source.is_file():
            artifacts.append(
                {
                    "build_relative_path": str(source.relative_to(build_dir)),
                    "bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                    "retained_path": str(source),
                }
            )
    return artifacts


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

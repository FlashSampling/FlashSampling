"""Validate the pinned CUTLASS toolchain on H100 and B200."""

import json
import platform
import subprocess
from pathlib import Path

from .utils import (
    CUTLASS_ROOT,
    CUTLASS_SHA,
    CUTLASS_VERSION,
    PYTORCH_CUDA_IMAGE,
    make_app,
    make_cutlass_image,
)

app = make_app()
image = make_cutlass_image()

SMOKE_SHAPE = {"m": 512, "n": 64, "k": 256, "l": 1}


@app.function(gpu="H100", image=image, timeout=20 * 60)
def smoke_h100() -> None:
    _run_smoke(
        architecture="sm_90a",
        executable=Path(CUTLASS_ROOT) / "build-h100/examples/48_hopper_warp_specialized_gemm/"
        "48_hopper_warp_specialized_gemm",
    )


@app.function(gpu="B200", image=image, timeout=20 * 60)
def smoke_b200() -> None:
    _run_smoke(
        architecture="sm_100a",
        executable=Path(CUTLASS_ROOT)
        / "build-b200/examples/71_blackwell_gemm_with_collective_builder/"
        "71_blackwell_gemm_with_collective_builder",
    )


def _run_smoke(architecture: str, executable: Path) -> None:
    # GPU runtime dependencies stay inside the remote path so Modal can import
    # this submission module on the local CPU-only client.
    import torch

    metadata = {
        "architecture": architecture,
        "base_image": PYTORCH_CUDA_IMAGE,
        "cccl": _preprocessor_macro("CCCL_VERSION"),
        "cutlass.version": CUTLASS_VERSION,
        "cutlass.sha": _command_output(["git", "-C", CUTLASS_ROOT, "rev-parse", "HEAD"]),
        "cuda.runtime": torch.version.cuda,
        "gpu.compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "gpu.name": torch.cuda.get_device_name(),
        "host.compiler": _first_line(["c++", "--version"]),
        "nvcc": _first_line(["nvcc", "--version"], match="release"),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if metadata["cutlass.sha"] != CUTLASS_SHA:
        raise RuntimeError(f"CUTLASS revision mismatch: {metadata['cutlass.sha']} != {CUTLASS_SHA}")

    print("TOOLCHAIN_METADATA")
    print(json.dumps(metadata, indent=2, sort_keys=True))
    command = [str(executable), *(f"--{key}={value}" for key, value in SMOKE_SHAPE.items())]
    print("SMOKE_COMMAND")
    print(" ".join(command))
    subprocess.run(command, check=True)


def _command_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True).strip()


def _first_line(command: list[str], match: str | None = None) -> str:
    lines = _command_output(command).splitlines()
    if match is not None:
        lines = [line for line in lines if match in line]
    return lines[0].strip()


def _preprocessor_macro(name: str) -> str:
    macros = _command_output(
        [
            "c++",
            "-E",
            "-dM",
            "-x",
            "c++",
            "-I",
            "/usr/local/cuda/include/cccl",
            "-include",
            "cuda/version",
            "/dev/null",
        ]
    )
    prefix = f"#define {name} "
    for line in macros.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix)
    raise RuntimeError(f"{name} not found in the CUDA toolkit's CCCL headers")


@app.local_entrypoint()
def main() -> None:
    smoke_h100.remote()
    smoke_b200.remote()

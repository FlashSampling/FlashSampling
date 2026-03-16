import torch
import triton


def gather_system_metadata() -> dict:
    return {
        "gpu_name": get_gpu_name(),
        "device_count": torch.cuda.device_count(),
        "python.version": _python_version(),
        "torch.version": torch.__version__,
        "triton.version": triton.__version__,
        "torch.version.cuda": torch.version.cuda,
        "nvcc.version": _system_cuda_version(),
    }


def get_gpu_name() -> str:
    return torch.cuda.get_device_name()


def _python_version() -> str:
    import sys

    return sys.version.split()[0]


def _system_cuda_version() -> str | None:
    import subprocess

    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        for line in out.splitlines():
            if "release" in line:
                return line.split("release")[-1].split(",")[0].strip()
    except FileNotFoundError:
        pass
    return None

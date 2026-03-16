import os
import shutil
import subprocess


def _find_cuda_home() -> str | None:
    """Find a CUDA toolkit installation with nvcc that supports the current GPU."""
    # Respect explicit CUDA_HOME
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]

    # Check /usr/local/cuda symlink (standard CUDA toolkit location)
    if os.path.isfile("/usr/local/cuda/bin/nvcc"):
        return "/usr/local/cuda"

    # Search /usr/local/cuda-* directories (prefer highest version)
    import glob

    candidates = sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True)
    for nvcc_path in candidates:
        cuda_home = os.path.dirname(os.path.dirname(nvcc_path))
        return cuda_home

    # Fall back to system nvcc location
    nvcc = shutil.which("nvcc")
    if nvcc:
        real = os.path.realpath(nvcc)
        bin_dir = os.path.dirname(real)
        return os.path.dirname(bin_dir)

    return None


def _check_nvcc_supports_gpu(cuda_home: str) -> None:
    """Raise if the nvcc in cuda_home is too old for the current GPU."""
    nvcc = os.path.join(cuda_home, "bin", "nvcc")
    if not os.path.isfile(nvcc):
        return
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        sm = f"compute_{major}{minor}"
        result = subprocess.run(
            [nvcc, f"-arch={sm}", "-x", "cu", "/dev/null", "-o", "/dev/null"],
            capture_output=True,
            text=True,
        )
        if "Unsupported gpu architecture" in result.stderr:
            raise RuntimeError(
                f"nvcc at {nvcc} does not support this GPU (sm_{major}{minor}). "
                f"Please install a newer CUDA toolkit (>= 12.0 for H100) and set "
                f"CUDA_HOME to its path, e.g.: export CUDA_HOME=/usr/local/cuda-12.2"
            )
    except ImportError:
        pass


cuda_home = _find_cuda_home()
if cuda_home is None:
    raise RuntimeError(
        "nvcc not found. Please install the CUDA toolkit and ensure nvcc is on PATH, "
        "or set the CUDA_HOME environment variable. "
        "Example: export CUDA_HOME=/usr/local/cuda-12.2"
    )

_check_nvcc_supports_gpu(cuda_home)
os.environ.setdefault("CUDA_HOME", cuda_home)

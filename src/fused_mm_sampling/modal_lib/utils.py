import os
from pathlib import Path

import modal
from pydantic_settings import BaseSettings

_repo_root = Path(__file__).resolve().parents[3]


class ModalEnvConfig(BaseSettings):
    """Common env-var config shared by Modal benchmark scripts.

    Fields are read from same-named env vars (case-insensitive).
    Subclass to add script-specific fields or override defaults.
    """

    gpu: str = "b200"
    n_procs: int = 1
    name: str | None = None
    n_hidden_states: int = 1
    case: str = "small"

    timeout: int = 20 * 60

    @property
    def gpu_spec(self) -> str:
        return f"{self.gpu}:{self.n_procs}" if self.n_procs > 1 else self.gpu


def make_app():
    return modal.App("fused-matmul-sample")


def make_image():
    img = modal.Image.from_registry("pytorch/pytorch:2.10.0-cuda13.0-cudnn9-devel")
    deps = [
        "flashinfer-python",
        "pandas",
        "pydantic-settings",
        "matplotlib",
        "nvtx",
        "llnl-hatchet",
        "scipy",
        "cuda-bench[cu13]",
        "cupti-python",
    ]
    return img.uv_pip_install(deps)


volume_path = "/vol-fused-mm-sample"


def make_volumes():
    return {volume_path: modal.Volume.from_name("fused-mm-sample")}


def set_volume_caches():
    """Point XDG_CACHE_HOME to the Modal volume so caches (Triton, flashinfer,
    torch.compile, etc.) persist across runs."""
    os.environ["XDG_CACHE_HOME"] = f"{volume_path}/cache"


def add_library_code(image: modal.Image) -> modal.Image:
    """Add the fused_mm_sampling source and benchmarking scripts to the image,
    then pip-install the package so subprocess-based tools (e.g. ncu) can import it."""
    return (
        image.add_local_dir(
            str(_repo_root / "src"),
            remote_path="/opt/fmms/src",
            copy=True,
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_file(
            str(_repo_root / "pyproject.toml"),
            remote_path="/opt/fmms/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            str(_repo_root / "benchmarking" / "speed_test.py"),
            remote_path="/opt/fmms/speed_test.py",
            copy=True,
        )
        .run_commands("cd /opt/fmms && pip install --break-system-packages -e .")
    )

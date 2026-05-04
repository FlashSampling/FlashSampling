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
    bench_fn: str = "fi-cupti"

    timeout: int = 20 * 60

    @property
    def gpu_spec(self) -> str:
        return f"{self.gpu}:{self.n_procs}" if self.n_procs > 1 else self.gpu


def make_app():
    return modal.App("fused-matmul-sample")


def make_image():
    # Pytorch 2.11.0 ships triton 3.6.0, whose bundled `ptxas-blackwell`
    # supports sm_103a (B300). We install deps via `uv pip install --system`
    # (mirroring `modal_vllm_benchmark.py`) so uv preserves the base image's
    # pre-installed torch 2.11.0+cu130 and triton 3.6.0 instead of re-resolving
    # them from PyPI (which yields torch 2.9.1+cu128 + triton 3.5.1 without
    # ptxas-blackwell).
    deps = " ".join(
        [
            "flashinfer-python",
            "pandas",
            "pydantic-settings",
            "matplotlib",
            "nvtx",
            "llnl-hatchet",
            "scipy",
            "'cuda-bench[cu13]'",
            "cupti-python",
        ]
    )
    return (
        modal.Image.from_registry("pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel")
        .run_commands("pip install --break-system-packages uv")
        .run_commands(f"uv pip install --system {deps}")
    )


volume_path = "/vol-fused-mm-sample"


def make_volumes():
    return {volume_path: modal.Volume.from_name("fused-mm-sample")}


def set_volume_caches():
    """Point cache env vars to the Modal volume and enable Triton autotune logging.

    XDG_CACHE_HOME: used by flashinfer, torch.compile, etc.
    TRITON_CACHE_DIR: used by Triton for compiled kernels and autotune results.
    Triton ignores XDG_CACHE_HOME and reads TRITON_CACHE_DIR (or TRITON_HOME) instead.
    TRITON_PRINT_AUTOTUNING: surfaces autotune progress so silent waits
    (cold cache, hangs) are debuggable from the run log.
    """
    os.environ["XDG_CACHE_HOME"] = f"{volume_path}/cache"
    os.environ["TRITON_CACHE_DIR"] = f"{volume_path}/cache/triton"
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"


def add_library_code(image: modal.Image) -> modal.Image:
    """Add the fused_mm_sampling source and benchmarking scripts to the image,
    then pip-install the package so subprocess-based tools (e.g. ncu) can import it.

    Layer order matters for caching: pyproject.toml (rarely changes) and dep
    install go first so that source-only changes don't re-run pip.
    """
    return (
        image
        # 1. Install deps (cached as long as pyproject.toml is unchanged).
        .add_local_file(
            str(_repo_root / "pyproject.toml"),
            remote_path="/opt/fmms/pyproject.toml",
            copy=True,
        )
        .run_commands(
            "mkdir -p /opt/fmms/src/fused_mm_sampling"
            " && touch /opt/fmms/src/fused_mm_sampling/__init__.py"
            " && cd /opt/fmms && pip install --break-system-packages -e ."
        )
        # 2. Copy source files (changes frequently, but deps layer is cached).
        #    The editable install points to /opt/fmms/src, so the real files
        #    are picked up at runtime.
        .add_local_dir(
            str(_repo_root / "src"),
            remote_path="/opt/fmms/src",
            copy=True,
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_file(
            str(_repo_root / "benchmarking" / "speed_test.py"),
            remote_path="/opt/fmms/speed_test.py",
            copy=True,
        )
        .add_local_file(
            str(_repo_root / "benchmarking" / "nsys_wrapper.py"),
            remote_path="/opt/fmms/nsys_wrapper.py",
            copy=True,
        )
    )

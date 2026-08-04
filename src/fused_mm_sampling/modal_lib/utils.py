import os
from pathlib import Path

import modal
from pydantic_settings import BaseSettings

_repo_root = Path(__file__).resolve().parents[3]

PYTORCH_CUDA_IMAGE = "pytorch/pytorch:2.11.0-cuda13.0-cudnn9-devel"


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
    deps: list[str] = [
        "flashinfer-python",
        "pandas",
        "pydantic-settings",
        "matplotlib",
        "nvtx",
        "llnl-hatchet",
        "scipy",
        "tqdm",
        "'cuda-bench[cu13]'",
        "cupti-python",
    ]
    deps_str: str = " ".join(deps)
    return (
        modal.Image.from_registry(PYTORCH_CUDA_IMAGE)
        .apt_install("numactl")
        .run_commands("pip install --break-system-packages uv")
        .run_commands(f"uv pip install --system {deps_str}")
    )


VLLM_FORK_BRANCH = "feature/fmms-sampler"
VLLM_FORK_SHA = "7a74973e4dc727df979f2a5ec9fff64ac5319467"
# Upstream-main parent of the latest main-to-feature merge in VLLM_FORK_SHA.
# Determine it with:
# merge=$(git rev-list --first-parent --merges -n 1 "$VLLM_FORK_SHA")
# git show -s --format='%P' "$merge"  # use the second parent
VLLM_PRECOMPILED_WHEEL_SHA = "1a2c17634eccc4e68d9e1ab654f702d55361c754"


def make_vllm_image() -> modal.Image:
    return (
        # Base image must match vLLM's pinned torch version (see requirements/cuda.txt
        # in the fork: torch==2.11.0). cuda13.0 is the newest CUDA tag pytorch/pytorch
        # publishes for 2.11.0. Because vLLM pins torch without a local version, uv
        # treats the base image's torch==2.11.0+cu130 as satisfying the pin and skips
        # reinstalling, so we keep cu13 throughout and the precompiled .so stays ABI-
        # compatible.
        modal.Image.from_registry(PYTORCH_CUDA_IMAGE)
        .apt_install("git", "curl")
        .run_commands("pip install --break-system-packages uv")
        .run_commands(
            f"git clone --depth 1 -b {VLLM_FORK_BRANCH}"
            " https://github.com/tomasruizt/vllm.git /opt/vllm"
            " && cd /opt/vllm"
            f" && git fetch --depth 1 origin {VLLM_FORK_SHA}"
            f" && git checkout {VLLM_FORK_SHA}",
            # Pre-install numpy at a vllm[bench]-compatible version before
            # installing vllm. The base image ships numpy 2.4.3, but
            # vllm[bench]'s deps (numba, mistral-common) need numpy<2.3.
            # If we let uv downgrade during vllm install, it leaves a
            # broken `numpy-2.4.3.dist-info` entry that makes
            # importlib.metadata.version("numpy") return None, which
            # aborts the vllm CLI on import.
            "pip install --break-system-packages 'numpy<2.3'",
            # Install build dependencies into the base environment because the
            # precompiled build below deliberately disables isolation.
            "uv pip install --system -r /opt/vllm/requirements/build.txt",
            # Let vLLM's resolver pick the torch version that matches the precompiled .so.
            # Earlier we pinned torch==2.10.0 to match a 2.10.0+cu130 .so, but the upstream
            # precompiled wheel is now built against torch 2.11.0, so any pin breaks the ABI.
            "cd /opt/vllm"
            f" && VLLM_PRECOMPILED_WHEEL_COMMIT={VLLM_PRECOMPILED_WHEEL_SHA}"
            " VLLM_USE_PRECOMPILED=1 uv pip install"
            " --system --no-build-isolation '.[bench]'",
            "test -f /usr/local/lib/python3.12/dist-packages/vllm/_C.abi3.so",
        )
        .add_local_dir(
            str(_repo_root / "src"),
            remote_path="/opt/fused-mm-sample/src",
            copy=True,
            ignore=["__pycache__", "*.pyc"],
        )
        .add_local_file(
            str(_repo_root / "pyproject.toml"),
            remote_path="/opt/fused-mm-sample/pyproject.toml",
            copy=True,
        )
        .add_local_file(
            str(_repo_root / "README.md"), remote_path="/opt/fused-mm-sample/README.md", copy=True
        )
        .run_commands(
            "uv pip install --system tabulate /opt/fused-mm-sample",
        )
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
        .add_local_file(
            str(_repo_root / "benchmarking" / "memory_traffic.py"),
            remote_path="/opt/fmms/memory_traffic.py",
            copy=True,
        )
    )


def make_ncu_image(*, include_library_code: bool = True) -> modal.Image:
    image = make_image().run_commands(
        "apt-get update && apt-get install -y cuda-nsight-compute-13-2",
        "ln -sf /opt/nvidia/nsight-compute/2026.1.0/ncu /usr/local/bin/ncu",
    )
    return add_library_code(image) if include_library_code else image

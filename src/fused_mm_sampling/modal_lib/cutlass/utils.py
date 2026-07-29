"""Pinned CUTLASS image and gate-specific binary builders."""

from pathlib import Path

import modal

from ..utils import PYTORCH_CUDA_IMAGE

_repo_root = Path(__file__).resolve().parents[4]

CUTLASS_VERSION = "4.2.1"
CUTLASS_SHA = "f3fde58372d33e9a5650ba7b80fc48b3b49d40c8"
CUTLASS_ROOT = "/opt/cutlass"


def make_cutlass_image() -> modal.Image:
    """Build the pinned CUTLASS toolchain and ordinary-GEMM smoke binaries."""
    return (
        modal.Image.from_registry(PYTORCH_CUDA_IMAGE)
        .apt_install("cmake", "git", "ninja-build")
        .run_commands("pip install --break-system-packages pandas pydantic-settings")
        .run_commands(
            f"git clone https://github.com/NVIDIA/cutlass.git {CUTLASS_ROOT}",
            f"cd {CUTLASS_ROOT} && git checkout --detach {CUTLASS_SHA}",
            f'test "$(cd {CUTLASS_ROOT} && git rev-parse HEAD)" = "{CUTLASS_SHA}"',
            f"cmake -S {CUTLASS_ROOT} -B {CUTLASS_ROOT}/build-h100 -G Ninja"
            " -DCUTLASS_NVCC_ARCHS=90a"
            " -DCUTLASS_ENABLE_TESTS=OFF"
            " -DCUTLASS_ENABLE_EXAMPLES=ON",
            f"cmake --build {CUTLASS_ROOT}/build-h100"
            " --target 48_hopper_warp_specialized_gemm --parallel 2",
            f"cmake -S {CUTLASS_ROOT} -B {CUTLASS_ROOT}/build-b200 -G Ninja"
            " -DCUTLASS_NVCC_ARCHS=100a"
            " -DCUTLASS_ENABLE_TESTS=OFF"
            " -DCUTLASS_ENABLE_EXAMPLES=ON",
            f"cmake --build {CUTLASS_ROOT}/build-b200"
            " --target 71_blackwell_gemm_with_collective_builder --parallel 2",
        )
    )


def add_cutlass_thread_local_max(image: modal.Image) -> modal.Image:
    """Add the Gate 1b thread-local max-with-index binaries."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    return (
        image.add_local_file(
            str(csrc_root / "max_with_index.cuh"),
            remote_path="/opt/fmms/max_with_index.cuh",
            copy=True,
        )
        .add_local_file(
            str(csrc_root / "thread_local_max.cu"),
            remote_path="/opt/fmms/thread_local_max.cu",
            copy=True,
        )
        .run_commands(
            "nvcc -std=c++17 -O2 -arch=sm_90a -DFMMS_ARCH_SM90"
            " -I/opt/fmms"
            " /opt/fmms/thread_local_max.cu"
            " -o /opt/fmms/cutlass_thread_local_max_sm90",
            "nvcc -std=c++17 -O2 -arch=sm_100a -DFMMS_ARCH_SM100"
            " -I/opt/fmms"
            " /opt/fmms/thread_local_max.cu"
            " -o /opt/fmms/cutlass_thread_local_max_sm100",
        )
    )


def add_cutlass_warp_max(image: modal.Image) -> modal.Image:
    """Add the Gate 1c warp-local max-with-index binaries."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    return (
        image.add_local_file(
            str(csrc_root / "max_with_index.cuh"),
            remote_path="/opt/fmms/max_with_index.cuh",
            copy=True,
        )
        .add_local_file(
            str(csrc_root / "warp_max.cu"),
            remote_path="/opt/fmms/warp_max.cu",
            copy=True,
        )
        .run_commands(
            "nvcc -std=c++17 -O2 -arch=sm_90a -DFMMS_ARCH_SM90"
            " -I/opt/fmms /opt/fmms/warp_max.cu"
            " -o /opt/fmms/cutlass_warp_max_sm90",
            "nvcc -std=c++17 -O2 -arch=sm_100a -DFMMS_ARCH_SM100"
            " -I/opt/fmms /opt/fmms/warp_max.cu"
            " -o /opt/fmms/cutlass_warp_max_sm100",
        )
    )


def add_cutlass_cta_max(image: modal.Image) -> modal.Image:
    """Add the Gate 1d CTA-local max-with-index binaries."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    return (
        image.add_local_file(
            str(csrc_root / "max_with_index.cuh"),
            remote_path="/opt/fmms/max_with_index.cuh",
            copy=True,
        )
        .add_local_file(
            str(csrc_root / "cta_max.cu"),
            remote_path="/opt/fmms/cta_max.cu",
            copy=True,
        )
        .run_commands(
            "nvcc -std=c++17 -O2 -arch=sm_90a -DFMMS_ARCH_SM90"
            " -I/opt/fmms /opt/fmms/cta_max.cu"
            " -o /opt/fmms/cutlass_cta_max_sm90",
            "nvcc -std=c++17 -O2 -arch=sm_100a -DFMMS_ARCH_SM100"
            " -I/opt/fmms /opt/fmms/cta_max.cu"
            " -o /opt/fmms/cutlass_cta_max_sm100",
        )
    )


def add_cutlass_cta_multi_column_max(image: modal.Image) -> modal.Image:
    """Add the Gate 1e multi-column CTA max-with-index binaries."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    return (
        image.add_local_file(
            str(csrc_root / "max_with_index.cuh"),
            remote_path="/opt/fmms/max_with_index.cuh",
            copy=True,
        )
        .add_local_file(
            str(csrc_root / "cta_multi_column_max.cu"),
            remote_path="/opt/fmms/cta_multi_column_max.cu",
            copy=True,
        )
        .run_commands(
            "nvcc -std=c++17 -O2 -lineinfo"
            " -arch=sm_90a -DFMMS_ARCH_SM90"
            " -I/opt/fmms /opt/fmms/cta_multi_column_max.cu"
            " -o /opt/fmms/cutlass_cta_multi_column_max_sm90",
            "nvcc -std=c++17 -O2 -lineinfo"
            " -arch=sm_100a -DFMMS_ARCH_SM100"
            " -I/opt/fmms /opt/fmms/cta_multi_column_max.cu"
            " -o /opt/fmms/cutlass_cta_multi_column_max_sm100",
        )
    )


def add_cutlass_cta_boundary_max(image: modal.Image) -> modal.Image:
    """Add the Gate 1f boundary-predicated CTA max binaries."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    return (
        image.add_local_file(
            str(csrc_root / "max_with_index.cuh"),
            remote_path="/opt/fmms/max_with_index.cuh",
            copy=True,
        )
        .add_local_file(
            str(csrc_root / "cta_boundary_max.cu"),
            remote_path="/opt/fmms/cta_boundary_max.cu",
            copy=True,
        )
        .run_commands(
            "nvcc -std=c++17 -O2 -lineinfo"
            " -arch=sm_90a -DFMMS_ARCH_SM90"
            " -I/opt/fmms /opt/fmms/cta_boundary_max.cu"
            " -o /opt/fmms/cutlass_cta_boundary_max_sm90",
            "nvcc -std=c++17 -O2 -lineinfo"
            " -arch=sm_100a -DFMMS_ARCH_SM100"
            " -I/opt/fmms /opt/fmms/cta_boundary_max.cu"
            " -o /opt/fmms/cutlass_cta_boundary_max_sm100",
        )
    )


def add_cutlass_evt_candidates(image: modal.Image) -> modal.Image:
    """Add the Gate 1g GEMM EVT candidate binaries."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    return (
        image.add_local_file(
            str(csrc_root / "evt_candidates.cu"),
            remote_path="/opt/fmms/evt_candidates.cu",
            copy=True,
        )
        .run_commands(
            f"nvcc -std=c++17 -O2 -lineinfo --expt-relaxed-constexpr"
            " -arch=sm_90a -DFMMS_ARCH_SM90"
            f" -I{CUTLASS_ROOT}/include -I{CUTLASS_ROOT}/tools/util/include"
            " /opt/fmms/evt_candidates.cu"
            " -o /opt/fmms/cutlass_evt_candidates_sm90",
            f"nvcc -std=c++17 -O2 -lineinfo --expt-relaxed-constexpr"
            " -arch=sm_100a -DFMMS_ARCH_SM100"
            f" -I{CUTLASS_ROOT}/include -I{CUTLASS_ROOT}/tools/util/include"
            " /opt/fmms/evt_candidates.cu"
            " -o /opt/fmms/cutlass_evt_candidates_sm100",
        )
    )


def add_cutlass_stage2(image: modal.Image) -> modal.Image:
    """Add the Gate 1h GEMM EVT plus GPU Stage 2 binaries."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    return (
        image.add_local_file(
            str(csrc_root / "evt_candidates.cu"),
            remote_path="/opt/fmms/evt_candidates.cu",
            copy=True,
        )
        .run_commands(
            f"nvcc -std=c++17 -O2 -lineinfo --expt-relaxed-constexpr"
            " -arch=sm_90a -DFMMS_ARCH_SM90 -DFMMS_GATE_STAGE2"
            f" -I{CUTLASS_ROOT}/include -I{CUTLASS_ROOT}/tools/util/include"
            " /opt/fmms/evt_candidates.cu"
            " -o /opt/fmms/cutlass_stage2_sm90",
            f"nvcc -std=c++17 -O2 -lineinfo --expt-relaxed-constexpr"
            " -arch=sm_100a -DFMMS_ARCH_SM100 -DFMMS_GATE_STAGE2"
            f" -I{CUTLASS_ROOT}/include -I{CUTLASS_ROOT}/tools/util/include"
            " /opt/fmms/evt_candidates.cu"
            " -o /opt/fmms/cutlass_stage2_sm100",
        )
    )


def add_cutlass_accumulator_layout(image: modal.Image) -> modal.Image:
    """Add the Gate 1a accumulator-layout diagnostic binaries."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    return (
        image.add_local_file(
            str(csrc_root / "accumulator_layout.cu"),
            remote_path="/opt/fmms/accumulator_layout.cu",
            copy=True,
        )
        .run_commands(
            f"nvcc -std=c++17 -O2 --expt-relaxed-constexpr"
            " -arch=sm_90a -DFMMS_ARCH_SM90"
            f" -I{CUTLASS_ROOT}/include -I{CUTLASS_ROOT}/tools/util/include"
            " /opt/fmms/accumulator_layout.cu"
            " -o /opt/fmms/cutlass_accumulator_layout_sm90",
            f"nvcc -std=c++17 -O2 --expt-relaxed-constexpr"
            " -arch=sm_100a -DFMMS_ARCH_SM100"
            f" -I{CUTLASS_ROOT}/include -I{CUTLASS_ROOT}/tools/util/include"
            " /opt/fmms/accumulator_layout.cu"
            " -o /opt/fmms/cutlass_accumulator_layout_sm100",
        )
    )

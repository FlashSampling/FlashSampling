"""Pinned CUTLASS image and gate-specific binary builders."""

from pathlib import Path

import modal

from ..utils import PYTORCH_CUDA_IMAGE, make_image

_repo_root = Path(__file__).resolve().parents[4]

CUTLASS_VERSION = "4.6.1"
CUTLASS_SHA = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
CUTLASS_ROOT = "/opt/cutlass"
NVMMH_VERSION = "0.1.0.27"
HEURISTICS_BUILD_DIR = f"{CUTLASS_ROOT}/build-heuristics-b200"
HEURISTICS_TESTLIST = "/opt/fmms/heuristics-testlist.csv"
HEURISTICS_BUILD_DIR_N256 = f"{CUTLASS_ROOT}/build-heuristics-b200-n256"
HEURISTICS_TESTLIST_N256 = "/opt/fmms/heuristics-testlist-n256.csv"


def make_cutlass_heuristics_image() -> modal.Image:
    """Build the Gate 2c heuristic kernel-discovery toolchain.

    Generates B200 kernel candidates with nvidia-matmul-heuristics at image
    build time (the builders have no GPU, so the heuristics GPU is pinned to
    B200) and compiles cutlass_profiler with the emitted kernel set.
    """
    problems_json = (
        _repo_root
        / "src/fused_mm_sampling/modal_lib/cutlass/gemm_problems_b200.json"
    )
    problems_n256_json = (
        _repo_root
        / "src/fused_mm_sampling/modal_lib/cutlass/gemm_problems_b200_n256.json"
    )
    return (
        modal.Image.from_registry(PYTORCH_CUDA_IMAGE)
        .apt_install("cmake", "git", "ninja-build")
        .run_commands(
            "pip install --break-system-packages pandas"
            f" nvidia-matmul-heuristics=={NVMMH_VERSION}",
            "pip show nvidia-matmul-heuristics | grep -E '^(Name|Version)'",
        )
        .run_commands(
            f"git clone https://github.com/NVIDIA/cutlass.git {CUTLASS_ROOT}",
            f"cd {CUTLASS_ROOT} && git checkout --detach {CUTLASS_SHA}",
            f'test "$(cd {CUTLASS_ROOT} && git rev-parse HEAD)" = "{CUTLASS_SHA}"',
        )
        .add_local_file(
            str(problems_json),
            remote_path="/opt/fmms/gemm_problems_b200.json",
            copy=True,
        )
        .run_commands(
            f"cmake -S {CUTLASS_ROOT} -B {HEURISTICS_BUILD_DIR} -G Ninja"
            " -DCUTLASS_NVCC_ARCHS=100a"
            " -DCUTLASS_ENABLE_TESTS=OFF"
            " -DCUTLASS_ENABLE_EXAMPLES=OFF"
            " -DCUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE="
            "/opt/fmms/gemm_problems_b200.json"
            " -DCUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM=16"
            f" -DCUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE={HEURISTICS_TESTLIST}"
            " -DCUTLASS_LIBRARY_HEURISTICS_GPU=B200",
            f"cmake --build {HEURISTICS_BUILD_DIR}"
            " --target cutlass_profiler --parallel 8",
            f"test -x {HEURISTICS_BUILD_DIR}/tools/profiler/cutlass_profiler",
            f"test -s {HEURISTICS_TESTLIST}",
        )
        # Gate 2c stop-rule expansion: the two N=256 problems that failed the
        # top-16 search get the top-32 heuristic population.
        .add_local_file(
            str(problems_n256_json),
            remote_path="/opt/fmms/gemm_problems_b200_n256.json",
            copy=True,
        )
        .run_commands(
            f"cmake -S {CUTLASS_ROOT} -B {HEURISTICS_BUILD_DIR_N256} -G Ninja"
            " -DCUTLASS_NVCC_ARCHS=100a"
            " -DCUTLASS_ENABLE_TESTS=OFF"
            " -DCUTLASS_ENABLE_EXAMPLES=OFF"
            " -DCUTLASS_LIBRARY_HEURISTICS_PROBLEMS_FILE="
            "/opt/fmms/gemm_problems_b200_n256.json"
            " -DCUTLASS_LIBRARY_HEURISTICS_CONFIGS_PER_PROBLEM=32"
            " -DCUTLASS_LIBRARY_HEURISTICS_TESTLIST_FILE="
            f"{HEURISTICS_TESTLIST_N256}"
            " -DCUTLASS_LIBRARY_HEURISTICS_GPU=B200",
            f"cmake --build {HEURISTICS_BUILD_DIR_N256}"
            " --target cutlass_profiler --parallel 8",
            f"test -x {HEURISTICS_BUILD_DIR_N256}/tools/profiler/cutlass_profiler",
            f"test -s {HEURISTICS_TESTLIST_N256}",
        )
        # Runtime-only deps for the Modal submission module. Kept in a
        # trailing layer so fixes here never invalidate the profiler build.
        .run_commands("pip install --break-system-packages pydantic-settings")
    )


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


def make_cutlass_provider_image(
    base_image: modal.Image | None = None,
) -> modal.Image:
    """Layer pinned CUTLASS sources onto the standard FMMS runtime image."""
    void_d_patch = (
        _repo_root
        / "src/fused_mm_sampling/csrc/cutlass/sm100-void-d.patch"
    )
    uint64_reduction_patch = (
        _repo_root
        / "src/fused_mm_sampling/csrc/cutlass/sm90-row-reduction-uint64.patch"
    )
    return (
        (base_image or make_image())
        .apt_install("git", "ninja-build")
        .add_local_file(
            str(void_d_patch),
            remote_path="/opt/fmms/sm100-void-d.patch",
            copy=True,
        )
        .add_local_file(
            str(uint64_reduction_patch),
            remote_path="/opt/fmms/sm90-row-reduction-uint64.patch",
            copy=True,
        )
        .run_commands(
            f"git clone https://github.com/NVIDIA/cutlass.git {CUTLASS_ROOT}",
            f"cd {CUTLASS_ROOT} && git checkout --detach {CUTLASS_SHA}",
            f'test "$(cd {CUTLASS_ROOT} && git rev-parse HEAD)" = "{CUTLASS_SHA}"',
            f"cd {CUTLASS_ROOT} && patch -p1 < /opt/fmms/sm100-void-d.patch",
            f"cd {CUTLASS_ROOT} && git apply /opt/fmms/sm90-row-reduction-uint64.patch",
            "pip install --break-system-packages pytest",
        )
        # Keep run metadata in a trailing layer so instrumentation changes do
        # not invalidate the expensive dependency and CUTLASS checkout layers.
        .env(
            {
                "FMMS_CUTLASS_TOOLCHAIN_ID": f"cutlass-{CUTLASS_SHA}",
                "FMMS_DEV_METRICS": "1",
            }
        )
    )


def _add_cutlass_max_binary(
    image: modal.Image,
    stem: str,
    source_file: str,
    *,
    lineinfo: bool = False,
) -> modal.Image:
    """Compile one max-with-index gate source into the SM90/SM100 binary pair.

    `stem` names the two output binaries (`cutlass_{stem}_sm90`/`_sm100`) and
    `source_file` is the `.cu` in `csrc/cutlass` to compile. The four
    max-hierarchy gates (1b-1f) differ only in this pair.
    """
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    lineinfo_flag = " -lineinfo" if lineinfo else ""
    return (
        image.add_local_file(
            str(csrc_root / "max_with_index.cuh"),
            remote_path="/opt/fmms/max_with_index.cuh",
            copy=True,
        )
        .add_local_file(
            str(csrc_root / "max_harness.h"),
            remote_path="/opt/fmms/max_harness.h",
            copy=True,
        )
        .add_local_file(
            str(csrc_root / source_file),
            remote_path=f"/opt/fmms/{source_file}",
            copy=True,
        )
        .run_commands(
            f"nvcc -std=c++17 -O2{lineinfo_flag}"
            " -arch=sm_90a -DFMMS_ARCH_SM90"
            f" -I/opt/fmms /opt/fmms/{source_file}"
            f" -o /opt/fmms/cutlass_{stem}_sm90",
            f"nvcc -std=c++17 -O2{lineinfo_flag}"
            " -arch=sm_100a -DFMMS_ARCH_SM100"
            f" -I/opt/fmms /opt/fmms/{source_file}"
            f" -o /opt/fmms/cutlass_{stem}_sm100",
        )
    )


def add_cutlass_thread_local_max(image: modal.Image) -> modal.Image:
    """Add the Gate 1b thread-local max-with-index binaries."""
    return _add_cutlass_max_binary(
        image, "thread_local_max", "thread_local_max.cu"
    )


def add_cutlass_warp_max(image: modal.Image) -> modal.Image:
    """Add the Gate 1c warp-local max-with-index binaries."""
    return _add_cutlass_max_binary(image, "warp_max", "warp_max.cu")


def add_cutlass_cta_max(image: modal.Image) -> modal.Image:
    """Add the Gate 1d CTA-local max-with-index binaries."""
    return _add_cutlass_max_binary(image, "cta_max", "cta_max.cu")


def add_cutlass_cta_multi_column_max(image: modal.Image) -> modal.Image:
    """Add the Gate 1e multi-column CTA max-with-index binaries."""
    return _add_cutlass_max_binary(
        image, "cta_multi_column_max", "cta_multi_column_max.cu", lineinfo=True
    )


def add_cutlass_cta_boundary_max(image: modal.Image) -> modal.Image:
    """Add the Gate 1f boundary-predicated CTA max binaries."""
    return _add_cutlass_max_binary(
        image, "cta_boundary_max", "cta_boundary_max.cu", lineinfo=True
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


def add_cutlass_greedy_provider(image: modal.Image) -> modal.Image:
    """Mount Gate 2a sources without invalidating the toolchain image."""
    return (
        image.env(
            {
                "CUTLASS_ROOT": CUTLASS_ROOT,
                "PYTHONPATH": "/opt/fmms/repo/src",
            }
        )
        .add_local_dir(
            str(_repo_root / "src"),
            remote_path="/opt/fmms/repo/src",
            copy=False,
        )
        .add_local_dir(
            str(_repo_root / "tests"),
            remote_path="/opt/fmms/repo/tests",
            copy=False,
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


# The five winning B200 2-SM schedule donors: (name, tile_m, tile_n, tile_k,
# cluster_m). Shared by the accumulator-ownership and fused-EVT gates.
WINNING_VARIANTS = (
    ("128x64x128-c2", 128, 64, 128, 2),
    ("256x128x64-c2", 256, 128, 64, 2),
    ("256x128x64-c4", 256, 128, 64, 4),
    ("256x128x128-c2", 256, 128, 128, 2),
    ("256x256x64-c2", 256, 256, 64, 2),
)


def add_cutlass_winning_schedule_layout(image: modal.Image) -> modal.Image:
    """Add Gate 2d ownership diagnostics for the winning B200 schedules."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    source = "/opt/fmms/accumulator_layout.cu"
    include_flags = (
        f"-I{CUTLASS_ROOT}/include -I{CUTLASS_ROOT}/tools/util/include"
    )
    commands = [
        "nvcc -std=c++17 -O2 --expt-relaxed-constexpr -arch=sm_100a "
        "-DFMMS_ARCH_SM100 -DFMMS_SM100_2SM "
        f"-DFMMS_TILE_M={tile_m} -DFMMS_TILE_N={tile_n} "
        f"-DFMMS_TILE_K={tile_k} -DFMMS_CLUSTER_M={cluster_m} "
        f"{include_flags} {source} -o /opt/fmms/cutlass_winning_layout_{name}"
        for name, tile_m, tile_n, tile_k, cluster_m in WINNING_VARIANTS
    ]
    return image.add_local_file(
        str(csrc_root / "accumulator_layout.cu"),
        remote_path=source,
        copy=True,
    ).run_commands(*commands)


def add_cutlass_winning_schedule_evt(image: modal.Image) -> modal.Image:
    """Add Gate 2d fused-EVT experiments for every winning B200 schedule."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    include_flags = (
        f"-I{CUTLASS_ROOT}/include -I{CUTLASS_ROOT}/tools/util/include"
    )
    commands = [
        "nvcc -std=c++17 -O2 -lineinfo --expt-relaxed-constexpr "
        "-arch=sm_100a -DFMMS_ARCH_SM100 -DFMMS_SM100_2SM "
        "-DFMMS_PER_CTA_CANDIDATES -DFMMS_GATE_STAGE2 "
        f"-DFMMS_TILE_M={tile_m} -DFMMS_TILE_N={tile_n} "
        f"-DFMMS_TILE_K={tile_k} -DFMMS_CLUSTER_M={cluster_m} "
        f"{include_flags} /opt/fmms/evt_candidates.cu "
        f"-o /opt/fmms/cutlass_winning_evt_{name}"
        for name, tile_m, tile_n, tile_k, cluster_m in WINNING_VARIANTS
    ]
    return (
        image.add_local_file(
            str(csrc_root / "evt_candidates.cu"),
            remote_path="/opt/fmms/evt_candidates.cu",
            copy=True,
        )
        .add_local_file(
            str(csrc_root / "sm90-row-reduction-uint64.patch"),
            remote_path="/opt/fmms/sm90-row-reduction-uint64.patch",
            copy=True,
        )
        .run_commands(
            f"cd {CUTLASS_ROOT} && "
            "git apply /opt/fmms/sm90-row-reduction-uint64.patch",
            *commands,
        )
    )


def add_cutlass_stateless_philox(image: modal.Image) -> modal.Image:
    """Add the Gate 3 stateless Philox correctness harness."""
    csrc_root = _repo_root / "src/fused_mm_sampling/csrc/cutlass"
    return (
        image.add_local_file(
            str(csrc_root / "stateless_philox.cuh"),
            remote_path="/opt/fmms/stateless_philox.cuh",
            copy=True,
        )
        .add_local_file(
            str(csrc_root / "stateless_philox.cu"),
            remote_path="/opt/fmms/stateless_philox.cu",
            copy=True,
        )
        .run_commands(
            "nvcc -std=c++17 -O3 -lineinfo -arch=sm_100a "
            "/opt/fmms/stateless_philox.cu "
            "-o /opt/fmms/cutlass_stateless_philox"
        )
    )

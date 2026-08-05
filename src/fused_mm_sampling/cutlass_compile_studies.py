"""CPU-importable configurations for CUTLASS compile-time studies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CutlassCompileStudy:
    extension_suffix: str
    cuda_flags: tuple[str, ...]
    cpu_cores: int
    device_trace: bool = False
    use_ccache: bool = False
    build_root_suffix: str | None = None
    ccache_probe: str | None = None
    architecture_flags: tuple[str, ...] = ("-arch=sm_100a",)
    include_gemm_tuning: bool = False


CUTLASS_COMPILE_STUDIES = {
    "baseline": CutlassCompileStudy(
        extension_suffix="compile_study_baseline_v3",
        cuda_flags=("--time=-",),
        cpu_cores=16,
        include_gemm_tuning=True,
    ),
    "pruned": CutlassCompileStudy(
        extension_suffix="compile_study_pruned_v1",
        cuda_flags=("--time=-",),
        cpu_cores=16,
    ),
    "split4": CutlassCompileStudy(
        extension_suffix="compile_study_split4_v3",
        cuda_flags=("--time=-", "--split-compile=4"),
        cpu_cores=16,
    ),
    "split8": CutlassCompileStudy(
        extension_suffix="compile_study_split8_v3",
        cuda_flags=("--time=-", "--split-compile=8"),
        cpu_cores=16,
    ),
    "sass-only": CutlassCompileStudy(
        extension_suffix="compile_study_sass_only_v1",
        cuda_flags=("--time=-",),
        cpu_cores=16,
        include_gemm_tuning=True,
        architecture_flags=(
            "--generate-code=arch=compute_100a,code=sm_100a",
        ),
    ),
    "sass-pruned": CutlassCompileStudy(
        extension_suffix="compile_study_sass_pruned_v1",
        cuda_flags=("--time=-",),
        cpu_cores=16,
        architecture_flags=(
            "--generate-code=arch=compute_100a,code=sm_100a",
        ),
    ),
    "advisor": CutlassCompileStudy(
        extension_suffix="compile_study_advisor_cuda132_ptx_v4",
        cuda_flags=(),
        cpu_cores=16,
        device_trace=True,
    ),
    "ccache-cold": CutlassCompileStudy(
        extension_suffix="compile_study_ccache_v1",
        cuda_flags=("--time=-",),
        cpu_cores=16,
        use_ccache=True,
        build_root_suffix="ccache-cold-v1",
    ),
    "ccache-exact": CutlassCompileStudy(
        extension_suffix="compile_study_ccache_v1",
        cuda_flags=("--time=-",),
        cpu_cores=16,
        use_ccache=True,
        build_root_suffix="ccache-exact-v1",
    ),
    "ccache-one-tu": CutlassCompileStudy(
        extension_suffix="compile_study_ccache_v1",
        cuda_flags=("--time=-",),
        cpu_cores=16,
        use_ccache=True,
        build_root_suffix="ccache-one-tu-v1",
        ccache_probe="greedy-header",
    ),
    "ccache-feature-flag": CutlassCompileStudy(
        extension_suffix="compile_study_ccache_v1",
        cuda_flags=("--time=-",),
        cpu_cores=16,
        use_ccache=True,
        build_root_suffix="ccache-feature-flag-v1",
        ccache_probe="feature-flag",
    ),
}


def get_cutlass_compile_study(name: str) -> CutlassCompileStudy:
    try:
        return CUTLASS_COMPILE_STUDIES[name]
    except KeyError as error:
        choices = ", ".join(CUTLASS_COMPILE_STUDIES)
        raise ValueError(
            f"Unknown CUTLASS compile study {name!r}. Choose one of: {choices}"
        ) from error

"""CPU-importable registry for compile-time CUTLASS sampling experiments."""

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class CutlassSamplingExperiment:
    extension_prefix: str
    cuda_flags: tuple[str, ...]


_WARPGROUP = ("-DFMMS_WARPGROUP_REDUCTION",)
_INLINE = (*_WARPGROUP, "-DFMMS_INLINE_GUMBEL")
_FAST_LOG = (*_INLINE, "-DFMMS_FAST_LOG")
_FAST_MATH = (*_FAST_LOG, "-DFMMS_FAST_DIV")

CUTLASS_SAMPLING_EXPERIMENTS = {
    "warpgroup-fastlog-smem": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastlog_smem_sm100",
        (*_FAST_LOG, "-DFMMS_WARPGROUP_SMEM_STAGE"),
    ),
    "warpgroup-fastmath-smem": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_smem_sm100",
        (*_FAST_MATH, "-DFMMS_WARPGROUP_SMEM_STAGE"),
    ),
    "warpgroup-fastmath": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_sm100",
        _FAST_MATH,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", required=True)
    args = parser.parse_args()
    try:
        get_cutlass_sampling_experiment(args.validate)
    except ValueError as error:
        parser.error(str(error))


def get_cutlass_sampling_experiment(variant: str) -> CutlassSamplingExperiment:
    try:
        return CUTLASS_SAMPLING_EXPERIMENTS[variant]
    except KeyError as error:
        choices = ", ".join(CUTLASS_SAMPLING_EXPERIMENTS)
        raise ValueError(
            f"Unknown CUTLASS sampling experiment {variant!r}. Choose one of: {choices}"
        ) from error


if __name__ == "__main__":
    main()

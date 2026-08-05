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
    "warpgroup-fastmath-smem-2wg-striped": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_smem_2wg_striped_sm100",
        (
            *_FAST_MATH,
            "-DFMMS_WARPGROUP_SMEM_STAGE",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS_STRIPED",
        ),
    ),
    "warpgroup-fastmath-smem-4wg-striped": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_smem_4wg_striped_sm100",
        (
            *_FAST_MATH,
            "-DFMMS_WARPGROUP_SMEM_STAGE",
            "-DFMMS_FOUR_EPILOGUE_WARPGROUPS",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS_STRIPED",
        ),
    ),
    "warpgroup-fastmath-smem-2wg-partitioned": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_smem_2wg_partitioned_sm100",
        (
            *_FAST_MATH,
            "-DFMMS_WARPGROUP_SMEM_STAGE",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS_STRIPED",
            "-DFMMS_PARTITIONED_TMEM_LOAD",
        ),
    ),
    "warpgroup-fastmath-smem-4wg-partitioned": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_smem_4wg_partitioned_sm100",
        (
            *_FAST_MATH,
            "-DFMMS_WARPGROUP_SMEM_STAGE",
            "-DFMMS_FOUR_EPILOGUE_WARPGROUPS",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS_STRIPED",
            "-DFMMS_PARTITIONED_TMEM_LOAD",
        ),
    ),
    "warpgroup-fastmath-2wg-partitioned": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_2wg_partitioned_sm100",
        (
            *_FAST_MATH,
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS_STRIPED",
            "-DFMMS_PARTITIONED_TMEM_LOAD",
        ),
    ),
    "warpgroup-fastmath-4wg-partitioned": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_4wg_partitioned_sm100",
        (
            *_FAST_MATH,
            "-DFMMS_FOUR_EPILOGUE_WARPGROUPS",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS_STRIPED",
            "-DFMMS_PARTITIONED_TMEM_LOAD",
        ),
    ),
    "warpgroup-2wg-partitioned": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_2wg_partitioned_sm100",
        (
            *_INLINE,
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS_STRIPED",
            "-DFMMS_PARTITIONED_TMEM_LOAD",
        ),
    ),
    "warpgroup-4wg-partitioned": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_4wg_partitioned_sm100",
        (
            *_INLINE,
            "-DFMMS_FOUR_EPILOGUE_WARPGROUPS",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS_STRIPED",
            "-DFMMS_PARTITIONED_TMEM_LOAD",
        ),
    ),
    "warpgroup-fastmath-smem-fragment4": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_smem_fragment4_sm100",
        (
            *_FAST_MATH,
            "-DFMMS_WARPGROUP_SMEM_STAGE",
            "-DFMMS_FRAGMENT_SIZE_4",
        ),
    ),
    "warpgroup-fastmath-smem-fragment8": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_smem_fragment8_sm100",
        (
            *_FAST_MATH,
            "-DFMMS_WARPGROUP_SMEM_STAGE",
            "-DFMMS_FOUR_EPILOGUE_WARPGROUPS",
            "-DFMMS_TWO_EPILOGUE_WARPGROUPS_STRIPED",
            "-DFMMS_FRAGMENT_SIZE_8",
        ),
    ),
    "warpgroup-fastmath": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_fastmath_sm100",
        _FAST_MATH,
    ),
    "warpgroup": CutlassSamplingExperiment(
        "fmms_cutlass_sampling_warpgroup_sm100",
        _INLINE,
    ),
}

CUTLASS_PROFILE_CONFIG_MENU = (
    (4_096, 128),
    (4_096, 256),
    (8_192, 128),
    (8_192, 256),
)


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

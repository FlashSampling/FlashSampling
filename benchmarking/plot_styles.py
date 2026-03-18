"""Shared color palette, markers, and hatches for benchmark plots."""

from fused_mm_sampling.alg_names import (  # noqa: F401 (re-exported)
    FLASHSAMPLING_RENAMES,
    LongNames,
)

L = LongNames

# Consistent color palette: FMMS stands out, baselines are muted.
PROVIDER_COLORS: dict[str, str] = {
    L.fmms_triton: "#d62728",  # bold red
    L.fmms_helion: "#e45756",  # lighter red
    L.fmms_greedy_triton: "#ff7f0e",  # orange
    L.multinomial_sampling_compiled: "#7f7f7f",  # gray
    L.multinomial_sampling_eager: "#bcbd22",  # olive
    L.flashinfer_top_k_top_p_sampling_from_logits: "#1f77b4",  # muted blue
    L.flashinfer_sampling_from_logits: "#aec7e8",  # light blue
    L.greedy_baseline: "#2ca02c",  # green
}

# Distinct markers so lines are distinguishable without color.
PROVIDER_MARKERS: dict[str, str] = {
    L.fmms_triton: "o",
    L.fmms_helion: "D",
    L.fmms_greedy_triton: "^",
    L.multinomial_sampling_compiled: "s",
    L.multinomial_sampling_eager: "P",
    L.flashinfer_top_k_top_p_sampling_from_logits: "X",
    L.flashinfer_sampling_from_logits: "v",
    L.greedy_baseline: "d",
}

# Hatch patterns for bar plots.
PROVIDER_HATCHES: dict[str, str] = {
    L.fmms_triton: "",
    L.fmms_helion: "//",
    L.fmms_greedy_triton: "\\\\",
    L.multinomial_sampling_compiled: "///",
    L.multinomial_sampling_eager: "\\\\",
    L.flashinfer_top_k_top_p_sampling_from_logits: "xxx",
    L.flashinfer_sampling_from_logits: "...",
    L.greedy_baseline: "++",
}

# Make the FlashSampling names point to the same colors, markers, and hatches as the FMMS names.
for _mapping in [PROVIDER_COLORS, PROVIDER_MARKERS, PROVIDER_HATCHES]:
    for _old_key, _new_key in FLASHSAMPLING_RENAMES.items():
        _mapping[_new_key] = _mapping[_old_key]

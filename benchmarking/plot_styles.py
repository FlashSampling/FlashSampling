"""Shared color palette, markers, and hatches for benchmark plots."""

# Consistent color palette: FMMS stands out, baselines are muted.
PROVIDER_COLORS: dict[str, str] = {
    "FMMS (Triton)": "#d62728",  # bold red
    "FMMS (Helion)": "#e45756",  # lighter red
    "FMMS (Triton NoNoise)": "#ff7f0e",  # orange
    "Multinomial Sampling (Compiled)": "#7f7f7f",  # gray
    "Multinomial Sampling (Eager)": "#bcbd22",  # olive
    "flashinfer:top_k_top_p_sampling_from_logits": "#1f77b4",  # muted blue
    "flashinfer:sampling_from_logits": "#aec7e8",  # light blue
}

# Distinct markers so lines are distinguishable without color.
PROVIDER_MARKERS: dict[str, str] = {
    "FMMS (Triton)": "o",
    "FMMS (Helion)": "D",
    "FMMS (Triton NoNoise)": "^",
    "Multinomial Sampling (Compiled)": "s",
    "Multinomial Sampling (Eager)": "P",
    "flashinfer:top_k_top_p_sampling_from_logits": "X",
    "flashinfer:sampling_from_logits": "v",
}

# Hatch patterns for bar plots.
PROVIDER_HATCHES: dict[str, str] = {
    "FMMS (Triton)": "",
    "FMMS (Helion)": "//",
    "FMMS (Triton NoNoise)": "\\\\",
    "Multinomial Sampling (Compiled)": "///",
    "Multinomial Sampling (Eager)": "\\\\",
    "flashinfer:top_k_top_p_sampling_from_logits": "xxx",
    "flashinfer:sampling_from_logits": "...",
}

FLASHSAMPLING_RENAMES = {
    "FMMS (Triton)": "FlashSampling",
    "FMMS (Helion)": "FlashSampling (Helion)",
    "FMMS (Triton NoNoise)": "FlashSampling (NoNoise)",
}

# Make the FlashSampling names point to the same colors, markers, and hatches as the FMMS names.
for _mapping in [PROVIDER_COLORS, PROVIDER_MARKERS, PROVIDER_HATCHES]:
    for _old_key, _new_key in FLASHSAMPLING_RENAMES.items():
        _mapping[_new_key] = _mapping[_old_key]

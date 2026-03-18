"""Canonical display names for algorithms/providers.

Import these instead of hardcoding strings so that renames propagate everywhere.
"""


class ShortNames:
    fused_triton = "fused-triton"
    fused_triton_ret_logits = "fused-triton-ret-logits"
    fused_triton_greedy = "fused-triton-greedy"
    fused_cuda = "fused-cuda"
    fused_topk = "fused-topk"
    helion = "helion"
    naive_pt = "naive-pt"
    naive_compiled = "naive-compiled"
    sequential_compiled = "sequential-compiled"
    naive_tl_matmul = "naive-tl-matmul"
    jl_compiled = "jl-compiled"
    pt_qitra = "pt-qitra"
    flashinfer_top_k_top_p_sampling_from_logits = "flashinfer:top_k_top_p_sampling_from_logits"
    flashinfer_sampling_from_logits = "flashinfer:sampling_from_logits"
    greedy_baseline = "greedy"


class LongNames:
    fmms_triton = "FMMS (Triton)"
    fmms_triton_ret_logits = "FMMS (Triton, Return Logits)"
    fmms_greedy_triton = "FMMS Greedy (Triton)"
    fmms_cuda = "FMMS (CUDA)"
    fmms_topk = "FMMS Top-k (Triton)"
    fmms_helion = "FMMS (Helion)"
    multinomial_sampling_compiled = "Multinomial Sampling (Compiled)"
    multinomial_sampling_eager = "Multinomial Sampling (Eager)"
    sequential_compiled = "Sequential PyTorch Compiled"
    naive_tl_matmul = "Naive Triton Matmul"
    jl_compiled = "JL Compiled"
    pt_qitra = "Qitra"
    flashinfer_top_k_top_p_sampling_from_logits = "flashinfer:top_k_top_p_sampling_from_logits"
    flashinfer_sampling_from_logits = "flashinfer:sampling_from_logits"
    greedy_baseline = "Greedy Baseline"


class FlashSamplingNames:
    fmms_triton = "FlashSampling"
    fmms_helion = "FlashSampling (Helion)"
    fmms_greedy_triton = "FlashSampling Greedy"
    vllm_fmms = "vLLM + FlashSampling"


S = ShortNames
L = LongNames
F = FlashSamplingNames

# Renames FMMS → FlashSampling for kernel benchmarks.
FLASHSAMPLING_RENAMES: dict[str, str] = {
    L.fmms_triton: F.fmms_triton,
    L.fmms_helion: F.fmms_helion,
    L.fmms_greedy_triton: F.fmms_greedy_triton,
}

# Maps provider keys (used in get_sampler()) to display names (used in plots/CSVs).
short2long: dict[str, str] = {
    S.fused_triton: L.fmms_triton,
    S.fused_triton_ret_logits: L.fmms_triton_ret_logits,
    S.fused_triton_greedy: L.fmms_greedy_triton,
    S.fused_cuda: L.fmms_cuda,
    S.fused_topk: L.fmms_topk,
    S.helion: L.fmms_helion,
    S.naive_pt: L.multinomial_sampling_eager,
    S.naive_compiled: L.multinomial_sampling_compiled,
    S.sequential_compiled: L.sequential_compiled,
    S.naive_tl_matmul: L.naive_tl_matmul,
    S.jl_compiled: L.jl_compiled,
    S.pt_qitra: L.pt_qitra,
    S.flashinfer_top_k_top_p_sampling_from_logits: L.flashinfer_top_k_top_p_sampling_from_logits,
    S.flashinfer_sampling_from_logits: L.flashinfer_sampling_from_logits,
    S.greedy_baseline: L.greedy_baseline,
}

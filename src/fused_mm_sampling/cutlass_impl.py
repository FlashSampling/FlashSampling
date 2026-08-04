"""CUTLASS BF16 TP1 greedy FMMS provider."""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from .tp_info import TP1, TPInfo

_CSRC_DIR = Path(__file__).resolve().parent / "csrc" / "cutlass"
_module = None


def fused_mm_sample_cutlass_greedy(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    num_samples: int,
    temperature: torch.Tensor,
    tp: TPInfo = TP1,
    **_kwargs,
) -> torch.Tensor:
    """Return exact greedy token indices through the Gate 1h CUTLASS path."""
    if tp.size != 1:
        raise NotImplementedError("The CUTLASS greedy provider supports TP1 only")
    if num_samples != 1:
        raise ValueError("The CUTLASS greedy provider returns exactly one sample")
    del temperature
    return _get_module().greedy(weights, hidden_states)


def cutlass_plain_gemm(
    weights: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Return a BF16 `[V, H]` output from the matching plain CUTLASS GEMM."""
    return _get_module().plain_gemm(weights, hidden_states)


def cutlass_winning_plain_gemm(
    weights: torch.Tensor, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Return BF16 logits from the plain schedule matching fused dispatch."""
    if torch.cuda.get_device_capability(weights.device) != (10, 0):
        return cutlass_plain_gemm(weights, hidden_states)
    padded_hidden_states, output, _ = cutlass_make_plain_gemm_buffers(
        weights, hidden_states
    )
    variant = _winning_plain_gemm_variant(
        weights.size(1), hidden_states.size(0)
    )
    cutlass_launch_plain_gemm_variant(
        variant, weights, padded_hidden_states, output
    )
    return output.narrow(1, 0, hidden_states.size(0))


def _winning_plain_gemm_variant(hidden_size: int, n_hidden_states: int) -> str:
    if n_hidden_states <= 64:
        return "heur-128x64x128-c2x1x1"
    if n_hidden_states <= 256 and hidden_size <= 4096:
        return "heur-256x128x64-c4x1x1"
    if 128 < n_hidden_states <= 256:
        return "heur-256x128x64-c2x1x1"
    if n_hidden_states <= 256:
        return "heur-256x128x128-c2x1x1"
    raise ValueError("The Gate 2c winning dispatch supports H <= 256")


def cutlass_make_plain_gemm_buffers(
    weights: torch.Tensor, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Allocate the padded input and BF16 output for ordinary GEMM timing."""
    return _get_module().make_plain_gemm_buffers(weights, hidden_states)


def cutlass_launch_plain_gemm(
    weights: torch.Tensor,
    padded_hidden_states: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Launch ordinary CUTLASS GEMM into a preallocated BF16 output."""
    _get_module().launch_plain_gemm(weights, padded_hidden_states, output)


def cutlass_launch_plain_gemm_variant(
    variant: str,
    weights: torch.Tensor,
    padded_hidden_states: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Launch one named ordinary-GEMM tuning variant."""
    _get_module().launch_plain_gemm_variant(
        variant, weights, padded_hidden_states, output
    )


def cutlass_launch_small_n_gemv(
    weights: torch.Tensor,
    hidden_states: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Launch the preallocated BF16 H=1 or H=2 specialization."""
    _get_module().launch_small_n_gemv(weights, hidden_states, output)


def cutlass_greedy_kernel_attributes() -> dict[str, int]:
    """Return static resource metadata for the fused GEMM and Stage 2 kernels."""
    return _get_module().kernel_attributes()


def cutlass_make_greedy_buffers(
    weights: torch.Tensor, hidden_states: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Allocate the padded input, candidates, and output used for profiling."""
    return _get_module().make_greedy_buffers(weights, hidden_states)


def cutlass_launch_greedy_gemm(
    weights: torch.Tensor,
    padded_hidden_states: torch.Tensor,
    candidates: torch.Tensor,
    gemm_n: int,
    rounded_n: int,
) -> None:
    """Launch only the greedy CUTLASS GEMM into preallocated candidates."""
    _get_module().launch_greedy_gemm(
        weights, padded_hidden_states, candidates, gemm_n, rounded_n
    )


def cutlass_launch_greedy_stage2(
    candidates: torch.Tensor,
    output: torch.Tensor,
    m_tiles: int,
    rounded_n: int,
    n_hidden_states: int,
) -> None:
    """Launch only Stage 2 using preallocated candidates and output."""
    _get_module().launch_greedy_stage2(
        candidates, output, m_tiles, rounded_n, n_hidden_states
    )


def _get_module():
    global _module
    if _module is None:
        cutlass_root = Path(os.environ.get("CUTLASS_ROOT", "/opt/cutlass"))
        include_dirs = [
            str(cutlass_root / "include"),
            str(cutlass_root / "tools" / "util" / "include"),
            str(_CSRC_DIR),
        ]
        major, minor = torch.cuda.get_device_capability()
        architecture = f"{major}{minor}"
        if architecture not in {"90", "100"}:
            raise RuntimeError("The CUTLASS greedy provider supports SM90 and SM100 only")
        _module = load(
            name=f"fmms_cutlass_greedy_sm{architecture}",
            sources=[
                str(_CSRC_DIR / "greedy_provider.cu"),
                str(_CSRC_DIR / "winning_schedule_provider.cu"),
            ],
            extra_include_paths=include_dirs,
            extra_cuda_cflags=[
                "-O3",
                "-lineinfo",
                "--expt-relaxed-constexpr",
                f"-arch=sm_{architecture}a",
                f"-DFMMS_ARCH_SM{architecture}",
            ],
            verbose=os.environ.get("FMMS_CUTLASS_VERBOSE", "") == "1",
        )
    return _module

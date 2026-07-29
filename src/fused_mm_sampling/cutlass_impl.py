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
    """Return an FP32 `[V, H]` output from the matching plain CUTLASS GEMM."""
    return _get_module().plain_gemm(weights, hidden_states)


def cutlass_greedy_kernel_attributes() -> dict[str, int]:
    """Return static resource metadata for the fused GEMM and Stage 2 kernels."""
    return _get_module().kernel_attributes()


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
            sources=[str(_CSRC_DIR / "greedy_provider.cu")],
            extra_include_paths=include_dirs,
            extra_cuda_cflags=[
                "-O3",
                "--expt-relaxed-constexpr",
                f"-arch=sm_{architecture}a",
                f"-DFMMS_ARCH_SM{architecture}",
            ],
            verbose=os.environ.get("FMMS_CUTLASS_VERBOSE", "") == "1",
        )
    return _module

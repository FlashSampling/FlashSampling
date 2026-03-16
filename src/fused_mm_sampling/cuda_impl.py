"""CUDA C++ implementation of the FMMS kernel.

JIT-compiles the .cu file via torch.utils.cpp_extension.load() on first import.
Stage 2 reduction is done in Python (identical to the Triton wrapper).
"""

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_CSRC_DIR = Path(__file__).resolve().parent / "csrc"

_module = None


def _get_module():
    global _module
    if _module is not None:
        return _module
    _module = load(
        name="fmms_cuda",
        sources=[str(_CSRC_DIR / "fmms_kernel.cu")],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            f"-gencode=arch=compute_{_sm_version()},code=sm_{_sm_version()}",
        ],
        verbose=os.environ.get("FMMS_CUDA_VERBOSE", "") == "1",
    )
    return _module


def _sm_version() -> str:
    """Return the SM version string (e.g. '86' for RTX 3090)."""
    major, minor = torch.cuda.get_device_capability()
    return f"{major}{minor}"


TILE_V = 128


def fused_mm_sample_cuda(
    weights: torch.Tensor,  # [V, D] bfloat16
    hidden_states: torch.Tensor,  # [H, D] bfloat16
    num_samples: int,
    temperature: torch.Tensor,  # scalar (0-d)
    seed: int = 0,
) -> torch.Tensor:
    """Fused matrix-multiply & sampling using the CUDA C++ kernel."""
    V, D = weights.shape  # noqa: N806
    H = hidden_states.shape[0]  # noqa: N806
    assert hidden_states.shape[1] == D

    n_tiles_v = (V + TILE_V - 1) // TILE_V

    # Temperature must be float32 on GPU
    if temperature.dtype != torch.float32:
        temperature = temperature.float()

    maxs = torch.empty((n_tiles_v, H, num_samples), dtype=torch.float32, device=weights.device)
    maxs_idx = torch.empty((n_tiles_v, H, num_samples), dtype=torch.long, device=weights.device)

    mod = _get_module()
    mod.fmms_stage1(weights, hidden_states, maxs, maxs_idx, temperature, seed)

    # Stage 2: reduce across V-tiles (identical to Triton wrapper)
    idxs = maxs.max(dim=0).indices  # [H, num_samples]
    samples = maxs_idx.gather(dim=0, index=idxs.unsqueeze(0)).squeeze(0)
    return samples  # [H, num_samples]

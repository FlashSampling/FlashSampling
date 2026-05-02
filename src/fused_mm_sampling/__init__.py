"""Fused Matrix Multiplication Sampling

This package provides an efficient GPU implementation of fused matrix multiplication
and sampling operations using PyTorch and Triton.
"""

# Backstop triton's backend discovery, which on the Modal pytorch:2.10/cuda13
# image non-deterministically returns an empty dict via both the entry_points
# and TRITON_BACKENDS_IN_TREE dir-scan paths (likely an overlayfs/uv install
# interaction), surfacing as ``RuntimeError: 0 active drivers ([])`` on the
# first kernel launch. Importing the nvidia modules directly goes through
# sys.path and is reliable; setdefault keeps this idempotent when discovery
# already populated nvidia.
import triton.backends as _tb
from triton.backends import Backend
from triton.backends.nvidia.compiler import CUDABackend
from triton.backends.nvidia.driver import CudaDriver

_tb.backends.setdefault("nvidia", Backend(CUDABackend, CudaDriver))

from .core import fused_mm_sample_triton  # noqa: E402
from .tp_info import TPInfo  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "TPInfo",
    "fused_mm_sample_triton",
]

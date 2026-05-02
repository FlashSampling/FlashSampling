"""Fused Matrix Multiplication Sampling

This package provides an efficient GPU implementation of fused matrix multiplication
and sampling operations using PyTorch and Triton.
"""

from .core import fused_mm_sample_triton
from .tp_info import TPInfo

__version__ = "0.1.0"

__all__ = [
    "TPInfo",
    "fused_mm_sample_triton",
]

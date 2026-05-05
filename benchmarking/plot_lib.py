"""Shared helpers for plot-triton-bench.py and plot_tp_scaling.py."""

from pathlib import Path

import numpy as np
import pandas as pd


def read_triton_bench_csv(path: Path) -> pd.DataFrame:
    """Read a Triton benchmark CSV, stripping the ' (Time (ms))' column suffix."""
    df = pd.read_csv(path)
    df.columns = [c.removesuffix(" (Time (ms))") for c in df.columns]
    return df


def minmax_skip_zero_range(x: pd.Series) -> tuple[float, float]:
    """Min-max errorbar that returns (nan, nan) when the range is zero so seaborn skips drawing."""
    lo, hi = x.min(), x.max()
    if lo == hi:
        return (np.nan, np.nan)
    return (lo, hi)

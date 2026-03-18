"""Stacked-bar kernel breakdown from NCU sweep data.

One bar per batch size, stacked by individual kernel. Shows a single method
at a time (default: all methods).

Usage:
    python benchmarking/plot_ncu_kernel_breakdown.py \
        --dir benchmarking/profiles/sweeps/bsz/ncu-txt/tp1/case-small

    python benchmarking/plot_ncu_kernel_breakdown.py \
        --dir benchmarking/profiles/sweeps/bsz/ncu-txt/tp1/case-small \
        --method naive-pt
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from parse_ncu_sweep import METHODS, parse_ncu_csv

METHOD_FILENAMES = {label: fname for fname, label in METHODS}


def main():
    args = parse_args()
    base = Path(args.dir)
    bsz_dirs = sorted(base.glob("bsz*"), key=lambda p: int(p.name[3:]))
    assert bsz_dirs, f"No bszN/ directories found in {base}"

    out_dir = base / "kernel-breakdowns"
    out_dir.mkdir(exist_ok=True)

    methods = list(METHOD_FILENAMES.keys()) if args.method == "all" else [args.method]
    for method in methods:
        df = load_method(base, bsz_dirs, method)
        if df.empty:
            print(f"No data for {method}, skipping")
            continue
        out = out_dir / f"{method}.{args.fmt}"
        plot_breakdown(df, out, method)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Directory with bszN/ subdirs")
    parser.add_argument(
        "--method",
        default="all",
        choices=list(METHOD_FILENAMES.keys()) + ["all"],
        help="Which method to plot (default: all)",
    )
    parser.add_argument("--fmt", default="png", help="Output image format (default: png)")
    return parser.parse_args()


def load_method(base: Path, bsz_dirs: list[Path], method: str) -> pd.DataFrame:
    """Load NCU data for a single method across all batch sizes.

    Kernel names are shortened and durations aggregated by short name, so
    multiple raw kernels that map to the same short name are summed.
    """
    fname = METHOD_FILENAMES[method]
    frames = []
    for d in bsz_dirs:
        bsz = int(d.name[3:])
        path = d / fname
        if not path.exists():
            continue
        method_df = parse_ncu_csv(path).assign(bsz=bsz)
        method_df["kernel"] = method_df["kernel_name"].map(_shorten_kernel)
        frames.append(method_df)
    df = pd.concat(frames, ignore_index=True)
    return df.groupby(["bsz", "kernel"], as_index=False, sort=False).agg(
        duration_us=("duration_us", "sum")
    )


def plot_breakdown(df: pd.DataFrame, out: Path, method: str) -> None:
    """Stacked bar chart: one bar per batch size, one segment per kernel."""
    batch_sizes = sorted(df["bsz"].unique())
    # Preserve first-seen order for consistent stacking
    kernels = df.drop_duplicates("kernel", keep="first")["kernel"].tolist()
    palette = sns.color_palette("tab10", n_colors=len(kernels))

    pivot = (
        df.assign(duration_ms=df["duration_us"] / 1000)
        .pivot(index="bsz", columns="kernel", values="duration_ms")
        .reindex(columns=kernels)
        .fillna(0)
    )

    sns.set_context("talk")
    fig, ax = plt.subplots()
    x = range(len(batch_sizes))
    bar_width = 0.6

    bottom = pd.Series(0.0, index=pivot.index)
    for i, kernel in enumerate(kernels):
        values = pivot[kernel]
        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            width=bar_width,
            label=kernel,
            color=palette[i],
        )
        for rect, v in zip(bars.patches, values):
            _label_bar(ax, rect, f"{v:.2f}")
        bottom = bottom + values

    ax.set_xticks(list(x))
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Time (ms)")
    ax.grid(alpha=0.5, axis="y")

    ncol = 2 if len(kernels) > 4 else 1
    ax.legend(
        title="Kernel",
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        fontsize=9,
        framealpha=0.9,
    )

    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def _shorten_kernel(name: str) -> str:
    """Shorten verbose CUDA kernel names for legend readability."""
    if "fused_mm_sample" in name:
        return "fmms_kernel"
    # FMMS post-kernel reductions (must come before generic triton_red pattern)
    if "triton_red_fused_add_gather_max" in name or "triton_per_fused_add_gather_max" in name:
        return "reduction"
    if "triton_poi_fused_argmax_gather_stack" in name:
        return "TP winner select"
    # torch.compile fused kernels: split by what they actually compute.
    # Names encode the fused ops, e.g. triton_red_fused__softmax_div_ge_...
    if "triton_poi_fused" in name or "triton_red_fused" in name or "triton_per_fused" in name:
        if "_arange_" in name:
            return "arange"
        if "_softmax_" in name:
            return "softmax + masking"
        if "_amax_" in name or "_where_" in name:
            return "top-k masking"
        if "_div_" in name:
            return "temp scaling"
        return "torch.compile fused"
    if "gemv2T" in name or "gemm" in name:
        return "cuBLAS matmul"
    if "flashinfer::SamplingFromLogits" in name:
        return "FI SamplingFromLogits"
    if "flashinfer::TopPSamplingFromProb" in name:
        return "FI TopPSamplingFromProb"
    if "flashinfer::RadixTopKMask" in name:
        return "FI RadixTopKMask"
    if "flashinfer::" in name:
        start = name.index("flashinfer::") + len("flashinfer::")
        end = name.index("<", start) if "<" in name[start:] else len(name)
        return f"FI {name[start:end]}"
    if "cunn_SoftMax" in name:
        return "softmax"
    if "distribution_elementwise" in name:
        return "rand / multinomial"
    if "ArgMaxOps" in name:
        return "argmax"
    if "reduce_kernel" in name:
        return "reduce"
    if "vectorized_elementwise" in name or "elementwise_kernel" in name:
        return "elementwise"
    if "_assert_async" in name:
        return "assert_async"
    if "direct_copy" in name or "unrolled_elementwise" in name:
        return "copy / cast"
    if "CatArrayBatchedCopy" in name:
        return "all-gather concat"
    return name[:40]


def _label_bar(ax, rect, text: str, min_height: float = 0.3) -> None:
    """Place a centered label inside a bar segment if it's tall enough."""
    h = rect.get_height()
    if h < min_height:
        return
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        rect.get_y() + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )


if __name__ == "__main__":
    main()

"""Plot FMMS vs baselines latency scaling across TP=1, 2, 4, 8.

Two-panel figure (one panel per H value) on log-log axes so the slope
visualizes scaling and absolute lead is read off the y-axis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plot_styles import FLASHSAMPLING_RENAMES, PROVIDER_COLORS, PROVIDER_MARKERS, LongNames
from pydantic_settings import BaseSettings

L = LongNames

DEFAULT_PROVIDERS: list[str] = [
    FLASHSAMPLING_RENAMES[L.fmms_triton],
    L.multinomial_sampling_compiled,
    L.flashinfer_sampling_from_logits,
    L.flashinfer_top_k_top_p_sampling_from_logits,
]


def read_triton_bench_csv(path: Path) -> pd.DataFrame:
    """Read a Triton benchmark CSV, stripping the ' (Time (ms))' column suffix."""
    df = pd.read_csv(path)
    df.columns = [c.removesuffix(" (Time (ms))") for c in df.columns]
    return df


def collect_long_df(
    base_dir: Path,
    gpu: str,
    bench_fn: str,
    case: str,
    tps: list[int],
) -> pd.DataFrame:
    """Read the per-TP batch-scaling CSVs and return a long DataFrame.

    Columns: tp, n_hidden_states, provider, time[ms].
    """
    frames: list[pd.DataFrame] = []
    for tp in tps:
        csv = (
            base_dir
            / "triton-bench"
            / bench_fn
            / gpu
            / f"tp{tp}"
            / f"fused-mm-sample-batch-scaling-{case}.csv"
        )
        if not csv.exists():
            raise FileNotFoundError(csv)
        wide = read_triton_bench_csv(csv).rename(columns=FLASHSAMPLING_RENAMES)
        long = wide.melt(id_vars=["n_hidden_states"], var_name="provider", value_name="time[ms]")
        long["tp"] = tp
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def plot_tp_scaling(
    long: pd.DataFrame,
    h_values: list[int],
    providers: list[str],
) -> plt.Figure:
    sns.set_context("talk")
    plot_df = long.query("n_hidden_states in @h_values and provider in @providers").copy()
    plot_df["n_hidden_states"] = plot_df["n_hidden_states"].astype(int)
    plot_df["time[us]"] = plot_df["time[ms]"] * 1000

    palette = {p: PROVIDER_COLORS[p] for p in providers}
    markers = {p: PROVIDER_MARKERS[p] for p in providers}

    fig, axes = plt.subplots(1, len(h_values), figsize=(5 * len(h_values), 4), sharey=False)
    if len(h_values) == 1:
        axes = [axes]

    unique_tps = sorted(plot_df["tp"].unique())

    for ax_idx, (ax, h) in enumerate(zip(axes, h_values)):
        sub = plot_df.query("n_hidden_states == @h")
        sns.lineplot(
            sub,
            x="tp",
            y="time[us]",
            hue="provider",
            hue_order=providers,
            style="provider",
            style_order=providers,
            markers=markers,
            dashes=False,
            ax=ax,
            palette=palette,
        )

        # Ideal 1/TP reference, anchored at FlashSampling TP=1.
        fs_name = FLASHSAMPLING_RENAMES[L.fmms_triton]
        fs_tp1 = sub.query("provider == @fs_name and tp == 1")["time[us]"].iloc[0]
        ref_values = [fs_tp1 / tp for tp in unique_tps]
        ax.plot(
            unique_tps,
            ref_values,
            linestyle=":",
            color=PROVIDER_COLORS[fs_name],
            linewidth=2.5,
            marker="*",
            label="Ideal 1/TP",
            zorder=1,
        )

        ax.set_xscale("log")
        ax.set_xticks(unique_tps, labels=[str(t) for t in unique_tps])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(plt.MultipleLocator(100))  # every 100us
        ax.grid(alpha=0.5)
        ax.set_xlabel("Tensor Parallel Size")
        ax.set_ylabel("Time (μs)" if ax_idx == 0 else "")
        ax.set_title(f"Batch Size = {h}")
        ax.annotate(
            "lower is better",
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            ha="left",
            va="bottom",
            fontsize=14,
            color="gray",
            style="italic",
        )
        ax.legend_.remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        title="Method",
        bbox_to_anchor=(0.5, 1.30),
        ncol=3,
        fontsize=18,
        title_fontsize=20,
    )
    fig.tight_layout()
    return fig


class Args(BaseSettings, cli_parse_args=True):
    base_dir: Path = Path(__file__).parent / "modal-results"
    gpu: str = "b200"
    bench_fn: str = "own"
    case: str = "large"
    tps: list[int] = [1, 2, 4, 8]
    h_values: list[int] = [16, 64, 256]
    out: Path = Path(__file__).parent.parent / "imgs" / "tp-scaling" / "tp-scaling.pdf"


if __name__ == "__main__":
    args = Args()
    long = collect_long_df(args.base_dir, args.gpu, args.bench_fn, args.case, args.tps)
    fig = plot_tp_scaling(long, args.h_values, DEFAULT_PROVIDERS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")

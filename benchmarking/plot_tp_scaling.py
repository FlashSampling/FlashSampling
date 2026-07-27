"""Plot FMMS vs baselines latency scaling across TP=1, 2, 4, 8.

Two-panel figure (one panel per H value) on log-log axes so the slope
visualizes scaling and absolute lead is read off the y-axis.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from plot_lib import read_triton_bench_csv
from plot_styles import FLASHSAMPLING_RENAMES, PROVIDER_COLORS, PROVIDER_MARKERS, LongNames
from pydantic_settings import BaseSettings

L = LongNames

DEFAULT_PROVIDERS: list[str] = [
    FLASHSAMPLING_RENAMES.get(name, name)
    for name in [
        L.fmms_triton,
        L.fmms_triton_p2p_no_overlap,
        L.multinomial_sampling_compiled,
        L.flashinfer_sampling_from_logits,
        L.flashinfer_top_k_top_p_sampling_from_logits,
    ]
]


def collect_long_df(args: "Args") -> pd.DataFrame:
    """Read the per-TP batch-scaling CSVs and return a long DataFrame.

    When ``args.use_reruns`` is set, also load sibling ``{gpu}-rerun*``
    folders so each (tp, n_hidden_states, provider) cell has multiple rows.
    Slow-pod runs are dropped per ``args.max_slowdown`` (FMMS-Triton median
    must be within that factor of the fastest run at the same tp).

    Columns: tp, n_hidden_states, provider, time[ms], run.
    """
    fmms = FLASHSAMPLING_RENAMES[L.fmms_triton]
    parent = args.base_dir / "triton-bench" / args.bench_fn
    frames: list[pd.DataFrame] = []
    for tp in args.tps:
        run_dirs = [parent / args.gpu]
        if args.use_reruns:
            run_dirs += sorted(parent.glob(f"{args.gpu}-rerun*"))

        # First pass: read all and compute FMMS median per run
        per_run: list[tuple[str, float, pd.DataFrame]] = []
        for run_dir in run_dirs:
            csv = run_dir / f"tp{tp}" / f"fused-mm-sample-batch-scaling-{args.case}.csv"
            if not csv.exists():
                if run_dir.name == args.gpu:
                    raise FileNotFoundError(csv)
                print(f"warn: missing {csv}")
                continue
            wide = read_triton_bench_csv(csv).rename(columns=FLASHSAMPLING_RENAMES)
            fmms_med = wide[fmms].median()
            per_run.append((run_dir.name, fmms_med, wide))

        if not per_run:
            continue
        fastest = min(m for _, m, _ in per_run)
        for name, fmms_med, wide in per_run:
            slowdown = fmms_med / fastest
            if slowdown > args.max_slowdown:
                print(f"drop tp{tp} {name}: FMMS={fmms_med * 1000:.1f}us, {slowdown:.2f}x slower")
                continue
            long = wide.melt(
                id_vars=["n_hidden_states"], var_name="provider", value_name="time[ms]"
            )
            long["tp"] = tp
            long["run"] = name
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

    lineplot_kwargs = {"estimator": "min", "errorbar": None}

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
            markersize=12,
            dashes=False,
            ax=ax,
            palette=palette,
            **lineplot_kwargs,
        )

        # Ideal 1/TP reference, anchored at FlashSampling TP=1 (min across runs).
        fs_name = FLASHSAMPLING_RENAMES[L.fmms_triton]
        fs_tp1 = sub.query("provider == @fs_name and tp == 1")["time[us]"].min()
        ref_values = [fs_tp1 / tp for tp in unique_tps]
        ax.plot(
            unique_tps,
            ref_values,
            linestyle=":",
            color=PROVIDER_COLORS[fs_name],
            linewidth=2.5,
            marker="*",
            markersize=14,
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
    use_reruns: bool = False
    max_slowdown: float = 2.0  # drop runs whose FMMS median is more than this factor slower than the fastest at the same tp


def write_summary_csv(long: pd.DataFrame, args: "Args") -> Path:
    """Aggregate (provider, n_hidden_states, tp) to mean/min/max in microseconds."""
    sel = long.query("n_hidden_states in @args.h_values and provider in @DEFAULT_PROVIDERS").copy()
    sel["time[us]"] = sel["time[ms]"] * 1000
    summary = (
        sel.groupby(["provider", "n_hidden_states", "tp"])["time[us]"]
        .agg(n_runs="count", mean_time_us="mean", min_time_us="min", max_time_us="max")
        .round(2)
        .reset_index()
        .rename(
            columns={
                "provider": "method",
                "n_hidden_states": "batch_size",
                "tp": "tensor_parallel_size",
            }
        )
    )
    summary["batch_size"] = summary["batch_size"].astype(int)
    summary = summary.sort_values(["method", "batch_size", "tensor_parallel_size"])
    summary["range_us"] = (summary["max_time_us"] - summary["min_time_us"]).round(2)
    csv_path = args.out.with_name("tp-scaling-runs-summary.csv")
    summary.to_csv(csv_path, index=False)
    return csv_path


if __name__ == "__main__":
    args = Args()
    long = collect_long_df(args)
    fig = plot_tp_scaling(long, args.h_values, DEFAULT_PROVIDERS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"wrote {args.out}")
    csv_path = write_summary_csv(long, args)
    print(f"wrote {csv_path}")

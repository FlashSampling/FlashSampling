"""Plot median TPOT vs concurrency for vLLM benchmark results.

Usage:
    python benchmarking/vllm/plot_tpot.py [--results-dir <path>]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASELINE_NAME = "vLLM Baseline"
FMMS_NAME = "vLLM + FMMS"
FMMS_DIR = "fmms-triton"

VARIANT_COLORS = {BASELINE_NAME: "#7f7f7f", FMMS_NAME: "#d62728"}
VARIANT_MARKERS = {BASELINE_NAME: "s", FMMS_NAME: "o"}

# Make the FlashSampling name point to the same color and marker as the FMMS name.
FLASHSAMPLING_RENAMES = {"vLLM + FMMS": "vLLM + FlashSampling"}
mappings = [VARIANT_COLORS, VARIANT_MARKERS]
for mapping in mappings:
    for old_key, new_key in FLASHSAMPLING_RENAMES.items():
        mapping[new_key] = mapping[old_key]

MODELS = [
    "Qwen3-1.7B",
    "Qwen3-8B",
    "Qwen3-32B",
    "gpt-oss-120b",
]

MAX_CONCURRENCY = 64


def resolve_model_dirs(results_dir: Path, model: str) -> list[Path]:
    """Find all model directories matching the model name, including -trialN suffixes."""
    pattern = re.compile(rf"^{re.escape(model)}(-trial\d+)?$")
    return sorted(d for d in results_dir.iterdir() if d.is_dir() and pattern.match(d.name))


def latest_timestamp(variant_dir: Path) -> Path:
    dirs = sorted(d for d in variant_dir.iterdir() if d.is_dir())
    if not dirs:
        raise FileNotFoundError(f"No runs found in {variant_dir}")
    return dirs[-1]


def load_variant(model_dir: Path, variant_key: str) -> pd.DataFrame | None:
    variant_dir = model_dir / variant_key
    if not variant_dir.exists():
        print(f"Warning: {variant_dir} not found, skipping")
        return None
    run_dir = latest_timestamp(variant_dir)
    return pd.read_csv(run_dir / "summary.csv")


def load_all_data(results_dir: Path, fmms_name: str) -> pd.DataFrame:
    frames = []
    for model in MODELS:
        model_dirs = resolve_model_dirs(results_dir, model)
        if not model_dirs:
            print(f"Warning: no directory matching {model} in {results_dir}, skipping")
            continue
        for model_dir in model_dirs:
            for variant_key, display_name in [("baseline", BASELINE_NAME), (FMMS_DIR, fmms_name)]:
                df = load_variant(model_dir, variant_key)
                if df is None:
                    continue
                df["variant"] = display_name
                df["model"] = model
                frames.append(
                    df[["model", "variant", "max_concurrency", "median_tpot_ms", "run_number"]]
                )
    return pd.concat(frames, ignore_index=True)


def paired_speedups(model_dirs: list[Path], max_concurrency: int) -> pd.DataFrame:
    """Compute paired speedup ratios between baseline and FMMS runs.

    Comparisons are made only within the same trial directory (e.g.
    Qwen3-8B-trial1/baseline vs Qwen3-8B-trial1/fmms-triton), never across
    trials. Within a trial, each run_number is paired 1:1 (run 0 baseline vs
    run 0 FMMS, etc.). The median across all paired speedups from all trials
    is reported.
    """
    frames = []
    for model_dir in model_dirs:
        baseline_df = load_variant(model_dir, "baseline")
        fmms_df = load_variant(model_dir, FMMS_DIR)
        if baseline_df is None or fmms_df is None:
            continue
        merged = (
            baseline_df[["max_concurrency", "run_number", "median_tpot_ms"]]
            .merge(
                fmms_df[["max_concurrency", "run_number", "median_tpot_ms"]],
                on=["max_concurrency", "run_number"],
                suffixes=("_base", "_fmms"),
            )
            .query("max_concurrency <= @max_concurrency")
        )
        merged["speedup_pct"] = (
            1 - merged["median_tpot_ms_fmms"] / merged["median_tpot_ms_base"]
        ) * 100
        frames.append(merged[["max_concurrency", "speedup_pct"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Shared scatter+median line plot
# ---------------------------------------------------------------------------

FIGSIZE = (6.4, 5)
DOT_SIZE = 50
DOT_ALPHA = 0.4


def _plot_scatter_line(
    ax: plt.Axes,
    series: list[tuple[str, pd.DataFrame]],
    x_col: str,
    y_col: str,
    ylabel: str,
    hline: float | None = None,
):
    """Plot median lines with scatter dots for one or more series.

    Each entry in *series* is (label, dataframe). Colors are looked up from
    VARIANT_COLORS. The dataframe must contain *x_col* and *y_col* columns.
    """
    all_x = set()
    for label, sdf in series:
        color = VARIANT_COLORS[label]
        marker = VARIANT_MARKERS.get(label, "o")
        medians = sdf.groupby(x_col)[y_col].median()
        ax.plot(medians.index, medians.values, marker=marker, color=color, label=label, zorder=3)
        ax.scatter(
            sdf[x_col],
            sdf[y_col],
            color=color,
            alpha=DOT_ALPHA,
            s=DOT_SIZE,
            zorder=2,
            edgecolors="none",
        )
        all_x.update(sdf[x_col].unique())
    concurrencies = sorted(all_x)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    ax.set_xticks(concurrencies, labels=[int(x) for x in concurrencies], minor=False)
    ax.set_xticks([], minor=True)
    if hline is not None:
        ax.axhline(hline, color="black", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.5)


# ---------------------------------------------------------------------------
# Per-model TPOT plots (imgs/tpots/)
# ---------------------------------------------------------------------------


def plot_tpots(
    df: pd.DataFrame, results_dir: Path, imgs_dir: Path, fmms_name: str, fmt: str = "png"
):
    tpots_dir = imgs_dir / "tpots"
    tpots_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        mdf = df.query("model == @model")
        if mdf.empty:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE)
        series = []
        for variant in [BASELINE_NAME, fmms_name]:
            vdf = mdf.query("variant == @variant")
            if not vdf.empty:
                series.append((variant, vdf.rename(columns={"median_tpot_ms": "y"})))
        _plot_scatter_line(
            ax, series=series, x_col="max_concurrency", y_col="y", ylabel="Median TPOT (ms)"
        )
        ax.set_title(model)
        ax.legend(title="Method")

        fig.tight_layout()
        out = tpots_dir / f"{model}.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {out}")


# ---------------------------------------------------------------------------
# Per-model speedup plots (imgs/speedups/)
# ---------------------------------------------------------------------------


def plot_speedups(
    results_dir: Path, imgs_dir: Path, max_concurrency: int, fmms_name: str, fmt: str = "png"
):
    speedups_dir = imgs_dir / "speedups"
    speedups_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        model_dirs = resolve_model_dirs(results_dir, model)
        if not model_dirs:
            continue
        sdf = paired_speedups(model_dirs, max_concurrency)
        if sdf.empty:
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE)
        _plot_scatter_line(
            ax,
            series=[(fmms_name, sdf)],
            x_col="max_concurrency",
            y_col="speedup_pct",
            ylabel="Speedup (%)",
            hline=0,
        )
        ax.set_title(model)

        fig.tight_layout()
        out = speedups_dir / f"{model}.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {out}")


# ---------------------------------------------------------------------------
# Per-model strip plots (imgs/strips/)
# ---------------------------------------------------------------------------


def plot_strips(df: pd.DataFrame, imgs_dir: Path, fmms_name: str, fmt: str = "png"):
    strips_dir = imgs_dir / "strips"
    strips_dir.mkdir(parents=True, exist_ok=True)

    variants = [BASELINE_NAME, fmms_name]

    for model in MODELS:
        mdf = df.query("model == @model and max_concurrency <= @MAX_CONCURRENCY")
        if mdf.empty:
            continue

        present_variants = [v for v in variants if v in mdf["variant"].unique()]
        concurrencies = sorted(mdf["max_concurrency"].unique())
        n_conc = len(concurrencies)
        n_variants = len(present_variants)

        fig, ax = plt.subplots(figsize=(max(8, n_conc * 1.8), 5))

        width = 0.3
        jitter = 0.06
        legend_handles = {}

        for i, conc in enumerate(concurrencies):
            medians_at_conc = {}
            for j, variant in enumerate(present_variants):
                offset = (j - (n_variants - 1) / 2) * width
                pos = i + offset
                vals = mdf.query("max_concurrency == @conc and variant == @variant")[
                    "median_tpot_ms"
                ].values

                if len(vals) == 0:
                    continue

                color = VARIANT_COLORS[variant]
                med = np.median(vals)
                medians_at_conc[variant] = med

                rng = np.random.default_rng(seed=hash((model, conc, variant)) & 0xFFFFFFFF)
                x_jittered = pos + rng.uniform(-jitter, jitter, size=len(vals))
                sc = ax.scatter(
                    x_jittered,
                    vals,
                    color=color,
                    alpha=0.5,
                    s=30,
                    zorder=3,
                    edgecolors="none",
                )
                legend_handles[variant] = sc

                ax.plot(
                    [pos - width * 0.35, pos + width * 0.35],
                    [med, med],
                    color=color,
                    linewidth=2.5,
                    zorder=4,
                )

            if BASELINE_NAME in medians_at_conc and fmms_name in medians_at_conc:
                b_med = medians_at_conc[BASELINE_NAME]
                f_med = medians_at_conc[fmms_name]
                tpot_change = (f_med / b_med - 1) * 100
                ann_color = "#2ca02c" if tpot_change < 0 else "#b22222"
                higher = max(b_med, f_med)
                ax.annotate(
                    "",
                    xy=(i, f_med),
                    xytext=(i, b_med),
                    arrowprops=dict(arrowstyle="->", color=ann_color, lw=1.5),
                )
                ax.annotate(
                    f"{tpot_change:+.1f}%",
                    xy=(i, higher),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color=ann_color,
                )

        ax.set_xticks(range(n_conc))
        ax.set_xticklabels([str(int(c)) for c in concurrencies])
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Median TPOT (ms)")
        ax.set_title(f"E2E Speedups for {model}")
        ax.grid(axis="y", alpha=0.3)

        if legend_handles:
            ax.legend(
                legend_handles.values(),
                legend_handles.keys(),
                loc="upper left",
                framealpha=0.9,
            )

        fig.tight_layout()
        out = strips_dir / f"{model}.{fmt}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot TPOT and speedup from vLLM benchmarks")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory containing model subdirectories (default: benchmarking/vllm/)",
    )
    parser.add_argument(
        "--fmt",
        default="png",
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--use-name-flashsampling",
        type=int,
        default=0,
        help="Use 'FlashSampling' instead of 'FMMS' in plot labels",
    )
    args = parser.parse_args()

    fmms_name = FLASHSAMPLING_RENAMES[FMMS_NAME] if args.use_name_flashsampling else FMMS_NAME

    results_dir = args.results_dir
    imgs_dir = results_dir / "imgs"
    imgs_dir.mkdir(parents=True, exist_ok=True)

    sns.set_context("talk")
    df = load_all_data(results_dir, fmms_name)
    df = df.query("max_concurrency <= @MAX_CONCURRENCY")

    plot_tpots(df, results_dir, imgs_dir, fmms_name, fmt=args.fmt)
    plot_speedups(results_dir, imgs_dir, MAX_CONCURRENCY, fmms_name, fmt=args.fmt)
    plot_strips(df, imgs_dir, fmms_name, fmt=args.fmt)


if __name__ == "__main__":
    main()

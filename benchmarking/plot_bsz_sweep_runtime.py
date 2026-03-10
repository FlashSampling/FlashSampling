"""Stacked bar chart: matmul vs sampling time for FMMS, Naive, and FI-sample.

Combines NCU sweep data (inter-kernel breakdown) with Proton sweep data
(intra-kernel breakdown for FlashSampling) to produce a stacked bar chart
and a CSV file.

Usage:
    python plot_bsz_sweep_runtime.py
    python plot_bsz_sweep_runtime.py --ncu-dir ... --proton-dir ... --out-dir ...
    python plot_bsz_sweep_runtime.py --fmt pdf
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from parse_ncu_sweep import parse_method
from parse_proton_intrakernel import parse_chrome_trace, trace_phase_pcts
from plot_styles import PROVIDER_COLORS, PROVIDER_HATCHES, PROVIDER_MARKERS

SWEEPS = Path("profiles/sweeps/bsz")
DEFAULT_NCU_DIR = SWEEPS / "ncu-txt" / "case-small"
DEFAULT_PROTON_DIR = SWEEPS / "proton" / "case-small"
DEFAULT_OUT_DIR = SWEEPS

# Methods: (NCU filename, internal key, is_fmms)
METHODS = [
    ("fused-triton.txt", "FMMS (Triton)", True),
    ("naive-compiled.txt", "Multinomial Sampling (Compiled)", False),
    ("flashinfer-sampling.txt", "flashinfer:sampling_from_logits", False),
]


def load_data(ncu_dir: Path, proton_dir: Path) -> list[dict]:
    """Load and combine NCU + Proton data into per-method, per-bsz rows."""
    bsz_dirs = sorted(ncu_dir.glob("bsz*"), key=lambda p: int(p.name[3:]))
    rows = []

    for d in bsz_dirs:
        bsz = int(d.name[3:])

        # Load Proton data for FMMS split
        proton_trace = parse_chrome_trace(proton_dir / f"bsz{bsz}" / "kernel.chrome_trace")
        proton_pcts = trace_phase_pcts(proton_trace) if proton_trace else None

        for fname, label, is_fmms in METHODS:
            m = parse_method(d / fname)
            if not m:
                continue

            if is_fmms and proton_pcts:
                # FMMS: use Proton percentages to split the NCU total
                matmul_frac = proton_pcts["matmul"] / 100
                sampling_frac = proton_pcts["sampling"] / 100
                matmul_us = m["total_us"] * matmul_frac
                sampling_us = m["total_us"] * sampling_frac
            else:
                # Baselines: NCU gives matmul (1st kernel) and post-matmul
                matmul_us = m["matmul_us"]
                sampling_us = m["post_us"]

            rows.append(
                {
                    "bsz": bsz,
                    "method": label,
                    "matmul_us": round(matmul_us, 1),
                    "sampling_us": round(sampling_us, 1),
                    "total_us": round(m["total_us"], 1),
                }
            )

    return rows


def save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bsz", "method", "matmul_us", "sampling_us", "total_us"])
        w.writeheader()
        w.writerows(rows)
    print(f"CSV saved to {path}")


def _apply_hatches(ax, methods):
    hatches = [PROVIDER_HATCHES.get(m, "") for m in methods]
    for container, hatch in zip(ax.containers, hatches):
        for bar in container:
            bar.set_hatch(hatch)
    return hatches


def plot(
    rows: list[dict],
    out_path: Path,
    fmt: str,
    y_col: str,
    y_label: str,
    y_cap: float | None = None,
    log_y: bool = False,
    style: str = "line",
) -> plt.Figure:
    methods = [label for _, label, _ in METHODS]
    df = pd.DataFrame(rows)

    palette = {m: PROVIDER_COLORS[m] for m in methods}
    fig, ax = plt.subplots()

    if style == "bar":
        sns.barplot(
            df,
            x="bsz",
            y=y_col,
            hue="method",
            hue_order=methods,
            palette=palette,
            ax=ax,
        )
        hatches = _apply_hatches(ax, methods)
        if y_cap is not None:
            ax.set_ylim(0, y_cap)
            for container in ax.containers:
                for bar in container:
                    if bar.get_height() > y_cap:
                        true_val = bar.get_height()
                        bar.set_height(y_cap)
                        bx = bar.get_x() + bar.get_width() / 2
                        ax.text(
                            bx,
                            y_cap * 1.01,
                            f"{true_val:.0f}\u00b5s",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            fontweight="bold",
                            clip_on=False,
                        )
        sns.move_legend(ax, "upper center", title="Method", bbox_to_anchor=(0.5, 1.35), ncol=1)
        for handle, hatch in zip(ax.get_legend().legend_handles, hatches):
            handle.set_hatch(hatch)
        ax.set_xticks(ax.get_xticks(), labels=df["bsz"].unique().astype(int))
    else:
        for method in methods:
            mdf = df.query("method == @method")
            ax.plot(
                mdf["bsz"],
                mdf[y_col],
                color=PROVIDER_COLORS[method],
                marker=PROVIDER_MARKERS.get(method, "o"),
                label=method,
                linewidth=2,
                markersize=6,
                zorder=3,
            )
        if y_cap is not None:
            ax.set_ylim(0, y_cap)
            # for method in methods:
            #     mdf = df[df["method"] == method]
            #     for _, row in mdf.iterrows():
            #         if row[y_col] > y_cap:
            #             ax.text(
            #                 row["bsz"], y_cap * 1.01, f"{row[y_col]:.0f}\u00b5s",
            #                 ha="center", va="bottom", fontsize=9, fontweight="bold",
            #                 color=PROVIDER_COLORS[method], clip_on=False,
            #             )
        ax.legend(title="Method", loc="upper center", bbox_to_anchor=(0.5, 1.35), ncol=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(df["bsz"].unique())
        ax.set_xticklabels(df["bsz"].unique().astype(int))
        ax.minorticks_off()

    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.5, axis="y")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel(y_label)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(f".{fmt}"), dpi=300, bbox_inches="tight")
    print(f"Plot saved to {out_path.with_suffix(f'.{fmt}')}")
    plt.close(fig)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu-dir", type=Path, default=DEFAULT_NCU_DIR)
    parser.add_argument("--proton-dir", type=Path, default=DEFAULT_PROTON_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fmt", default="png", help="Output image format (default: png)")
    args = parser.parse_args()

    rows = load_data(args.ncu_dir, args.proton_dir)
    save_csv(rows, args.out_dir / "runtime-breakdown.csv")
    plot(
        rows,
        args.out_dir / "sampling-latency",
        args.fmt,
        y_col="sampling_us",
        y_label="Sampling Latency (\u00b5s)",
        y_cap=800,
    )
    plot(
        rows,
        args.out_dir / "matmul-latency",
        args.fmt,
        y_col="matmul_us",
        y_label="Matmul Latency (\u00b5s)",
        style="bar",
    )


if __name__ == "__main__":
    main()

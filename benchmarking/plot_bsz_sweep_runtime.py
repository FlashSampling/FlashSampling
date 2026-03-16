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
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from parse_ncu_sweep import parse_ncu_csv
from parse_proton_intrakernel import parse_chrome_trace, trace_phase_pcts
from plot_styles import FLASHSAMPLING_RENAMES, PROVIDER_COLORS, PROVIDER_HATCHES, PROVIDER_MARKERS

SWEEPS = Path("profiles/sweeps/bsz")
N_PROCS = 1
CASE = "small"
PROTON_DIR = SWEEPS / "proton" / f"tp{N_PROCS}" / f"case-{CASE}"

# Methods: (NCU filename, internal key, is_fmms)
METHODS = [
    ("fused-triton.txt", "FMMS (Triton)", True),
    ("naive-compiled.txt", "Multinomial Sampling (Compiled)", False),
    ("flashinfer:sampling_from_logits.txt", "flashinfer:sampling_from_logits", False),
]


def load_data(ncu_dir: Path, proton_dir: Path) -> pd.DataFrame:
    """Load and combine NCU + Proton data into a DataFrame.

    Columns: bsz, method, matmul_us, sampling_us, total_us.
    """
    bsz_dirs = sorted(ncu_dir.glob("bsz*"), key=lambda p: int(p.name[3:]))
    frames = []

    for d in bsz_dirs:
        bsz = int(d.name[3:])

        # Load Proton data for FMMS split
        proton_trace = parse_chrome_trace(proton_dir / f"bsz{bsz}" / "kernel.chrome_trace")
        proton_pcts = trace_phase_pcts(proton_trace) if proton_trace else None

        for fname, label, is_fmms in METHODS:
            path = d / fname
            if not path.exists():
                continue
            kdf = parse_ncu_csv(path)
            is_matmul = kdf["kernel_name"].str.contains("gemm|fused_mm_sample", case=False)
            assert is_matmul.iloc[0], (
                f"First kernel in {path} is not a matmul: {kdf['kernel_name'].iloc[0]}"
            )
            row = {
                "bsz": bsz,
                "method": label,
                "total_us": kdf["duration_us"].sum(),
            }
            if is_fmms:
                if not proton_pcts:
                    print(f"WARNING: skipping FMMS at bsz={bsz}: no Proton trace found")
                    continue
                # Split the fused kernel using Proton percentages, then add
                # auxiliary kernels (local reduce, TP reduce) to sampling.
                fused_us = kdf.loc[is_matmul, "duration_us"].sum()
                aux_us = kdf.loc[~is_matmul, "duration_us"].sum()
                matmul_frac = proton_pcts["matmul"] / 100
                sampling_frac = proton_pcts["sampling"] / 100
                row["matmul_us"] = fused_us * matmul_frac
                row["sampling_us"] = fused_us * sampling_frac + aux_us
            else:
                row["matmul_us"] = kdf.loc[is_matmul, "duration_us"].sum()
                row["sampling_us"] = kdf.loc[~is_matmul, "duration_us"].sum()

            frames.append(row)

    df = pd.DataFrame(frames)
    df[["matmul_us", "sampling_us", "total_us"]] = df[
        ["matmul_us", "sampling_us", "total_us"]
    ].round(1)
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"CSV saved to {path}")


def _apply_hatches(ax, methods):
    hatches = [PROVIDER_HATCHES.get(m, "") for m in methods]
    for container, hatch in zip(ax.containers, hatches):
        for bar in container:
            bar.set_hatch(hatch)
    return hatches


def plot(
    df: pd.DataFrame,
    out_path: Path,
    fmt: str,
    y_col: str,
    y_label: str,
    y_cap: float | None = None,
    log_y: bool = False,
    style: str = "line",
) -> plt.Figure:
    methods = list(df["method"].unique())

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
    parser.add_argument(
        "--n_procs", type=int, default=N_PROCS, help="Number of processes (default: 1)"
    )
    parser.add_argument("--case", default=CASE, help="Benchmark case (default: small)")
    parser.add_argument("--ncu-dir", type=Path, default=None)
    parser.add_argument("--proton-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fmt", default="png", help="Output image format (default: png)")
    parser.add_argument(
        "--use-name-flashsampling",
        action="store_true",
        help="Use 'FlashSampling' instead of 'FMMS' in plot labels",
    )
    args = parser.parse_args()
    if args.ncu_dir is None:
        args.ncu_dir = SWEEPS / "ncu-txt" / f"tp{args.n_procs}" / f"case-{args.case}"
    if args.proton_dir is None:
        args.proton_dir = SWEEPS / "proton" / f"tp{args.n_procs}" / f"case-{args.case}"
    if args.out_dir is None:
        args.out_dir = SWEEPS / f"tp{args.n_procs}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_data(args.ncu_dir, args.proton_dir)
    if args.use_name_flashsampling:
        rows["method"] = rows["method"].replace(FLASHSAMPLING_RENAMES)
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

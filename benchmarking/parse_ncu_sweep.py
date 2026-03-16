"""Parse NCU CSV exports from a batch-size sweep and produce a summary.

Usage:
    python parse_ncu_sweep.py --dir profiles/sweeps/bsz/ncu-txt/tp1/case-small
    python parse_ncu_sweep.py --dir profiles/sweeps/bsz/ncu-txt/tp1/case-small > summary.txt

Expects directory layout (files are CSV despite .txt extension):
    <dir>/bsz1/fused-triton.txt
    <dir>/bsz1/naive-compiled.txt
    <dir>/bsz1/flashinfer:sampling_from_logits.txt  (optional)
    <dir>/bsz1/flashinfer:top_k_top_p_sampling_from_logits.txt (optional)
    <dir>/bsz4/fused-triton.txt
    ...
"""

import argparse
import io
from pathlib import Path

import pandas as pd

# Methods to look for in each bszN/ directory: (filename, display name)
METHODS = [
    ("fused-triton.txt", "fused-triton"),
    ("naive-pt.txt", "naive-pt"),
    ("naive-compiled.txt", "naive-compiled"),
    ("flashinfer:sampling_from_logits.txt", "fi-sample"),
    ("flashinfer:top_k_top_p_sampling_from_logits.txt", "fi-topkp"),
]

FUSED = "fused-triton"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", required=True, help="Directory with bszN/ subdirs")
    args = parser.parse_args()

    base = Path(args.dir)
    bsz_dirs = sorted(base.glob("bsz*"), key=lambda p: int(p.name[3:]))
    if not bsz_dirs:
        print(f"No bszN/ directories found in {base}")
        return

    df = _load_sweep(base, bsz_dirs)
    if df.empty:
        print("No data parsed. Are the .txt files in NCU CSV format (--csv --page raw)?")
        return

    print(f"NCU Batch-Size Sweep Summary: {base}")
    print()
    _print_summary(df)


def _load_sweep(base: Path, bsz_dirs: list[Path]) -> pd.DataFrame:
    """Load all NCU CSV files into a single DataFrame.

    Columns: bsz, method, kernel_name, duration_us.
    TP>1 durations are averaged across ranks.
    """
    frames = []
    for d in bsz_dirs:
        bsz = int(d.name[3:])
        for fname, label in METHODS:
            path = d / fname
            if not path.exists():
                continue
            method_df = parse_ncu_csv(path).assign(bsz=bsz, method=label)
            frames.append(method_df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_ncu_csv(path: Path) -> pd.DataFrame:
    """Parse a single NCU CSV file into a DataFrame.

    NCU CSV files contain non-CSV lines (==PROF== messages, Python stdout).
    Only lines starting with '"' are actual CSV rows. The second CSV row is a
    units row (e.g. "ns", "block") that must be skipped. Numbers use comma as
    thousands separator. Duration is in nanoseconds.

    Returns a DataFrame with columns: kernel_name, duration_us,
    averaged across ranks for TP>1.
    """
    content = path.read_text()
    csv_lines = [line for line in content.splitlines() if line.startswith('"')]
    assert csv_lines, f"No CSV data found in {path}"
    df = pd.read_csv(
        io.StringIO("\n".join(csv_lines)),
        skiprows=[1],  # skip units row
        thousands=",",
    )
    assert "Kernel Name" in df.columns and "gpu__time_duration.sum" in df.columns, (
        f"Expected columns 'Kernel Name' and 'gpu__time_duration.sum' in {path}, "
        f"got: {list(df.columns)}"
    )
    return (
        df.rename(columns={"Kernel Name": "kernel_name", "gpu__time_duration.sum": "duration_ns"})
        .groupby("kernel_name", as_index=False, sort=False)
        .agg(duration_us=("duration_ns", "mean"))
        .assign(duration_us=lambda d: d["duration_us"] / 1000)
    )


def _print_summary(df: pd.DataFrame) -> None:
    """Print all summary sections from the combined sweep DataFrame."""
    labels = df["method"].unique().tolist()
    baselines = [m for m in labels if m != FUSED]
    has_fused = FUSED in labels

    # Tag each kernel as matmul or sampling
    matmul_patterns = ("gemm", "gemv", "fused_mm_sample")
    df = df.assign(
        is_matmul=df["kernel_name"]
        .str.lower()
        .apply(lambda n: any(p in n for p in matmul_patterns))
    )
    # The first kernel per (bsz, method) must be the matmul
    first_kernels = df.groupby(["bsz", "method"]).first()
    bad = first_kernels.query("not is_matmul")
    assert bad.empty, (
        f"First kernel is not a matmul (expected one of {matmul_patterns} in name):\n"
        + bad[["kernel_name"]].to_string()
    )

    # Per-method totals: bsz x method -> total_us, matmul_us, sampling_us
    totals = (
        df.groupby(["bsz", "method"])
        .apply(
            lambda g: pd.Series(
                {
                    "total_us": g["duration_us"].sum(),
                    "matmul_us": g.query("is_matmul")["duration_us"].sum(),
                    "sampling_us": g.query("not is_matmul")["duration_us"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    # ── Section 1: Total time per method ──
    df_total = totals.pivot(index="bsz", columns="method", values="total_us").round(1)
    df_total = df_total.reindex(columns=labels)
    df_total.index.name = "N"
    print("1. TOTAL TIME (us)")
    print()
    print(df_total.to_markdown())

    # ── Section 2: Speedup vs fused-triton ──
    if has_fused and baselines:
        fused_totals = totals.query("method == @FUSED")[["bsz", "total_us"]].rename(
            columns={"total_us": "fused_us"}
        )
        speedup = (
            totals.query("method != @FUSED")
            .merge(fused_totals, on="bsz")
            .assign(speedup=lambda d: d["total_us"] / d["fused_us"])
        )
        df_speedup = (
            speedup.pivot(index="bsz", columns="method", values="speedup")
            .reindex(columns=baselines)
            .map(lambda x: f"{x:.2f}x" if pd.notna(x) else None)
        )
        df_speedup.index.name = "N"
        print()
        print("2. SPEEDUP (baseline / fused-triton)")
        print()
        print(df_speedup.to_markdown())

    # ── Section 3: Matmul decomposition ──
    if has_fused and baselines:
        first_baseline = baselines[0]  # noqa: F841 (used via pandas @first_baseline query)
        fused_mm = totals.query("method == @FUSED")[["bsz", "matmul_us"]].rename(
            columns={"matmul_us": "fused-triton MM"}
        )
        cublas_mm = totals.query("method == @first_baseline")[["bsz", "matmul_us"]].rename(
            columns={"matmul_us": "cuBLAS MM"}
        )
        df_mm = (
            cublas_mm.merge(fused_mm, on="bsz")
            .assign(**{"MM delta": lambda d: d["fused-triton MM"] - d["cuBLAS MM"]})
            .set_index("bsz")
            .round(0)
            .astype(int)
        )
        df_mm.index.name = "N"
        print()
        print("3. MATMUL DECOMPOSITION (us)")
        print()
        print(df_mm.to_markdown())

    # ── Section 4: Sampling cost per baseline ──
    if baselines:
        bl_totals = totals.query("method in @baselines")
        sampling = bl_totals.assign(
            pct=lambda d: (d["sampling_us"] / d["total_us"] * 100).round(0).astype(int),
        )
        parts = []
        for label in baselines:
            m = sampling.query("method == @label")[["bsz", "sampling_us", "pct"]]
            m = m.rename(columns={"sampling_us": f"{label} (us)", "pct": f"{label} (%)"})
            m[f"{label} (us)"] = m[f"{label} (us)"].round(1)
            m[f"{label} (%)"] = m[f"{label} (%)"].astype(str) + "%"
            parts.append(m.set_index("bsz"))
        df_sampling = pd.concat(parts, axis=1)
        df_sampling.index.name = "N"
        print()
        print("4. SAMPLING COST — mitigated by fused-triton")
        print()
        print(df_sampling.to_markdown())

    # ── Section 5: Net advantage breakdown ──
    if has_fused and baselines:
        fused_t = totals.query("method == @FUSED")[["bsz", "total_us", "matmul_us"]].rename(
            columns={"total_us": "fused_total", "matmul_us": "fused_mm"}
        )
        print()
        print("5. NET ADVANTAGE BREAKDOWN (us)")
        print("   saved = sampling eliminated - matmul overhead")
        for bl_label in baselines:
            bl_t = totals.query("method == @bl_label")[
                ["bsz", "total_us", "matmul_us", "sampling_us"]
            ]
            net = bl_t.merge(fused_t, on="bsz").assign(
                mm_overhead=lambda d: d["fused_mm"] - d["matmul_us"],
                net_saved=lambda d: d["sampling_us"] - (d["fused_mm"] - d["matmul_us"]),
                speedup=lambda d: d["total_us"] / d["fused_total"],
            )
            df_net = (
                net[["bsz", "sampling_us", "mm_overhead", "net_saved", "speedup"]]
                .rename(
                    columns={
                        "sampling_us": "sampling elim",
                        "mm_overhead": "MM overhead",
                        "net_saved": "net saved",
                    }
                )
                .set_index("bsz")
            )
            df_net[["sampling elim", "MM overhead", "net saved"]] = (
                df_net[["sampling elim", "MM overhead", "net saved"]].round(0).astype(int)
            )
            df_net["speedup"] = df_net["speedup"].map(lambda x: f"{x:.2f}x")
            df_net.index.name = "N"
            print()
            print(f"  vs {bl_label}:")
            print()
            print(df_net.to_markdown())

    # ── Section 6: Per-bsz kernel breakdown ──
    multi_kernel = df.groupby(["bsz", "method"]).filter(lambda g: len(g) > 1)
    if not multi_kernel.empty:
        print()
        print("6. KERNEL BREAKDOWNS")
        for (bsz, method), group in multi_kernel.groupby(["bsz", "method"], sort=True):
            kernel_df = (
                group[["duration_us", "kernel_name"]]
                .reset_index(drop=True)
                .rename_axis("#")
                .rename(columns={"duration_us": "Duration (us)", "kernel_name": "Kernel"})
            )
            kernel_df["Duration (us)"] = kernel_df["Duration (us)"].round(1)
            kernel_df["Kernel"] = kernel_df["Kernel"].str[:65]
            print()
            print(f"  {method} (N={bsz}):")
            print()
            print(kernel_df.to_markdown())


if __name__ == "__main__":
    main()

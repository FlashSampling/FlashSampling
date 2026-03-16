import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plot_styles import (
    FLASHSAMPLING_RENAMES,
    PROVIDER_COLORS,
    PROVIDER_HATCHES,
    PROVIDER_MARKERS,
)
from pydantic_settings import BaseSettings

# Must match BENCHMARK_CASES in src/fused_mm_sampling/bench/triton_benchmark_lib.py
BENCHMARK_CASES = {
    "qwen3-1.7b": {"vocab_size": 151_936, "hidden_size": 2_048},
    "small": {"vocab_size": 128_256, "hidden_size": 4_096},
    "large": {"vocab_size": 128_256, "hidden_size": 8_192},
    "gpt-oss-120b": {"vocab_size": 201_088, "hidden_size": 2_880},
}

BYTES_PER_ELEMENT = 2  # bf16
BYTES_PER_INDEX = 8  # int64 (torch.multinomial output)

# Peak HBM/GDDR bandwidth in GB/s per GPU.
GPU_PEAK_BW_GBS: dict[str, float] = {
    # https://www.nvidia.com/en-us/data-center/h100/
    "NVIDIA H100 80GB HBM3": 3350,
    # https://www.nvidia.com/en-us/data-center/h200/
    "NVIDIA H200": 4800,
    # https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf
    "NVIDIA A100-SXM4-80GB": 2039,
    # 64/8=8 in https://resources.nvidia.com/en-us-dgx-systems/dgx-b200-datasheet
    "NVIDIA B200": 8000,
    # 64/8=8 in https://resources.nvidia.com/en-us-dgx-systems/dgx-b300-datasheet
    "NVIDIA B300 SXM6 AC": 8000,
    # https://www.nvidia.com/en-us/data-center/l4/
    "NVIDIA L4": 300,
    # https://www.techpowerup.com/gpu-specs/geforce-rtx-3090.c3622
    "NVIDIA GeForce RTX 3090": 936,
}

# Peak BF16 dense tensor core TFLOP/s per GPU.
GPU_PEAK_COMPUTE_TFLOPS: dict[str, float] = {
    "NVIDIA H100 80GB HBM3": 989,  # 1979 with 2:4 sparsity
    "NVIDIA H200": 989,  # 1979 with 2:4 sparsity
    "NVIDIA A100-SXM4-80GB": 312,
    # https://www.civo.com/blog/comparing-nvidia-b200-and-h100
    "NVIDIA B200": 2250,
    "NVIDIA B300 SXM6 AC": 2250,
    "NVIDIA L4": 121,
    "NVIDIA GeForce RTX 3090": 142,
}


def _provider_palette(providers: pd.Series | list[str]) -> dict[str, str]:
    """Return a color mapping for the providers present in the data."""
    unique = providers if isinstance(providers, list) else providers.unique()
    return {p: PROVIDER_COLORS[p] for p in unique}


def _provider_markers(providers: pd.Series | list[str]) -> dict[str, str]:
    """Return a marker mapping for the providers present in the data."""
    unique = providers if isinstance(providers, list) else providers.unique()
    return {p: PROVIDER_MARKERS[p] for p in unique}


def read_metadata(folder: Path) -> dict:
    """Read metadata.json dumped by the benchmark runner.

    Falls back to args.json and logs.txt for older result directories.
    """
    metadata_file = folder / "metadata.json"
    if metadata_file.exists():
        return json.loads(metadata_file.read_text())

    # defaults for old results that don't have metadata.json
    metadata: dict = {
        "gpu_name": _read_gpu_name_from_logs(folder),
        "device_count": 1,
    }
    return metadata


def _read_gpu_name_from_logs(folder: Path) -> str | None:
    """Parse the GPU name from the 'GPU: ...' line in logs.txt."""
    logs = folder / "logs.txt"
    if not logs.exists():
        return None
    for line in logs.read_text().splitlines():
        m = re.match(r"^GPU:\s*(.+)$", line)
        if m:
            return m.group(1).strip()
    return None


def plot_batch_scaling(bdf_long: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    palette = _provider_palette(bdf_long["provider"])
    markers = _provider_markers(bdf_long["provider"])

    sns.lineplot(
        bdf_long,
        x="n_hidden_states",
        y="time[ms]",
        hue="provider",
        style="provider",
        markers=markers,
        dashes=False,
        ax=ax1,
        palette=palette,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    unique_n_hidden = sorted(bdf_long["n_hidden_states"].unique())
    ax1.set_xticks(unique_n_hidden, labels=[int(x) for x in unique_n_hidden])
    ax1.xaxis.set_minor_locator(plt.NullLocator())
    yfmt = plt.ScalarFormatter()
    yfmt.set_scientific(False)
    ax1.yaxis.set_major_formatter(yfmt)
    ax1.yaxis.set_minor_formatter(yfmt)
    ax1.grid(alpha=0.5, which="both")
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Time (ms)")
    ax1.annotate(
        "lower is better",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
        style="italic",
    )
    ax1.legend_.remove()

    sns.lineplot(
        bdf_long,
        x="n_hidden_states",
        y="samples/ms",
        hue="provider",
        style="provider",
        markers=markers,
        dashes=False,
        ax=ax2,
        palette=palette,
    )
    ax2.set_xscale("log")
    ax2.set_xticks(unique_n_hidden, labels=[int(x) for x in unique_n_hidden])
    ax2.xaxis.set_minor_locator(plt.NullLocator())
    ax2.grid(alpha=0.5)
    ax2.set_xlabel("Batch Size")
    ax2.set_ylabel("Samples/ms")
    ax2.annotate(
        "higher is better",
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=8,
        color="gray",
        style="italic",
    )
    ax2.legend_.remove()

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        title="Method",
        bbox_to_anchor=(0.5, 1.25),
        ncol=2,
    )

    fig.tight_layout()
    return ax1


def plot_relative_performance(
    bdf_rel_long: pd.DataFrame, ref_method: str, show_providers: list[str]
) -> None:
    plot_df = bdf_rel_long.query("provider in @show_providers")
    palette = _provider_palette(show_providers)
    ax = sns.barplot(
        plot_df,
        x="n_hidden_states",
        y="relative-perf",
        hue="provider",
        hue_order=show_providers,
        palette=palette,
    )
    hatches = [PROVIDER_HATCHES.get(p, "") for p in show_providers]
    for container, hatch in zip(ax.containers, hatches):
        for bar in container:
            bar.set_hatch(hatch)
    ax.grid(alpha=0.5, axis="y")
    ncol = 1  # min(len(show_providers), 2)
    sns.move_legend(ax, "upper center", title="Method", bbox_to_anchor=(0.5, 1.35), ncol=ncol)
    for handle, hatch in zip(ax.get_legend().legend_handles, hatches):
        handle.set_hatch(hatch)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Relative Performance")
    ax.set_xticks(ax.get_xticks(), labels=bdf_rel_long["n_hidden_states"].unique().astype(int))
    ax.figure.tight_layout()
    return ax


def model_bytes(vocab_size: int, hidden_size: int, n_hidden_states: float) -> float:
    """Minimum bytes transferred for fused matmul+sampling (read W + read X + write indices)."""
    H = int(n_hidden_states)  # noqa: N806
    return (
        vocab_size * hidden_size * BYTES_PER_ELEMENT  # read weights [V, D]
        + H * hidden_size * BYTES_PER_ELEMENT  # read hidden_states [H, D]
        + H * BYTES_PER_INDEX  # write sampled indices [H]
    )


def model_flops(vocab_size: int, hidden_size: int, n_hidden_states: float) -> float:
    """FLOPs for the fused matmul: 2 * V * D * H."""
    return 2 * vocab_size * hidden_size * int(n_hidden_states)


def assign_col_mem_throughput(df: pd.DataFrame, vocab_size: int, hidden_size: int) -> pd.DataFrame:
    """Add a 'mem_throughput[GB/s]' column: model_bytes / kernel_time."""
    nbytes = df["n_hidden_states"].apply(lambda h: model_bytes(vocab_size, hidden_size, h))
    throughput = nbytes / (df["time[ms]"] / 1000) / 1e9
    return df.assign(**{"mem_throughput[GB/s]": throughput})


def plot_memory_throughput(bdf_long: pd.DataFrame, peak_bw_gbs: float | None = None):
    palette = _provider_palette(bdf_long["provider"])
    markers = _provider_markers(bdf_long["provider"])

    if peak_bw_gbs is not None:
        # Primary axis: Speed-of-Light %, so grid lines align with SoL ticks.
        plot_df = bdf_long.copy()
        plot_df["SoL %"] = plot_df["mem_throughput[GB/s]"] / peak_bw_gbs * 100

        ax = sns.lineplot(
            plot_df,
            x="n_hidden_states",
            y="SoL %",
            hue="provider",
            style="provider",
            markers=markers,
            dashes=False,
            palette=palette,
        )

        ax.axhline(100, color="black", linestyle="--", linewidth=1)
        ax.text(
            0.01,
            100,
            "Peak Memory Bandwidth",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="left",
            fontsize=9,
            color="black",
        )

        ax.set_ylabel("Speed-of-Light %")
        ax.set_ylim(bottom=0, top=110)
        ax.yaxis.set_minor_locator(plt.FixedLocator([10, 30, 50, 70, 90]))

        # Secondary axis: GB/s on the right
        ax2 = ax.secondary_yaxis(
            "right",
            functions=(
                lambda pct: pct / 100 * peak_bw_gbs,
                lambda gbs: gbs / peak_bw_gbs * 100,
            ),
        )
        ax2.set_ylabel("Memory Throughput (GB/s)")
    else:
        ax = sns.lineplot(
            bdf_long,
            x="n_hidden_states",
            y="mem_throughput[GB/s]",
            hue="provider",
            style="provider",
            markers=markers,
            dashes=False,
            palette=palette,
        )
        ax.set_ylabel("Memory Throughput (GB/s)")

    ax.set_xscale("log")
    unique_n_hidden = sorted(bdf_long["n_hidden_states"].unique())
    ax.set_xticks(unique_n_hidden, labels=[int(x) for x in unique_n_hidden])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.grid(alpha=0.5, which="both")
    ax.set_xlabel("Batch Size")

    sns.move_legend(ax, "upper center", title="Method", bbox_to_anchor=(0.5, 1.35), ncol=2)

    ax.figure.tight_layout()
    return ax


def plot_roofline(
    bdf_long: pd.DataFrame,
    vocab_size: int,
    hidden_size: int,
    peak_bw_gbs: float,
    peak_compute_tflops: float,
):
    """Classic roofline plot: achieved TFLOP/s vs arithmetic intensity (FLOP/byte)."""
    df = bdf_long.copy()
    df["flops"] = df["n_hidden_states"].apply(lambda h: model_flops(vocab_size, hidden_size, h))
    df["bytes"] = df["n_hidden_states"].apply(lambda h: model_bytes(vocab_size, hidden_size, h))
    df["ai"] = df["flops"] / df["bytes"]  # arithmetic intensity (FLOP/byte)
    df["achieved_tflops"] = df["flops"] / (df["time[ms]"] / 1000) / 1e12

    fig, ax = plt.subplots(figsize=(7, 4))

    # Roofline ceiling
    ridge_ai = peak_compute_tflops / (peak_bw_gbs / 1000)  # TFLOP/s / (TB/s) = FLOP/byte
    ai_min = df["ai"].min() * 0.5
    ai_max = max(df["ai"].max() * 2, ridge_ai * 2)
    ai_range = np.geomspace(ai_min, ai_max, 200)
    mem_ceiling = peak_bw_gbs / 1000 * ai_range  # TB/s * FLOP/byte = TFLOP/s
    compute_ceiling = np.full_like(ai_range, peak_compute_tflops)
    roofline = np.minimum(mem_ceiling, compute_ceiling)
    ax.plot(ai_range, roofline, color="black", linewidth=2, label="Roofline", zorder=1)

    # Ridge point annotation
    ax.axvline(ridge_ai, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.annotate(
        f"Ridge: AI={ridge_ai:.0f}",
        xy=(ridge_ai, peak_compute_tflops),
        xytext=(ridge_ai * 1.3, peak_compute_tflops * 0.7),
        fontsize=10,
        color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    # Data points per provider
    providers = df["provider"].unique()
    palette = _provider_palette(providers)
    for idx, provider in enumerate(providers):
        color = palette[provider]
        pdf = df[df["provider"] == provider]
        ax.plot(
            pdf["ai"],
            pdf["achieved_tflops"],
            marker=PROVIDER_MARKERS.get(provider, "o"),
            label=provider,
            color=color,
            zorder=3,
        )
        # Annotate each point with H value
        # but only on the last provider
        if idx == len(providers) - 1:
            for _, row in pdf.iterrows():
                ax.annotate(
                    f"bsz={int(row['n_hidden_states'])}",
                    xy=(row["ai"], row["achieved_tflops"]),
                    xytext=(5, -10),
                    textcoords="offset points",
                    fontsize=10,
                    color="black",
                )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)")
    ax.set_ylabel("Achieved Performance (TFLOP/s)")
    ax.minorticks_off()
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return ax


def assign_col_samples_per_ms(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{"samples/ms": lambda df: df["n_hidden_states"] / df["time[ms]"]})


def read_triton_bench_csv(path: Path) -> pd.DataFrame:
    """Read a Triton benchmark CSV, stripping the ' (Time (ms))' column suffix."""
    df = pd.read_csv(path)
    df.columns = [c.removesuffix(" (Time (ms))") for c in df.columns]
    return df


def plot_relative_performance_from_wide(
    bdf: pd.DataFrame,
    ref_method: str,
    ref_slug: str,
    show_providers: list[str],
    case: str,
    plot_folder: Path,
    csv_folder: Path,
    fmt: str = "png",
):
    """Compute relative performance vs ref_method and save plot + CSV."""
    methods = [c for c in bdf.columns if c != ref_method and c != "n_hidden_states"]
    bdf_rel = bdf.copy()
    bdf_rel[[*methods, ref_method]] = bdf[[*methods, ref_method]].div(bdf[ref_method], axis=0)
    bdf_rel_long = bdf_rel.melt(
        id_vars=["n_hidden_states"], var_name="provider", value_name="relative-time"
    )
    bdf_rel_long["relative-perf"] = 1 / bdf_rel_long["relative-time"]
    bdf_rel_long.round(3).to_csv(
        csv_folder / f"relative-performance-vs-{ref_slug}-{case}.csv", index=False
    )
    ax = plot_relative_performance(bdf_rel_long, ref_method, show_providers)
    ax.figure.savefig(
        plot_folder / f"relative-performance-vs-{ref_slug}-{case}.{fmt}",
        dpi=300,
        bbox_inches="tight",
    )
    return ax


def create_and_triton_bench_plots(
    folder: Path,
    fmt: str = "png",
    use_name_flashsampling: bool = False,
    skip_multinomial_eager: bool = False,
):
    tgt_folder = folder / "custom-plots"
    tgt_folder.mkdir(parents=True, exist_ok=True)

    metadata = read_metadata(folder)
    gpu_name = metadata["gpu_name"]
    n_gpus = metadata["device_count"]
    peak_bw_gbs = GPU_PEAK_BW_GBS[gpu_name] * n_gpus
    peak_compute_tflops = GPU_PEAK_COMPUTE_TFLOPS[gpu_name] * n_gpus
    tp_suffix = f" x {n_gpus} GPUs" if n_gpus > 1 else ""
    print(
        f"GPU: {gpu_name}{tp_suffix} → peak HBM BW: {peak_bw_gbs} GB/s, peak compute: {peak_compute_tflops} TFLOP/s"
    )

    csv_prefix = "fused-mm-sample-batch-scaling-"
    for csv_path in sorted(folder.glob(f"{csv_prefix}*.csv")):
        case = csv_path.stem.removeprefix(csv_prefix)
        print(f"Plotting case: {case}")

        bdf = read_triton_bench_csv(csv_path)
        if skip_multinomial_eager:
            bdf = bdf.drop(columns=["Multinomial Sampling (Eager)"], errors="ignore")
        if use_name_flashsampling:
            bdf = apply_flashsampling_rename(bdf)
        bdf_long = bdf.melt(id_vars=["n_hidden_states"], var_name="provider", value_name="time[ms]")
        bdf_long = assign_col_samples_per_ms(bdf_long)

        ax = plot_batch_scaling(bdf_long)
        ax.figure.savefig(tgt_folder / f"batch-scaling-{case}.{fmt}", dpi=300, bbox_inches="tight")
        plt.close(ax.figure)

        # Memory throughput plot
        if case in BENCHMARK_CASES:
            cfg = BENCHMARK_CASES[case]
            bdf_mem = bdf_long.pipe(
                assign_col_mem_throughput, cfg["vocab_size"], cfg["hidden_size"]
            )
            ax = plot_memory_throughput(bdf_mem, peak_bw_gbs=peak_bw_gbs)
            ax.figure.savefig(
                tgt_folder / f"memory-throughput-{case}.{fmt}", dpi=300, bbox_inches="tight"
            )
            plt.close(ax.figure)

            if peak_bw_gbs is not None and peak_compute_tflops is not None:
                ax = plot_roofline(
                    bdf_long,
                    cfg["vocab_size"],
                    cfg["hidden_size"],
                    peak_bw_gbs,
                    peak_compute_tflops,
                )
                ax.figure.savefig(
                    tgt_folder / f"roofline-{case}.{fmt}", dpi=300, bbox_inches="tight"
                )
                plt.close(ax.figure)

        FMMS = "FMMS (Triton)"  # noqa: N806
        if use_name_flashsampling:
            FMMS = FLASHSAMPLING_RENAMES[FMMS]  # noqa: N806
        NAIVE = "Multinomial Sampling (Compiled)"  # noqa: N806
        FI_SAMPLE = "flashinfer:sampling_from_logits"  # noqa: N806
        FI_TOPK = "flashinfer:top_k_top_p_sampling_from_logits"  # noqa: N806

        rel_plots = [
            # (1) FMMS vs PyTorch Compiled (baseline)
            {"ref": NAIVE, "slug": "pytorch", "show": [FMMS, NAIVE]},
            # (2) FMMS vs both FlashInfer kernels (top_k_top_p as baseline)
            {"ref": FI_TOPK, "slug": "flashinfer", "show": [FMMS, FI_SAMPLE, FI_TOPK]},
        ]
        for rp in rel_plots:
            if rp["ref"] not in bdf.columns:
                continue
            show = [p for p in rp["show"] if p in bdf.columns]
            ax = plot_relative_performance_from_wide(
                bdf, rp["ref"], rp["slug"], show, case, tgt_folder, folder, fmt=fmt
            )
            plt.close(ax.figure)


def apply_flashsampling_rename(df: pd.DataFrame) -> pd.DataFrame:
    """Rename FMMS columns/values to FlashSampling equivalents."""
    return df.rename(columns=FLASHSAMPLING_RENAMES)


class Args(BaseSettings, cli_parse_args=True):
    tgt_dir: Path = Path(__file__).parent / "profiles/triton-bench/"
    fmt: str = "png"
    use_name_flashsampling: bool = False
    skip_multinomial_eager: bool = False


if __name__ == "__main__":
    args = Args()
    create_and_triton_bench_plots(
        args.tgt_dir,
        fmt=args.fmt,
        use_name_flashsampling=args.use_name_flashsampling,
        skip_multinomial_eager=args.skip_multinomial_eager,
    )

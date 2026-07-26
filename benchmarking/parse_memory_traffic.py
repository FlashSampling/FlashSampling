import argparse
import io
from pathlib import Path

import pandas as pd

BYTE_SCALES = {
    "byte": 1,
    "Kbyte": 10**3,
    "Mbyte": 10**6,
    "Gbyte": 10**9,
    "Tbyte": 10**12,
}


def main() -> None:
    args = parse_args()
    traffic = pd.concat(
        [read_traffic_csv(path) for path in sorted(args.results_dir.glob("*/traffic.csv"))],
        ignore_index=True,
    )
    memory = pd.concat(
        [
            read_memory_json(path)
            for path in sorted(args.results_dir.glob("*/memory.json"))
        ],
        ignore_index=True,
    )
    memory["peak_temporary_bytes"] = pd.to_numeric(memory["peak_temporary_bytes"])
    summary = (
        traffic.groupby("provider", as_index=False)
        .agg(
            hbm_read_bytes=("dram__bytes_read.sum", "sum"),
            hbm_write_bytes=("dram__bytes_write.sum", "sum"),
        )
        .merge(
            memory.loc[:, ["provider", "peak_temporary_bytes"]],
            on="provider",
            validate="one_to_one",
        )
    )
    summary["hbm_read_gib"] = summary["hbm_read_bytes"] / 2**30
    summary["hbm_write_mib"] = summary["hbm_write_bytes"] / 2**20
    summary["peak_temporary_mib"] = summary["peak_temporary_bytes"] / 2**20
    fused = summary.query("provider == 'fused-triton'").iloc[0]
    summary["fmms_read_reduction_pct"] = (
        1 - fused["hbm_read_bytes"] / summary["hbm_read_bytes"]
    ) * 100
    summary["fmms_write_reduction_pct"] = (
        1 - fused["hbm_write_bytes"] / summary["hbm_write_bytes"]
    ) * 100
    summary["fmms_peak_memory_reduction_pct"] = (
        1 - fused["peak_temporary_bytes"] / summary["peak_temporary_bytes"]
    ) * 100
    summary.to_csv(args.results_dir / "summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


def read_memory_json(path: Path) -> pd.DataFrame:
    return pd.read_json(path, typ="series").to_frame().T.assign(provider=path.parent.name)


def read_traffic_csv(path: Path) -> pd.DataFrame:
    csv_lines = [line for line in path.read_text().splitlines() if line.startswith('"')]
    df = pd.read_csv(io.StringIO("\n".join(csv_lines)))
    metric_columns = ["dram__bytes_read.sum", "dram__bytes_write.sum"]
    byte_scales = df.loc[0, metric_columns].map(BYTE_SCALES)
    df[metric_columns] = (
        df[metric_columns]
        .apply(pd.to_numeric, errors="coerce")
        .mul(byte_scales, axis="columns")
    )
    df = df.dropna(subset=["Kernel Name", *metric_columns])
    provider = path.parent.name
    return df.loc[:, ["Kernel Name", *metric_columns]].assign(provider=provider)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    main()

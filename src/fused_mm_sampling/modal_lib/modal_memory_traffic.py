import os
import subprocess

import modal

from .utils import (
    ModalEnvConfig,
    make_app,
    make_ncu_image,
    make_volumes,
    set_volume_caches,
    volume_path,
)


cfg = ModalEnvConfig()
app = make_app()

ncu_image = make_ncu_image()


@app.function(
    gpu=cfg.gpu_spec,
    image=ncu_image,
    volumes=make_volumes(),
    timeout=cfg.timeout,
)
def measure_memory_traffic(
    name: str,
    output_name: str,
    case: str,
    n_hidden_states: int,
) -> None:
    set_volume_caches()
    results_dir = (
        f"{volume_path}/memory-traffic/{cfg.gpu}/case-{case}/bsz{n_hidden_states}"
    )
    os.makedirs(results_dir, exist_ok=True)
    provider_dir = f"{results_dir}/{output_name}"
    os.makedirs(provider_dir, exist_ok=True)
    report_base = f"{provider_dir}/report"
    report_path = f"{report_base}.ncu-rep"
    csv_path = f"{provider_dir}/traffic.csv"
    memory_path = f"{provider_dir}/memory.json"
    cmd = [
        "ncu",
        "--metrics",
        "dram__bytes_read.sum,dram__bytes_write.sum",
        "--nvtx",
        "--nvtx-include",
        "kernel/",
        "--target-processes",
        "all",
        "-fo",
        report_base,
        "python",
        "/opt/fmms/memory_traffic.py",
        "--name",
        name,
        "--case",
        case,
        "--n-hidden-states",
        str(n_hidden_states),
        "--memory-output",
        memory_path,
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    export_cmd = ["ncu", "--import", report_path, "--csv", "--page", "raw"]
    print(f"Exporting: {' '.join(export_cmd)}")
    with open(csv_path, "w") as csv_file:
        subprocess.run(export_cmd, stdout=csv_file, check=True)

    modal.Volume.from_name("fused-mm-sample").commit()


@app.local_entrypoint()
def main(
    name: str,
    output_name: str,
    case: str = "large",
    n_hidden_states: int = 64,
) -> None:
    measure_memory_traffic.remote(name, output_name, case, n_hidden_states)

import os
import subprocess

import modal

from .utils import (
    ModalEnvConfig,
    add_library_code,
    make_app,
    make_image,
    make_volumes,
    set_volume_caches,
    volume_path,
)


class Config(ModalEnvConfig):
    ncu_mode: str = "export"


cfg = Config()
app = make_app()

ncu_image = add_library_code(make_image()).run_commands(
    "apt-get update && apt-get install -y cuda-nsight-compute-13-2",
    # Put the new NCU first on PATH so it shadows the old one
    "ln -sf /opt/nvidia/nsight-compute/2026.1.0/ncu /usr/local/bin/ncu",
)


@app.function(gpu=cfg.gpu_spec, image=ncu_image, volumes=make_volumes(), timeout=cfg.timeout)
def ncu_run(name: str, n_hidden_states: str, case: str, n_procs: int, mode: str, gpu_name: str):
    set_volume_caches()
    cmd = [
        "ncu",
        "--set",
        "full",
        "--target-processes",
        "all",
    ]
    if n_procs > 1:
        # Symmetric memory breaks kernel replay (can't save/restore memory).
        # Use application replay (re-runs the whole app per metric pass) and
        # filter by kernel name instead of NVTX (NVTX ranges aren't stable
        # across application re-runs due to autotuning).
        cmd += [
            "--replay-mode",
            "application",
            "-k",
            "fused_mm_sample_triton_kernel",
        ]
    else:
        cmd += ["--nvtx", "--nvtx-include", "kernel/"]
    if mode == "profile":
        ncu_dir = f"{volume_path}/ncu-rep/{gpu_name}/tp{n_procs}/case-{case}/bsz{n_hidden_states}"
        out_path = f"{ncu_dir}/{name}"
        os.makedirs(ncu_dir, exist_ok=True)
        cmd += [
            "--source-folders",
            "/opt/fmms/src/fused_mm_sampling",
            "--import-source",
            "yes",
            "-fo",
            out_path,
        ]
    cmd += [
        "python",
        "/opt/fmms/speed_test.py",
        "--name",
        name,
        "--n_hidden_states",
        n_hidden_states,
        "--n_runs_benchmark",
        "1",
        "--bench_fn=own",
        f"--case={case}",
        f"--n_procs={n_procs}",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("=== stdout ===")
    print(result.stdout)
    print("=== stderr ===")
    print(result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
    print(f"=== returncode: {result.returncode} ===")
    if mode == "profile":
        modal.Volume.from_name("fused-mm-sample").commit()


@app.local_entrypoint()
def main():
    ncu_run.remote(
        name=cfg.name,
        n_hidden_states=str(cfg.n_hidden_states),
        case=cfg.case,
        n_procs=cfg.n_procs,
        mode=cfg.ncu_mode,
        gpu_name=cfg.gpu,
    )

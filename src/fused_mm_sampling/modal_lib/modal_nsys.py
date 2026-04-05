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
    postfix: str = ""
    n_runs_benchmark: int = 5


cfg = Config()
app = make_app()

nsys_image = add_library_code(make_image().apt_install("cuda-nsight-systems-13-0"))


@app.function(gpu=cfg.gpu_spec, image=nsys_image, volumes=make_volumes(), timeout=cfg.timeout)
def nsys_profile(
    name: str,
    n_hidden_states: str,
    case: str,
    n_procs: int,
    gpu_name: str,
    postfix: str,
    n_runs_benchmark: str,
):
    set_volume_caches()
    report_dir = f"{volume_path}/nsys-profiles/{gpu_name}/tp{n_procs}/case-{case}/bsz{n_hidden_states}{postfix}"
    os.makedirs(report_dir, exist_ok=True)
    report_path = f"{report_dir}/{name}"

    speed_test_args = [
        "/opt/fmms/speed_test.py",
        "--name",
        name,
        "--n_hidden_states",
        n_hidden_states,
        "--n_runs_warmup",
        "25",
        "--n_runs_benchmark",
        n_runs_benchmark,
        "--bench_fn=own",
        f"--case={case}",
        f"--n_procs={n_procs}",
        "--nsys_profile=true",
    ]

    if n_procs > 1:
        # Per-rank nsys: torchrun launches the wrapper, each rank gets its
        # own nsys instance and .nsys-rep file. This avoids the known nsys
        # limitation where cudaProfilerApi capture misses child-process GPU
        # activity when wrapping torchrun from the outside.
        cmd = [
            "torchrun",
            f"--nproc_per_node={n_procs}",
            "/opt/fmms/nsys_wrapper.py",
            "--report-dir",
            report_dir,
            "--name",
            name,
            "--",
            *speed_test_args,
        ]
        env = None
    else:
        cmd = [
            "nsys",
            "profile",
            "-o",
            report_path,
            "--force-overwrite=true",
            "--capture-range=cudaProfilerApi",
            "--cuda-memory-usage=true",
            "python",
            *speed_test_args,
        ]
        env = None

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    print("=== stdout ===")
    print(result.stdout)
    print("=== stderr ===")
    print(result.stderr)
    print(f"=== returncode: {result.returncode} ===")

    if n_procs > 1:
        report_files = [f"{report_dir}/{name}-rank{r}.nsys-rep" for r in range(n_procs)]
    else:
        report_files = [f"{report_path}.nsys-rep"]

    for report_file in report_files:
        if os.path.exists(report_file):
            size_mb = os.path.getsize(report_file) / (1024 * 1024)
            print(f"Report saved: {report_file} ({size_mb:.1f} MB)")
        else:
            print(f"Report file not found at {report_file}")

    modal.Volume.from_name("fused-mm-sample").commit()


@app.local_entrypoint()
def main():
    nsys_profile.remote(
        name=cfg.name,
        n_hidden_states=str(cfg.n_hidden_states),
        case=cfg.case,
        n_procs=cfg.n_procs,
        gpu_name=cfg.gpu,
        postfix=cfg.postfix,
        n_runs_benchmark=str(cfg.n_runs_benchmark),
    )

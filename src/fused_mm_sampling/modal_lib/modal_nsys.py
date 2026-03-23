import os
import subprocess

import modal

from .utils import add_library_code, make_app, make_image, make_volumes, set_volume_caches, volume_path

app = make_app()

gpu = os.getenv("GPU", "B200")
name = os.getenv("NAME", "fused-triton")
n_hidden_states = os.getenv("N_HIDDEN_STATES", "1")
case = os.getenv("CASE", "small")
n_procs = int(os.getenv("N_PROCS", "1"))
postfix = os.getenv("POSTFIX", "")
n_runs_benchmark = os.getenv("N_RUNS_BENCHMARK", "5")

gpu_spec = f"{gpu}:{n_procs}" if n_procs > 1 else gpu

nsys_image = add_library_code(make_image()).apt_install("cuda-nsight-systems-13-0")


@app.function(gpu=gpu_spec, image=nsys_image, volumes=make_volumes(), timeout=15 * 60)
def nsys_profile(
    name: str, n_hidden_states: str, case: str, n_procs: int, gpu_name: str,
    postfix: str, n_runs_benchmark: str,
):
    set_volume_caches()
    report_dir = f"{volume_path}/nsys-profiles/{gpu_name}/tp{n_procs}/case-{case}/bsz{n_hidden_states}{postfix}"
    os.makedirs(report_dir, exist_ok=True)
    report_path = f"{report_dir}/{name}"

    cmd = [
        "nsys", "profile",
        "-o", report_path,
        "--force-overwrite=true",
        "--capture-range=cudaProfilerApi",
        "--cuda-memory-usage=true",
        "--env-var", "FMMS_CUDA_PROFILER=1",
        "python", "/opt/fmms/speed_test.py",
        "--name", name,
        "--n_hidden_states", n_hidden_states,
        "--n_runs_warmup", "3",
        "--n_runs_benchmark", n_runs_benchmark,
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

    report_file = f"{report_path}.nsys-rep"
    if os.path.exists(report_file):
        size_mb = os.path.getsize(report_file) / (1024 * 1024)
        print(f"Report saved: {report_file} ({size_mb:.1f} MB)")
        modal.Volume.from_name("fused-mm-sample").commit()
    else:
        print(f"Report file not found at {report_file}")


@app.local_entrypoint()
def main():
    nsys_profile.remote(
        name=name,
        n_hidden_states=n_hidden_states,
        case=case,
        n_procs=n_procs,
        gpu_name=gpu,
        postfix=postfix,
        n_runs_benchmark=n_runs_benchmark,
    )

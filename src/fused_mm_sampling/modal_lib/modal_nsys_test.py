import os
import subprocess

from .utils import make_app, make_image, make_volumes, volume_path

app = make_app()

REPORT_DIR = f"{volume_path}/nsys-test"
REPORT_PATH = f"{REPORT_DIR}/report"


nsys_image = make_image().apt_install("cuda-nsight-systems-13-0")


@app.function(gpu="B200", image=nsys_image, volumes=make_volumes())
def nsys_test():
    os.makedirs(REPORT_DIR, exist_ok=True)

    script = """
import torch

a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
c = a + b
torch.cuda.synchronize()
"""
    with open("/tmp/nsys_kernel.py", "w") as f:
        f.write(script)

    result = subprocess.run(
        [
            "nsys",
            "profile",
            "-o",
            REPORT_PATH,
            "--force-overwrite=true",
            "python",
            "/tmp/nsys_kernel.py",
        ],
        capture_output=True,
        text=True,
    )
    print("=== stdout ===")
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    print("=== stderr ===")
    print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
    print(f"=== returncode: {result.returncode} ===")

    report_file = f"{REPORT_PATH}.nsys-rep"
    if os.path.exists(report_file):
        size_mb = os.path.getsize(report_file) / (1024 * 1024)
        print(f"Report saved: {report_file} ({size_mb:.1f} MB)")
        from modal import Volume

        Volume.from_name("fused-mm-sample").commit()
    else:
        print(f"Report file not found at {report_file}")


@app.local_entrypoint()
def main():
    nsys_test.remote()

import subprocess

from .utils import make_app, make_image

app = make_app()


@app.function(gpu="B200", image=make_image())
def ncu_test():
    script = """
import torch

a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
c = a + b
torch.cuda.synchronize()
"""
    with open("/tmp/ncu_kernel.py", "w") as f:
        f.write(script)

    result = subprocess.run(
        [
            "ncu",
            "--target-processes",
            "all",
            "--set",
            "basic",
            "python",
            "/tmp/ncu_kernel.py",
        ],
        capture_output=True,
        text=True,
    )
    print("=== stdout ===")
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    print("=== stderr ===")
    print(result.stderr[-3000:] if len(result.stderr) > 3000 else result.stderr)
    print(f"=== returncode: {result.returncode} ===")


@app.local_entrypoint()
def main():
    ncu_test.remote()

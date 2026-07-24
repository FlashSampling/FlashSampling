import os
import subprocess
import sys

from .utils import ModalEnvConfig, make_app, make_image, make_volumes, set_volume_caches

cfg = ModalEnvConfig()
app = make_app()


@app.function(gpu=cfg.gpu_spec, image=make_image(), volumes=make_volumes(), timeout=cfg.timeout)
def modal_pytest_distributed():
    set_volume_caches()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={cfg.n_procs}",
            "-m",
            "src.fused_mm_sampling.modal_lib.modal_pytest_distributed_worker",
        ],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main():
    modal_pytest_distributed.remote()

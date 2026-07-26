import os
import subprocess
import sys

from .utils import ModalEnvConfig, make_app, make_image, make_volumes, set_volume_caches


class Config(ModalEnvConfig):
    tgt_dir: str | None = None
    disable_compile: bool = False


cfg = Config()
app = make_app()


@app.function(gpu=cfg.gpu_spec, image=make_image(), volumes=make_volumes(), timeout=cfg.timeout)
def function(
    tgt_dir: str | None,
    case: str,
    n_procs: int,
    name: str | None,
    disable_compile: bool,
    bench_fn: str,
):
    set_volume_caches()
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "FMMS_TGT_DIR": tgt_dir or "",
            "FMMS_CASE": case,
            "FMMS_N_PROCS": str(n_procs),
            "FMMS_NAME": name or "",
            "FMMS_DISABLE_COMPILE": str(int(disable_compile)),
            "FMMS_BENCH_FN": bench_fn,
        }
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--numa-binding=node",
            f"--nproc-per-node={n_procs}",
            "-m",
            "src.fused_mm_sampling.modal_lib.modal_triton_benchmark_worker",
        ],
        check=True,
        env=env,
    )


@app.local_entrypoint()
def main():
    function.remote(
        tgt_dir=cfg.tgt_dir,
        case=cfg.case,
        n_procs=cfg.n_procs,
        name=cfg.name,
        disable_compile=cfg.disable_compile,
        bench_fn=cfg.bench_fn,
    )

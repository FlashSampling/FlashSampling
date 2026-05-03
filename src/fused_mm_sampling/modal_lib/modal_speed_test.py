from pathlib import Path

from ..bench.speed_test import run_speed_test
from ..bench.triton_benchmark_lib import Args
from .utils import (
    ModalEnvConfig,
    make_app,
    make_image,
    make_volumes,
    set_volume_caches,
    volume_path,
)

cfg = ModalEnvConfig()
app = make_app()


@app.function(gpu=cfg.gpu_spec, image=make_image(), volumes=make_volumes(), timeout=cfg.timeout)
def speed_test(args: Args):
    set_volume_caches()
    run_speed_test(args)


@app.local_entrypoint()
def main():
    args = Args(
        n_hidden_states=cfg.n_hidden_states,
        n_procs=cfg.n_procs,
        name=cfg.name,
        bench_fn=cfg.bench_fn,
        case=cfg.case,
        tgt_dir=Path(volume_path) / "speed-test",
    )
    speed_test.remote(args=args)

from ..bench.triton_benchmark_lib import Args, run_triton_bechmark
from .utils import ModalEnvConfig, make_app, make_image, make_volumes, set_volume_caches


class Config(ModalEnvConfig):
    tgt_dir: str | None = None
    disable_compile: bool = False


cfg = Config()
app = make_app()


@app.function(gpu=cfg.gpu_spec, image=make_image(), volumes=make_volumes(), timeout=cfg.timeout)
def function(args: Args):
    set_volume_caches()
    run_triton_bechmark(args)


@app.local_entrypoint()
def main():
    args = Args(
        tgt_dir=cfg.tgt_dir,
        case=cfg.case,
        n_procs=cfg.n_procs,
        name=cfg.name,
        disable_compile=cfg.disable_compile,
        bench_fn=cfg.bench_fn,
    )
    function.remote(args=args)

import os

from ..bench.triton_benchmark_lib import Args, run_triton_bechmark
from .utils import make_app, make_image, make_volumes, set_volume_caches

app = make_app()

gpu = os.getenv("GPU")
tgt_dir = os.getenv("TGT_DIR")
case = os.getenv("CASE", "all")
n_procs = int(os.getenv("N_PROCS", "1"))
name = os.getenv("NAME") or None

gpu_spec = f"{gpu}:{n_procs}" if n_procs > 1 else gpu


@app.function(gpu=gpu_spec, image=make_image(), volumes=make_volumes(), timeout=20 * 60)
def function(args: Args):
    set_volume_caches()
    run_triton_bechmark(args)


@app.local_entrypoint()
def main():
    args = Args(tgt_dir=tgt_dir, case=case, n_procs=n_procs, name=name)
    function.remote(args=args)

import os

from ..bench.triton_benchmark_lib import Args, run_triton_bechmark
from .utils import make_app, make_image, make_volumes

app = make_app()

gpu = os.getenv("GPU")
tgt_dir = os.getenv("TGT_DIR")
case = os.getenv("CASE", "all")


@app.function(gpu=gpu, image=make_image(), volumes=make_volumes(), timeout=10 * 60)
def function(args: Args):
    run_triton_bechmark(args)


@app.local_entrypoint()
def main():
    args = Args(tgt_dir=tgt_dir, case=case)
    function.remote(args=args)

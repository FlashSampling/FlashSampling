from ..bench.matmul_comparison import matmul_comparison_main
from .utils import ModalEnvConfig, make_app, make_image, make_volumes

cfg = ModalEnvConfig()
app = make_app()


@app.function(gpu=cfg.gpu_spec, image=make_image(), volumes=make_volumes(), timeout=cfg.timeout)
def my_func():
    matmul_comparison_main()


@app.local_entrypoint()
def main():
    my_func.remote()

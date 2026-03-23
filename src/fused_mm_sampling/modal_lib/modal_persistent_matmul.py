from ..persistent_matmul import main as persistent_matmul_main
from .utils import ModalEnvConfig, make_app, make_image, make_volumes

cfg = ModalEnvConfig()
app = make_app()


@app.function(gpu=cfg.gpu_spec, image=make_image(), volumes=make_volumes(), timeout=cfg.timeout)
def speed_test():
    persistent_matmul_main()


@app.local_entrypoint()
def main():
    speed_test.remote()

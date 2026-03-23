import os
from pathlib import Path

from ..bench.speed_test import Args, run_speed_test
from .utils import make_app, make_image, make_volumes, set_volume_caches, volume_path

gpu = os.getenv("GPU", "H100")
n_procs = int(os.getenv("N_PROCS", "1"))
name = os.getenv("NAME") or None
n_hidden_states = int(os.getenv("N_HIDDEN_STATES", "4"))

gpu_spec = f"{gpu}:{n_procs}" if n_procs > 1 else gpu
app = make_app()


@app.function(gpu=gpu_spec, image=make_image(), volumes=make_volumes(), timeout=20 * 60)
def speed_test(args: Args):
    set_volume_caches()
    run_speed_test(args)


@app.local_entrypoint()
def main():
    args = Args(
        n_hidden_states=n_hidden_states,
        n_procs=n_procs,
        name=name,
        tgt_dir=Path(volume_path) / "speed-test",
    )
    speed_test.remote(args=args)

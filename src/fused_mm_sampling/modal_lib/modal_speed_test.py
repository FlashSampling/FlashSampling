from pathlib import Path

from ..bench.speed_test import Args, run_speed_test
from .utils import make_app, make_image, make_volumes, set_volume_caches, volume_path

args = Args(n_hidden_states=4, tgt_dir=Path(volume_path) / "speed-test")
app = make_app()


@app.function(gpu="H100", image=make_image(), volumes=make_volumes())
def speed_test(args: Args):
    set_volume_caches()
    run_speed_test(args)


@app.local_entrypoint()
def main():
    speed_test.remote(args=args)

import os

from ..testing import run_greedy_tp2, run_sampling_distribution_tp2
from ..tp_info import run_maybe_distributed
from .utils import make_app, make_image, make_volumes, set_volume_caches

app = make_app()

gpu = os.getenv("GPU", "b200")


@app.function(gpu=f"{gpu}:2", image=make_image(), volumes=make_volumes(), timeout=10 * 60)
def modal_pytest_distributed():
    set_volume_caches()
    print("=== test_sampling_distribution_tp2 ===")
    run_maybe_distributed(run_sampling_distribution_tp2, n_procs=2)
    print("\n=== test_greedy_tp2 ===")
    run_maybe_distributed(run_greedy_tp2, n_procs=2)
    print("\nAll distributed tests passed.")


@app.local_entrypoint()
def main():
    modal_pytest_distributed.remote()

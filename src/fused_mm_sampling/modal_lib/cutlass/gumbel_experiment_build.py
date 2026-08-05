"""Single-writer cache-fill gate for one CUTLASS sampling experiment."""

from ...cutlass_experiments import get_cutlass_sampling_experiment
from ...dev_metrics import emit_dev_event, timed_dev_stage
from ..utils import (
    commit_shared_volume,
    make_app,
    make_volumes,
    reload_shared_volume,
    set_volume_caches,
)
from .utils import add_cutlass_greedy_provider, make_cutlass_provider_image

app = make_app()
image = add_cutlass_greedy_provider(make_cutlass_provider_image())


@app.function(gpu="B200", image=image, volumes=make_volumes(), timeout=60 * 60)
def build_sm100(variant: str) -> dict:
    from fused_mm_sampling import cutlass_impl

    reload_shared_volume()
    set_volume_caches()
    get_cutlass_sampling_experiment(variant)
    emit_dev_event("remote_start", stage="experiment_build", variant=variant)
    module = cutlass_impl._get_experimental_sampling_module(variant)
    with timed_dev_stage("volume_commit", variant=variant):
        commit_shared_volume()
    emit_dev_event(
        "cache_published",
        variant=variant,
        binary_path=str(module.__file__),
    )
    return {"variant": variant, "binary_path": str(module.__file__)}


@app.local_entrypoint()
def main(variant: str) -> None:
    get_cutlass_sampling_experiment(variant)
    result = build_sm100.remote(variant)
    print(
        f"Published CUTLASS experiment cache for {result['variant']}: "
        f"{result['binary_path']}"
    )

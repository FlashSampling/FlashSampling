import os

# Inductor compile-worker subprocesses don't import this package, so they
# can't apply the triton.backends backstop in ``src/fused_mm_sampling/__init__.py``
# and re-trip the Modal pytorch:2.10/cuda13 discovery flake as
# ``Could not find an active GPU backend`` from ``triton_helpers.set_driver_to_gpu()``.
# Force serial inductor compilation so all generated modules load in this user
# worker. Scoped to the pytest-distributed entry point only — vLLM-bench and
# friends keep parallel compile.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

from ..testing import verify_correctness_tp2  # noqa: E402
from ..tp_info import run_maybe_distributed  # noqa: E402
from .utils import set_volume_caches  # noqa: E402


def main() -> None:
    set_volume_caches()
    run_maybe_distributed(verify_correctness_tp2, n_procs=2)


if __name__ == "__main__":
    main()

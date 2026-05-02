import os
import site

# Inductor compile-worker subprocesses are spawned as fresh Python processes
# that don't import this package, so they can't apply the triton.backends
# backstop in ``src/fused_mm_sampling/__init__.py`` and re-trip the Modal
# pytorch:2.10/cuda13 discovery flake as ``Could not find an active GPU
# backend`` from ``triton_helpers.set_driver_to_gpu()``. Drop a .pth file in
# system site-packages so every Python process (including inductor's compile
# workers) applies the backstop at interpreter startup, regardless of how the
# subprocess customizes ``PYTHONPATH``/``env``.

_BACKSTOP_MODULE_NAME = "_fmms_triton_backstop"
_BACKSTOP_CODE = """\
try:
    import triton.backends as _tb
    from triton.backends import Backend
    from triton.backends.nvidia.compiler import CUDABackend
    from triton.backends.nvidia.driver import CudaDriver
    _tb.backends.setdefault("nvidia", Backend(CUDABackend, CudaDriver))
except Exception:
    pass
"""
_site_dir = site.getsitepackages()[0]
with open(os.path.join(_site_dir, f"{_BACKSTOP_MODULE_NAME}.py"), "w") as f:
    f.write(_BACKSTOP_CODE)
with open(os.path.join(_site_dir, f"{_BACKSTOP_MODULE_NAME}.pth"), "w") as f:
    f.write(f"import {_BACKSTOP_MODULE_NAME}\n")

# Belt-and-suspenders: keep inductor's compile thread count at 1 to reduce the
# number of subprocess workers that need the backstop applied.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

from ..testing import verify_correctness_tp2  # noqa: E402
from ..tp_info import run_maybe_distributed  # noqa: E402
from .utils import set_volume_caches  # noqa: E402


def main() -> None:
    set_volume_caches()
    run_maybe_distributed(verify_correctness_tp2, n_procs=2)


if __name__ == "__main__":
    main()

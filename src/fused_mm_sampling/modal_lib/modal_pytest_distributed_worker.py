import os

from ..testing import verify_correctness_tp2
from ..tp_info import run_maybe_distributed
from .utils import set_volume_caches


def _run() -> None:
    providers_csv = os.environ.get("FMMS_PROVIDERS_CSV")
    verify_correctness_tp2(providers_csv)


def main() -> None:
    set_volume_caches()
    run_maybe_distributed(_run, n_procs=2)


if __name__ == "__main__":
    main()

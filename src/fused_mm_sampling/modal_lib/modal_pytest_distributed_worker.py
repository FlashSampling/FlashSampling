import os

from ..testing import verify_correctness_tp
from ..tp_info import run_maybe_distributed
from .utils import set_volume_caches


def main() -> None:
    set_volume_caches()
    run_maybe_distributed(
        verify_correctness_tp,
        n_procs=int(os.environ["WORLD_SIZE"]),
    )


if __name__ == "__main__":
    main()

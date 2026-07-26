import os

from ..bench.triton_benchmark_lib import Args, run_triton_bechmark


def main() -> None:
    args = Args(
        tgt_dir=os.environ["FMMS_TGT_DIR"] or None,
        case=os.environ["FMMS_CASE"],
        n_procs=int(os.environ["FMMS_N_PROCS"]),
        name=os.environ["FMMS_NAME"] or None,
        disable_compile=bool(int(os.environ["FMMS_DISABLE_COMPILE"])),
        bench_fn=os.environ["FMMS_BENCH_FN"],
    )
    run_triton_bechmark(args)


if __name__ == "__main__":
    main()

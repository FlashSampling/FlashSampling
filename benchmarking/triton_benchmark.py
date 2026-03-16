from pathlib import Path

from fused_mm_sampling.bench.triton_benchmark_lib import CliArgs, run_triton_bechmark

if __name__ == "__main__":
    args = CliArgs(tgt_dir=Path(__file__).parent / "profiles/triton-bench/")
    run_triton_bechmark(args)

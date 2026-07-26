import argparse
import gc
import json
from pathlib import Path

import torch

from fused_mm_sampling.bench.speed_test import Case
from fused_mm_sampling.bench.triton_benchmark_lib import BENCHMARK_CASES
from fused_mm_sampling.core import get_sampler


def main() -> None:
    args = parse_args()
    result = measure_peak_memory(
        provider=args.name,
        case_name=args.case,
        n_hidden_states=args.n_hidden_states,
    )
    args.memory_output.write_text(json.dumps(result, indent=2, sort_keys=True))


def measure_peak_memory(
    provider: str,
    case_name: str,
    n_hidden_states: int,
) -> dict:
    case_config = BENCHMARK_CASES[case_name]
    case = Case(
        name=provider,
        n_runs_benchmark=1,
        n_runs_warmup=1,
        n_hidden_states=n_hidden_states,
        n_samples=1,
        vocab_size=case_config["vocab_size"],
        hidden_size=case_config["hidden_size"],
    )
    kwargs = case.make_fn_kwargs()
    input_bytes = torch.cuda.memory_allocated()
    sampler = get_sampler(provider, weights=kwargs["weights"])
    sampler.prepare()

    warmup_result = sampler.sample(**kwargs)
    torch.cuda.synchronize()
    del warmup_result
    gc.collect()

    torch.cuda.reset_peak_memory_stats()
    with torch.cuda.nvtx.range("kernel"):
        result = sampler.sample(**kwargs)
    torch.cuda.synchronize()

    peak_allocated_bytes = torch.cuda.max_memory_allocated()
    peak_temporary_bytes = peak_allocated_bytes - input_bytes
    del result
    return {
        "provider": provider,
        "case": case_name,
        "batch_size": n_hidden_states,
        "input_bytes": input_bytes,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_temporary_bytes": peak_temporary_bytes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--case", default="large", choices=BENCHMARK_CASES)
    parser.add_argument("--n-hidden-states", type=int, default=64)
    parser.add_argument("--memory-output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()

import os
from pathlib import Path

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import json

import torch
import triton
from pydantic_settings import BaseSettings

from ..core import get_sampler, set_torch_allocator_for_tma_descriptors
from ..testing import shard_weights
from ..tp_info import TP1, TPInfo, run_maybe_distributed
from .sys_metadata import gather_system_metadata

# prevent torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with fullgraph=True
assert torch._dynamo.config.cache_size_limit == 8
torch._dynamo.config.cache_size_limit = 1_000

device = torch.device("cuda")
set_torch_allocator_for_tma_descriptors()

# Benchmark configurations representing real LLM sizes.
# See findings/lm-head-configurations.md for details.
BENCHMARK_CASES = {
    "qwen3-1.7b": {"vocab_size": 151_936, "hidden_size": 2_048},  # Qwen3 1.7B
    "small": {"vocab_size": 151_936, "hidden_size": 4_096},  # Qwen3 8B, Qwen3-235B MoE
    "large": {"vocab_size": 128_256, "hidden_size": 8_192},  # Llama 3 70B, DeepSeek V3
    "gpt-oss-120b": {"vocab_size": 201_088, "hidden_size": 2_880},  # GPT-OSS 120B
    "kimi-k2.5": {"vocab_size": 163_840, "hidden_size": 7_168},  # Kimi K2.5
    # Half-V cases for estimating TP2 collective overhead (TP1 at V/2 ≈ per-GPU compute on TP2)
    "large-halfv": {"vocab_size": 64_128, "hidden_size": 8_192},
    "small-halfv": {"vocab_size": 75_968, "hidden_size": 4_096},
}

N_SAMPLES = 1
TEMPERATURE = 1.0


ALL_CASES = list(BENCHMARK_CASES.keys())
DEFAULT_CASES = ["large", "small"]


class Args(BaseSettings):
    tgt_dir: Path
    name: str | None = None
    n_hidden_states: int | None = None
    case: str = "all"
    n_procs: int = 1

    def make_tp(self) -> TPInfo:
        if self.n_procs > 1:
            return TPInfo.from_world()
        return TP1

    def providers(self) -> list[str]:
        return self.name.split(",") if self.name is not None else list(provider_names)


class CliArgs(Args, cli_parse_args=True):
    pass


provider_names = {
    "fused-triton": "FMMS (Triton)",
    # "fused-cuda": "FMMS (CUDA)",
    # "fused-triton-no-gumbel": "FMMS (Triton NoNoise)",
    # "helion": "FMMS (Helion)",  # autotuning too slow atm. It runs on every bsz change
    "naive-pt": "Multinomial Sampling (Eager)",
    "naive-compiled": "Multinomial Sampling (Compiled)",
    # "sequential-compiled": "Sequential PyTorch Compiled",
    # "naive-tl-matmul": "Naive Triton Matmul",
    # "jl-compiled": "JL Compiled",
    "flashinfer:top_k_top_p_sampling_from_logits": "flashinfer:top_k_top_p_sampling_from_logits",
    "flashinfer:sampling_from_logits": "flashinfer:sampling_from_logits",
}

all_styles = [
    ("blue", "-"),
    ("green", "-"),
    ("cyan", "-"),
    ("orange", "-"),
    ("red", "-"),
    ("purple", "-"),
    ("brown", "-"),
]


def create_benchmark(args: Args, case: str):
    """Create a benchmark function for a specific case."""

    case_config = BENCHMARK_CASES[case]
    vocab_size = case_config["vocab_size"]
    hidden_size = case_config["hidden_size"]
    tp = args.make_tp()

    if args.n_hidden_states is not None:
        x_vals = [args.n_hidden_states]
    else:
        x_vals = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # nobody uses 512 or 1024

    providers = args.providers()
    lines_names = [provider_names.get(prov, prov) for prov in providers]

    config = triton.testing.Benchmark(
        x_names=["n_hidden_states"],
        x_vals=x_vals,
        x_log=True,
        line_arg="provider",
        line_vals=providers,
        line_names=lines_names,
        styles=all_styles[: len(providers)],
        ylabel="Time (ms)",
        plot_name=f"fused-mm-sample-batch-scaling-{case}",
        args={},
    )

    @triton.testing.perf_report(config)
    def benchmark(n_hidden_states, provider):
        hidden_states = torch.randn(
            (n_hidden_states, hidden_size), dtype=torch.bfloat16, device=device
        )
        weights = torch.randn((vocab_size, hidden_size), dtype=torch.bfloat16, device=device)
        weights = shard_weights(weights, tp)
        return _run_benchmark(hidden_states, weights, provider, tp)

    return benchmark


def _run_benchmark(
    hidden_states: torch.Tensor, weights: torch.Tensor, provider: str, tp: TPInfo = TP1
) -> float:
    """Common benchmark logic for all modes."""
    tp.rank0_print(f"Running benchmark for provider: {provider}")

    kwargs = dict(
        hidden_states=hidden_states,
        weights=weights,
        num_samples=N_SAMPLES,
        temperature=torch.tensor(TEMPERATURE, device=weights.device),
        tp=tp,
    )

    sampler = get_sampler(provider, weights=weights)
    sampler.prepare()

    def fn():
        return sampler.sample(**kwargs)

    quantiles = [0.5, 0.1, 0.9]  # perf_report unpacks as [center, min, max]
    if tp.size > 1:
        return _do_bench_fixed_iters(fn, quantiles=quantiles)
    return triton.testing.do_bench(fn, quantiles=quantiles)


def _do_bench_fixed_iters(
    fn,
    warmup_iters: int = 25,
    rep_iters: int = 100,
    quantiles: list[float] | None = None,
):
    """Like triton.testing.do_bench but with fixed iteration counts.

    triton.testing.do_bench calibrates iteration counts by wall-clock time,
    so different distributed ranks can run different numbers of iterations,
    causing collective mismatches (https://github.com/triton-lang/triton/issues/9683).
    This version uses fixed counts instead.

    Follows the same L2 cache flushing strategy as triton.testing.do_bench
    (https://github.com/triton-lang/triton/blob/dacfe7ad8939/python/triton/testing.py#L152-L182):
    a 256 MB buffer is zeroed before each timed iteration to evict stale L2
    cache lines, ensuring consistent cold-cache measurements.
    """
    cache = torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")

    for _ in range(warmup_iters):
        fn()
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep_iters)]
    for i in range(rep_iters):
        cache.zero_()
        start_events[i].record()
        fn()
        end_events[i].record()
    torch.cuda.synchronize()

    times = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]
    if quantiles is not None:
        import numpy as np

        return [np.quantile(times, q) for q in quantiles]
    return float(torch.tensor(times).median())


def _resolve_cases(case: str) -> list[str]:
    if case == "all":
        return DEFAULT_CASES
    if case not in BENCHMARK_CASES:
        raise ValueError(f"Unknown case: {case!r}. Choose from: {ALL_CASES + ['all']}")
    return [case]


def run_triton_bechmark(args: Args):
    run_maybe_distributed(_run_triton_benchmark_impl, args.n_procs, args)


def _run_triton_benchmark_impl(args: Args):
    tp = args.make_tp()
    cases = _resolve_cases(args.case)
    directory = args.tgt_dir
    os.makedirs(directory, exist_ok=True)

    metadata = {
        **gather_system_metadata(),
        "args": args.model_dump(mode="json"),
    }
    metadata_file = Path(directory) / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2))
    tp.rank0_print("Metadata:", json.dumps(metadata, indent=2))

    for case in cases:
        case_config = BENCHMARK_CASES[case]
        tp.rank0_print("=" * 80)
        tp.rank0_print(f"Benchmark Case: {case}")
        tp.rank0_print("Configuration:")
        tp.rank0_print(f"  vocab_size: {case_config['vocab_size']}")
        tp.rank0_print(f"  hidden_size: {case_config['hidden_size']}")
        tp.rank0_print(f"  n_samples: {N_SAMPLES}")
        tp.rank0_print(f"  temperature: {TEMPERATURE}")
        tp.rank0_print(f"  n_procs: {args.n_procs}")
        tp.rank0_print()

        benchmark = create_benchmark(args, case)
        benchmark.run(print_data=tp.is_rank0(), save_path=directory if tp.is_rank0() else None)
        tp.rank0_print()

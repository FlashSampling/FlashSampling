import os
from pathlib import Path
from typing import Callable, Literal

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import json

import numpy as np
import torch
import torch.distributed as dist
import triton
from flashinfer.testing import bench_gpu_time
from pydantic import model_validator
from pydantic_settings import BaseSettings

from ..alg_names import ShortNames as S
from ..alg_names import short2long
from ..core import get_sampler
from ..testing import shard_weights
from ..tp_info import TP1, TPInfo, run_maybe_distributed
from .sys_metadata import gather_system_metadata

# prevent torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with fullgraph=True
assert torch._dynamo.config.cache_size_limit == 8
torch._dynamo.config.cache_size_limit = 1_000

device = torch.device("cuda")

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


DEFAULT_PROVIDERS = [
    S.fused_triton,
    S.naive_pt,
    S.naive_compiled,
    S.flashinfer_top_k_top_p_sampling_from_logits,
    S.flashinfer_sampling_from_logits,
]


class Args(BaseSettings):
    tgt_dir: Path | None = None
    name: str | None = None
    n_hidden_states: int | None = None
    case: str = "all"
    n_procs: int = 1
    disable_compile: bool = False
    n_runs_warmup: int = 25
    n_runs_benchmark: int = 100
    n_samples: int = 1
    bench_fn: Literal["own", "nvbench", "fi-cupti"] = "fi-cupti"
    nsys_profile: bool = False
    top_k: int | None = None
    top_p: float | None = None

    @model_validator(mode="after")
    def _validate_distributed_bench_fn(self) -> "Args":
        if self.n_procs > 1 and self.bench_fn == "nvbench":
            raise ValueError(
                "Distributed benchmarking is not supported with --bench_fn=nvbench. "
                "nvbench controls iteration counts internally, which causes collective op "
                "deadlocks when ranks run different numbers of iterations. "
                "Use --bench_fn=own instead."
            )
        return self

    def make_tp(self) -> TPInfo:
        if self.n_procs > 1:
            return TPInfo.from_world()
        return TP1

    def providers(self) -> list[str]:
        if self.name is None or self.name == "default":
            return DEFAULT_PROVIDERS
        return self.name.split(",")


class CliArgs(Args, cli_parse_args=True):
    pass


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
    lines_names = [short2long.get(prov, prov) for prov in providers]

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
        return _run_benchmark(hidden_states, weights, provider, tp, bench_fn=args.bench_fn)

    return benchmark


def _run_benchmark(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    provider: str,
    tp: TPInfo = TP1,
    bench_fn: Literal["own", "fi-cupti"] = "fi-cupti",
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

    is_distributed = tp.size > 1
    match bench_fn:
        case "fi-cupti":
            times_ms = bench_cupti(fn, is_distributed=is_distributed)
        case "own":
            times_ms = bench_cuda_events(fn, is_distributed=is_distributed)

    quantiles = [0.5, 0.1, 0.9]  # perf_report unpacks as [center, min, max]
    return [np.quantile(times_ms, q) for q in quantiles]


def bench_cupti(
    fn: Callable,
    warmup_iters: int = 25,
    rep_iters: int = 100,
    is_distributed: bool = False,
) -> list[float]:
    """Time a callable using FlashInfer's CUPTI-based bench_gpu_time.

    Returns per-iteration times in milliseconds.
    """
    bench_kwargs: dict = dict(fn=fn, cold_l2_cache=True, enable_cupti=True)
    if is_distributed:
        bench_kwargs["dry_run_iters"] = warmup_iters
        bench_kwargs["repeat_iters"] = rep_iters
    return bench_gpu_time(**bench_kwargs)


def bench_cuda_events(
    fn: Callable,
    warmup_iters: int = 25,
    rep_iters: int = 100,
    is_distributed: bool = False,
) -> list[float]:
    """Time a callable using CUDA events with L2 cache flushing.

    Returns per-iteration times in milliseconds.

    Uses fixed iteration counts (not adaptive wall-clock calibration) to
    avoid collective mismatches in distributed runs where different ranks
    would otherwise run different numbers of iterations
    (https://github.com/triton-lang/triton/issues/9683).
    """
    cache = create_l2_cache()
    for _ in range(warmup_iters):
        fn()
    synchronize(is_distributed)

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(rep_iters)]
    for i in range(rep_iters):
        clear_l2_cache(cache)
        start_events[i].record()
        fn()
        end_events[i].record()
    synchronize(is_distributed)

    return [s.elapsed_time(e) for s, e in zip(start_events, end_events)]


def create_l2_cache() -> torch.Tensor:
    """Allocate a 256 MB buffer for L2 cache flushing.

    Follows the same strategy as triton.testing.do_bench: the buffer is zeroed
    before each timed iteration to evict stale L2 cache lines, ensuring
    consistent cold-cache measurements.
    """
    return torch.empty(256 * 1024 * 1024 // 4, dtype=torch.int, device="cuda")


@torch.cuda.nvtx.range("clear-l2-cache")
def clear_l2_cache(cache: torch.Tensor) -> None:
    cache.zero_()


def synchronize(is_distributed: bool) -> None:
    if is_distributed:
        dist.barrier()
    else:
        torch.cuda.synchronize()


def _resolve_cases(case: str) -> list[str]:
    if case == "all":
        return DEFAULT_CASES
    if case not in BENCHMARK_CASES:
        raise ValueError(f"Unknown case: {case!r}. Choose from: {ALL_CASES + ['all']}")
    return [case]


def run_triton_bechmark(args: Args):
    run_maybe_distributed(_run_triton_benchmark_impl, args.n_procs, args)


def _run_triton_benchmark_impl(args: Args):
    if args.disable_compile:
        torch._dynamo.config.disable = True
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

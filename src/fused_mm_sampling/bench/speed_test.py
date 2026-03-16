import os
import timeit
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cuda.bench as nvbench
import pandas as pd
import torch
import triton
import triton.profiler as proton
from pydantic import model_validator
from pydantic_settings import BaseSettings

from ..core import (
    fused_mm_sample_triton_kernel,
    get_sampler,
    sample,
    set_torch_allocator_for_tma_descriptors,
)
from ..testing import shard_weights
from ..tp_info import TP1, TPInfo, run_maybe_distributed
from .sys_metadata import get_gpu_name
from .triton_benchmark_lib import BENCHMARK_CASES, provider_names

device = torch.device("cuda")
set_torch_allocator_for_tma_descriptors()


class Args(BaseSettings):
    name: str | None = None
    n_runs_warmup: int = 25
    n_runs_benchmark: int = 100

    n_hidden_states: int = 1
    n_samples: int = 1
    tgt_dir: Path | None = None
    use_proton: bool = False
    # "pcsampling": per-line CUPTI PC sampling → kernel.hatchet
    #   On CUDA 13+ drivers, set TRITON_CUPTI_LIB_PATH to the system CUPTI dir.
    # "trace": chrome timeline (CUPTI) → kernel.chrome_trace (open in chrome://tracing)
    proton_mode: Literal["pcsampling", "trace"] = "pcsampling"
    case: str = "small"
    bench_fn: Literal["own", "nvbench", "fi-cupti"] = "fi-cupti"
    top_k: int | None = None
    top_p: float | None = None
    n_procs: int = 1

    @model_validator(mode="after")
    def _validate_distributed_bench_fn(self) -> "Args":
        if self.n_procs > 1 and self.bench_fn == "nvbench":
            raise ValueError(
                "Distributed benchmarking is not supported with --bench_fn=nvbench. "
                "nvbench controls iteration counts internally, which causes collective op "
                "deadlocks when ranks run different numbers of iterations. "
                "Use --bench_fn=own or --bench_fn=fi-cupti instead."
            )
        return self

    def make_tp(self) -> TPInfo:
        if self.n_procs > 1:
            return TPInfo.from_world()
        return TP1

    def as_case(self, name: str) -> "Case":
        assert self.n_runs_warmup is not None
        assert self.n_runs_benchmark is not None
        if self.case not in BENCHMARK_CASES:
            raise ValueError(
                f"Unknown case: {self.case!r}. Choose from: {list(BENCHMARK_CASES.keys())}"
            )
        case_config = BENCHMARK_CASES[self.case]
        return Case(
            name=name,
            n_runs_benchmark=self.n_runs_benchmark,
            n_runs_warmup=self.n_runs_warmup,
            n_hidden_states=self.n_hidden_states,
            n_samples=self.n_samples,
            use_proton=self.use_proton,
            proton_mode=self.proton_mode,
            vocab_size=case_config["vocab_size"],
            hidden_size=case_config["hidden_size"],
            top_k=self.top_k,
            top_p=self.top_p,
            tp=self.make_tp(),
        )

    def providers(self) -> list[str]:
        return self.name.split(",") if self.name is not None else list(provider_names)

    def all_cases(self) -> list["Case"]:
        return [self.as_case(name=provider) for provider in self.providers()]


class CliArgs(Args, cli_parse_args=True):
    pass


sample_compiled = torch.compile(sample)


@dataclass
class Case:
    name: str
    n_runs_benchmark: int
    n_runs_warmup: int
    n_hidden_states: int
    n_samples: int
    use_proton: bool
    proton_mode: str
    vocab_size: int
    hidden_size: int
    top_k: int | None = None
    top_p: float | None = None
    tp: TPInfo = TP1

    def make_fn_kwargs(self) -> dict:
        """This function can be slow because it allocates tensors."""
        weights = torch.randn(
            (self.vocab_size, self.hidden_size), dtype=torch.bfloat16, device=device
        )
        weights = shard_weights(weights, self.tp)
        kwargs = dict(
            hidden_states=torch.randn(
                (self.n_hidden_states, self.hidden_size), dtype=torch.bfloat16, device=device
            ),
            weights=weights,
            num_samples=self.n_samples,
            temperature=torch.tensor(1.0, device=device),
            tp=self.tp,
        )
        if self.top_k is not None:
            kwargs["top_k"] = self.top_k
        if self.top_p is not None:
            kwargs["top_p"] = self.top_p
        return kwargs


def setup_proton(mode: Literal["pcsampling", "trace"]) -> None:
    import triton.profiler.language as pl

    print(f"⚙️ Proton profiling enabled (mode={mode})")
    if mode == "pcsampling":
        proton.start(name="kernel", hook="triton", backend="cupti", mode="pcsampling")
    elif mode == "trace":
        os.environ["USE_PROTON_SCOPES"] = "1"
        # Enable pl.enter_scope()/pl.exit_scope() annotations in Triton kernels.
        # hook="triton" instruments the JIT so pl scopes are recorded.
        # TRITON_ALWAYS_COMPILE=1 forces recompilation with the hooks injected.
        pl.enable_semantic("triton")
        proton.start(name="kernel", data="trace", hook="triton", backend="instrumentation")
    else:
        raise ValueError(f"Unknown proton_mode: {mode!r}")

    def enter_autotune(args, reset_only=False):
        if reset_only:
            return
        proton.enter_scope("<autotune>")

    def exit_autotune(args, exception):
        proton.exit_scope()

    fused_mm_sample_triton_kernel.pre_hook = enter_autotune
    fused_mm_sample_triton_kernel.post_hook = exit_autotune


@proton.scope("clear-l2-cache")
def clear_l2_cache(cache):
    with torch.cuda.nvtx.range("clear-l2-cache"):
        triton.runtime.driver.active.clear_cache(cache)


def benchmark(case: Case) -> pd.DataFrame:
    """Inspired by triton.testing.do_bench"""
    case.tp.rank0_print("=" * 80)
    case.tp.rank0_print(f"Benchmarking {case.name}...")
    kwargs = case.make_fn_kwargs()
    sampler = get_sampler(case.name, weights=kwargs["weights"])
    sampler.prepare()

    def fn():
        return sampler.sample(**kwargs)

    di = triton.runtime.driver.active.get_device_interface()

    if case.use_proton and case.tp.is_rank0():
        setup_proton(case.proton_mode)

    # 2026-03-01 Tomas: Not sure we need this separately from the warmup.
    # with proton.scope("first-run"):
    #     # Compile, etc.
    #     fn()
    #     di.synchronize()

    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    start_events = [di.Event(enable_timing=True) for _ in range(case.n_runs_benchmark)]
    end_events = [di.Event(enable_timing=True) for _ in range(case.n_runs_benchmark)]

    if case.n_runs_warmup > 0:
        case.tp.rank0_print("Warming up...")
        with proton.scope("warmup"):
            for _ in range(case.n_runs_warmup):
                clear_l2_cache(cache)
                fn()

    case.tp.rank0_print("Timing...")
    with proton.scope("timing"):
        for _, start_event, end_event in zip(
            range(case.n_runs_benchmark), start_events, end_events
        ):
            clear_l2_cache(cache)
            with torch.cuda.nvtx.range("kernel"):
                start_event.record()
                timeit.timeit(fn, number=1)
                end_event.record()
        di.synchronize()

    if case.use_proton and case.tp.is_rank0():
        proton.finalize()

    times_ms = [s.elapsed_time(e) for s, e in zip(start_events, end_events)]

    results = {
        "name": case.name,
        "total[s]": sum(times_ms) / 1_000,
        "time[s]": [t / 1_000 for t in times_ms],
        "time[ms]": times_ms,
        "time[µs]": [t * 1_000 for t in times_ms],
    }
    df = pd.DataFrame(results)
    return df


def benchmark_all(cases: list[Case]) -> pd.DataFrame:
    dfs = [benchmark(case) for case in cases]
    return pd.concat(dfs, ignore_index=True)


def run_nvbench(args: Args) -> None:
    """Run benchmarks using NVBench."""

    def nvbench_kernel(state: "nvbench.State"):
        provider = state.get_string("Provider")
        case = args.as_case(name=provider)
        kwargs = case.make_fn_kwargs()
        sampler = get_sampler(provider, weights=kwargs["weights"])
        sampler.prepare()

        # Warmup (compile, autotune, etc.)
        sampler.sample(**kwargs)
        torch.cuda.synchronize()

        def launcher(launch: "nvbench.Launch"):
            stream = _as_torch_stream(launch.get_stream())
            with torch.cuda.stream(stream):
                sampler.sample(**kwargs)

        state.exec(launcher, batched=False)

    csv_args = []
    if args.tgt_dir is not None:
        args.tgt_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.tgt_dir / "nvbench.csv"
        csv_args = ["--csv", str(csv_path)]

    b = nvbench.register(nvbench_kernel)
    b.add_string_axis("Provider", args.providers())
    b.add_string_axis("Case", [args.case])
    nvbench.run_all_benchmarks(["speed_test"] + csv_args)

    if args.tgt_dir is not None:
        df = pd.read_csv(csv_path)
        df = assign_col_time_ms(df).sort_values("GPU Time (sec)")
        df.to_csv(csv_path, index=False)
        print("Saved results to", csv_path)


def assign_col_time_ms(df: pd.DataFrame) -> pd.DataFrame:
    df["GPU Time (ms)"] = (df["GPU Time (sec)"] * 1e3).round(3)
    df["CPU Time (ms)"] = (df["CPU Time (sec)"] * 1e3).round(3)
    return df


def _as_torch_stream(cs: "nvbench.CudaStream") -> torch.cuda.ExternalStream:
    return torch.cuda.ExternalStream(cs.addressof())


def run_cupti(args: Args) -> None:
    """Run benchmarks using FlashInfer's CUPTI-based bench_gpu_time."""
    from flashinfer.testing import bench_gpu_time

    tp = args.make_tp()
    rows = []
    for provider in args.providers():
        case = args.as_case(name=provider)
        kwargs = case.make_fn_kwargs()
        sampler = get_sampler(provider, weights=kwargs["weights"])
        sampler.prepare()

        # Warmup (compile, autotune, etc.)
        sampler.sample(**kwargs)
        torch.cuda.synchronize()

        tp.rank0_print(f"Benchmarking {provider}...")
        bench_kwargs = dict(
            fn=lambda s=sampler: s.sample(**kwargs),
            cold_l2_cache=True,
            enable_cupti=True,
        )
        if args.n_procs > 1:
            tp.rank0_print("Reading bench iteration counts from Args")
            bench_kwargs["dry_run_iters"] = args.n_runs_warmup
            bench_kwargs["repeat_iters"] = args.n_runs_benchmark
        else:
            tp.rank0_print("Using adaptive bench iteration counts")

        times_ms = bench_gpu_time(**bench_kwargs)
        times_md = pd.Series(times_ms)
        rows.append(
            {
                "Provider": provider,
                "median_ms": times_md.median(),
                "min_ms": times_md.min(),
                "max_ms": times_md.max(),
                "iters": len(times_ms),
            }
        )

    if tp.is_rank0():
        _print_and_dump_cupti_results(rows, args)


def _print_and_dump_cupti_results(rows: list[dict], args: Args) -> None:
    df = pd.DataFrame(rows).sort_values("median_ms")
    print()
    print(df.round(3))

    if args.tgt_dir is not None:
        args.tgt_dir.mkdir(parents=True, exist_ok=True)
        out = args.tgt_dir / "fi-cupti.csv"
        df.round(3).to_csv(out, index=False)
        print("Saved results to", out)


def run_own_benchmark(args: Args) -> None:
    tp = args.make_tp()
    cases: list[Case] = args.all_cases()
    df = benchmark_all(cases)
    if tp.is_rank0():
        _print_and_dump_own_results(df, args)


def _print_and_dump_own_results(df: pd.DataFrame, args: Args) -> None:
    print(f"{args.n_samples=}")

    total_runtimes = df.groupby(["name", "total[s]"], as_index=False).size()
    print(total_runtimes.sort_values("total[s]").round(2))

    time_distribution = df.groupby("name")["time[ms]"].describe().sort_values("50%")
    print(time_distribution.round(2))

    if args.tgt_dir is not None:
        args.tgt_dir.mkdir(parents=True, exist_ok=True)
        total_runtimes.to_csv(args.tgt_dir / "total-runtimes.csv")
        time_distribution.to_csv(args.tgt_dir / "time-distribution.csv")
        print("Saved results to ", args.tgt_dir)


def run_speed_test(args: Args) -> None:
    """Run a speed test for a given set of arguments."""
    run_maybe_distributed(_run_speed_test_impl, args.n_procs, args)


def _run_speed_test_impl(args: Args) -> None:
    tp = args.make_tp()
    case_config = BENCHMARK_CASES[args.case]
    tp.rank0_print("GPU:", get_gpu_name())
    tp.rank0_print("Arguments:", args.model_dump_json())
    tp.rank0_print(f"Benchmark case: {args.case}")
    tp.rank0_print(f"  vocab_size: {case_config['vocab_size']}")
    tp.rank0_print(f"  hidden_size: {case_config['hidden_size']}")
    tp.rank0_print(f"  n_hidden_states: {args.n_hidden_states}")
    tp.rank0_print(f"  n_samples: {args.n_samples}")
    tp.rank0_print(f"  n_procs: {args.n_procs}")
    tp.rank0_print()

    match args.bench_fn:
        case "nvbench":
            return run_nvbench(args)
        case "fi-cupti":
            return run_cupti(args)
        case "own":
            return run_own_benchmark(args)
        case _:
            raise ValueError("Unknown bench_fn: {args.bench_fn!r}")

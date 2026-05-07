from dataclasses import dataclass

import cuda.bench as nvbench
import pandas as pd
import torch

from ..core import (
    get_sampler,
    sample,
)
from ..testing import shard_weights
from ..tp_info import TP1, TPInfo, run_maybe_distributed
from .sys_metadata import get_gpu_name
from .triton_benchmark_lib import (
    BENCHMARK_CASES,
    Args,
    bench_cuda_events,
    bench_cupti,
    clear_l2_cache,
    create_l2_cache,
    synchronize,
)

device = torch.device("cuda")


class CliArgs(Args, cli_parse_args=True):
    case: str = "small"
    n_hidden_states: int = 1


sample_compiled = torch.compile(sample)


def as_case(args: Args, name: str) -> "Case":
    assert args.n_hidden_states is not None
    assert args.n_runs_warmup is not None
    assert args.n_runs_benchmark is not None
    if args.case not in BENCHMARK_CASES:
        raise ValueError(
            f"Unknown case: {args.case!r}. Choose from: {list(BENCHMARK_CASES.keys())}"
        )
    case_config = BENCHMARK_CASES[args.case]
    return Case(
        name=name,
        n_runs_benchmark=args.n_runs_benchmark,
        n_runs_warmup=args.n_runs_warmup,
        n_hidden_states=args.n_hidden_states,
        n_samples=args.n_samples,
        vocab_size=case_config["vocab_size"],
        hidden_size=case_config["hidden_size"],
        top_k=args.top_k,
        top_p=args.top_p,
        tp=args.make_tp(),
    )


def all_cases(args: Args) -> list["Case"]:
    return [as_case(args, name=provider) for provider in args.providers()]


@dataclass
class Case:
    name: str
    n_runs_benchmark: int
    n_runs_warmup: int
    n_hidden_states: int
    n_samples: int
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


def _prepare_case(case: Case):
    """Set up sampler and fn for a benchmark case."""
    case.tp.rank0_print("=" * 80)
    case.tp.rank0_print(f"Benchmarking {case.name}...")
    kwargs = case.make_fn_kwargs()
    sampler = get_sampler(case.name, weights=kwargs["weights"])
    sampler.prepare()

    def fn():
        return sampler.sample(**kwargs)

    return fn


def benchmark(case: Case) -> pd.DataFrame:
    """Time kernel execution using CUDA events."""
    fn = _prepare_case(case)
    is_distributed = case.tp.size > 1
    times_ms = bench_cuda_events(
        fn,
        warmup_iters=case.n_runs_warmup,
        rep_iters=case.n_runs_benchmark,
        is_distributed=is_distributed,
    )
    results = {
        "name": case.name,
        "total[s]": sum(times_ms) / 1_000,
        "time[s]": [t / 1_000 for t in times_ms],
        "time[ms]": times_ms,
        "time[µs]": [t * 1_000 for t in times_ms],
    }
    return pd.DataFrame(results)


def nsys_profile(case: Case) -> None:
    """Run under nsys: warmup, sync ranks, then profile timed iterations."""
    fn = _prepare_case(case)
    cache = create_l2_cache()
    is_distributed = case.tp.size > 1
    for _ in range(case.n_runs_warmup):
        clear_l2_cache(cache)
        fn()
    synchronize(is_distributed)

    torch.cuda.cudart().cudaProfilerStart()
    synchronize(is_distributed)

    for _ in range(case.n_runs_benchmark):
        clear_l2_cache(cache)
        with torch.cuda.nvtx.range("kernel"):
            fn()

    synchronize(is_distributed)
    torch.cuda.cudart().cudaProfilerStop()


def benchmark_all(cases: list[Case]) -> pd.DataFrame:
    dfs = [benchmark(case) for case in cases]
    return pd.concat(dfs, ignore_index=True)


def run_nvbench(args: Args) -> None:
    """Run benchmarks using NVBench."""

    def nvbench_kernel(state: "nvbench.State"):
        provider = state.get_string("Provider")
        case = as_case(args, provider)
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
    tp = args.make_tp()
    is_distributed = tp.size > 1
    rows = []
    for provider in args.providers():
        case = as_case(args, provider)
        fn = _prepare_case(case)

        tp.rank0_print(f"Benchmarking {provider}...")
        times_ms = bench_cupti(
            fn,
            warmup_iters=args.n_runs_warmup,
            rep_iters=args.n_runs_benchmark,
            is_distributed=is_distributed,
        )
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
    cases: list[Case] = all_cases(args)
    if args.nsys_profile:
        for case in cases:
            nsys_profile(case)
    else:
        df = benchmark_all(cases)
        if tp.is_rank0():
            _print_and_dump_own_results(df, args)


def _print_and_dump_own_results(df: pd.DataFrame, args: Args) -> None:
    print(f"{args.n_samples=}")

    total_runtimes = df.groupby(["name", "total[s]"], as_index=False).size()
    print(total_runtimes.sort_values("total[s]").round(2).to_markdown())
    print()

    time_distribution = df.groupby("name")["time[ms]"].describe().sort_values("50%")
    print(time_distribution.round(2).to_markdown())

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

import timeit
from dataclasses import dataclass
from pathlib import Path

import cuda.bench as nvbench
import pandas as pd
import torch
import triton
import triton.profiler as proton
from pydantic_settings import BaseSettings

from ..core import fused_mm_sample_triton_kernel, get_gpu_name, get_sampler, sample
from .triton_benchmark import BENCHMARK_CASES

device = torch.device("cuda")


class Args(BaseSettings):
    name: str | None = None
    n_runs_warmup: int = 25
    n_runs_benchmark: int = 100

    n_hidden_states: int = 1
    n_samples: int = 1
    tgt_dir: Path | None = None
    use_proton: bool = False
    case: str = "large"
    nvbench: bool = False

    def as_case(self, name: str | None = None) -> "Case":
        if name is None:
            name = self.name
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
            vocab_size=case_config["vocab_size"],
            hidden_size=case_config["hidden_size"],
        )

    def all_cases(self) -> list["Case"]:
        return [self.as_case(name=provider) for provider in all_providers]


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
    vocab_size: int
    hidden_size: int

    def make_fn_kwargs(self) -> dict:
        """This function can be slow because it allocates tensors."""
        return dict(
            hidden_states=torch.randn(
                (self.n_hidden_states, self.hidden_size), dtype=torch.bfloat16, device=device
            ),
            weights=torch.randn(
                (self.vocab_size, self.hidden_size), dtype=torch.bfloat16, device=device
            ),
            num_samples=self.n_samples,
            temperature=torch.tensor(1.0, device=device),
        )


all_providers = [
    "fused-triton",
    # "fused-cuda",
    # "helion",
    "naive-compiled",
    # "sequential-compiled",
    # "naive-tl-matmul",
    # "jl-compiled",
    "flashinfer:top_k_top_p_sampling_from_logits",
    "flashinfer:sampling_from_logits",
]


def setup_proton() -> None:
    # Start proton BEFORE kernel compilation so hook="triton" can instrument the JIT
    print("⚙️ Proton profiling enabled")
    proton.start(name="kernel", hook="triton", backend="cupti", mode="pcsampling")

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
    print("=" * 80)
    print(f"Benchmarking {case.name}...")
    kwargs = case.make_fn_kwargs()
    sampler = get_sampler(case.name, weights=kwargs["weights"])
    sampler.prepare()

    def fn():
        return sampler.sample(**kwargs)

    di = triton.runtime.driver.active.get_device_interface()

    if case.use_proton:
        setup_proton()

    with proton.scope("first-run"):
        # Compile, etc.
        fn()
        di.synchronize()

    cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

    start_events = [di.Event(enable_timing=True) for _ in range(case.n_runs_benchmark)]
    end_events = [di.Event(enable_timing=True) for _ in range(case.n_runs_benchmark)]

    print("Warming up...")
    with proton.scope("warmup"):
        for _ in range(case.n_runs_warmup):
            clear_l2_cache(cache)
            fn()

    print("Timing...")
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

    if case.use_proton:
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

    providers = [args.name] if args.name is not None else list(all_providers)

    csv_args = []
    if args.tgt_dir is not None:
        args.tgt_dir.mkdir(parents=True, exist_ok=True)
        csv_path = args.tgt_dir / "nvbench.csv"
        csv_args = ["--csv", str(csv_path)]

    b = nvbench.register(nvbench_kernel)
    b.add_string_axis("Provider", providers)
    b.add_string_axis("Case", [args.case])
    nvbench.run_all_benchmarks(["speed_test"] + csv_args)

    if args.tgt_dir is not None:
        df = pd.read_csv(csv_path)
        df = assign_col_time_ms(df)
        df.to_csv(csv_path, index=False)
        print("Saved results to", csv_path)


def assign_col_time_ms(df: pd.DataFrame) -> pd.DataFrame:
    df["GPU Time (ms)"] = (df["GPU Time (sec)"] * 1e3).round(3)
    return df


def _as_torch_stream(cs: "nvbench.CudaStream") -> torch.cuda.ExternalStream:
    return torch.cuda.ExternalStream(cs.addressof())


def run_own_benchmark(args: Args) -> None:
    if args.name is not None:
        cases = [args.as_case()]
    else:
        cases = args.all_cases()
    df = benchmark_all(cases)
    print(f"{args.n_samples=}")

    total_runtimes = df.groupby(["name", "total[s]"], as_index=False).size()
    print(total_runtimes.sort_values("total[s]").round(2))

    time_distribution = df.groupby("name")["time[ms]"].describe()
    print(time_distribution.sort_values("min").round(2))

    if args.tgt_dir is not None:
        args.tgt_dir.mkdir(parents=True, exist_ok=True)
        total_runtimes.to_csv(args.tgt_dir / "total-runtimes.csv")
        time_distribution.to_csv(args.tgt_dir / "time-distribution.csv")
        print("Saved results to ", args.tgt_dir)


def run_speed_test(args: Args) -> None:
    """Run a speed test for a given set of arguments."""
    print("GPU:", get_gpu_name())
    print("Arguments:", args.model_dump_json())
    case_config = BENCHMARK_CASES[args.case]
    print(f"Benchmark case: {args.case}")
    print(f"  vocab_size: {case_config['vocab_size']}")
    print(f"  hidden_size: {case_config['hidden_size']}")
    print(f"  n_hidden_states: {args.n_hidden_states}")
    print(f"  n_samples: {args.n_samples}")
    print()

    if args.nvbench:
        return run_nvbench(args)
    else:
        return run_own_benchmark(args)

import pandas as pd
from pydantic_settings import BaseSettings

from fused_mm_sampling.bench.triton_benchmark_lib import BENCHMARK_CASES


def our_method(vocab_size: int, hidden_size: int, n_hidden_states: int):
    m = vocab_size
    n = n_hidden_states
    k = hidden_size
    # theory: fused matmul + reduction
    # we write only bsz results back to HBM
    reads = m * k + n * k
    writes = n
    return reads, writes


def naive_method(vocab_size: int, hidden_size: int, n_hidden_states: int):
    n = n_hidden_states
    m = vocab_size
    k = hidden_size
    # first stage: matmul
    reads1 = m * k + n * k
    writes1 = m * n
    # second stage: sampling
    reads2 = writes1
    writes2 = n
    return reads1 + reads2, writes1 + writes2


class Args(BaseSettings):
    config: str = "small"


class CliArgs(Args, cli_parse_args=True):
    pass


args = CliArgs()
case = BENCHMARK_CASES[args.config]
vocab_size = case["vocab_size"]
hidden_size = case["hidden_size"]

print(f"Config: {args.config} (V={vocab_size}, d={hidden_size})")

rows = []
for n_hidden_states in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]:
    our_r, our_w = our_method(vocab_size, hidden_size, n_hidden_states)
    naive_r, naive_w = naive_method(vocab_size, hidden_size, n_hidden_states)
    rows.append(
        {
            "n_hidden_states": n_hidden_states,
            "our_reads": our_r,
            "our_writes": our_w,
            "naive_reads": naive_r,
            "naive_writes": naive_w,
            "ratio": (our_r + our_w) / (naive_r + naive_w),
        }
    )

df = pd.DataFrame(rows)
print(df.round(2))

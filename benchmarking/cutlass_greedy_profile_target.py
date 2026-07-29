"""Single-range target used by Nsight Compute for CUTLASS greedy profiling."""

import argparse

import torch

from fused_mm_sampling.cutlass_impl import (
    cutlass_launch_greedy_gemm,
    cutlass_launch_greedy_stage2,
    cutlass_make_greedy_buffers,
    cutlass_plain_gemm,
)


def main() -> None:
    args = parse_args()
    weights = torch.randn(
        (args.vocab_size, args.hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    hidden_states = torch.randn(
        (args.n_hidden_states, args.hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    padded, candidates, output, gemm_n, rounded_n, m_tiles = (
        cutlass_make_greedy_buffers(weights, hidden_states)
    )

    def fused_gemm() -> None:
        cutlass_launch_greedy_gemm(
            weights, padded, candidates, gemm_n, rounded_n
        )

    def stage2() -> None:
        cutlass_launch_greedy_stage2(
            candidates,
            output,
            m_tiles,
            rounded_n,
            args.n_hidden_states,
        )

    functions = {
        "fused-gemm": fused_gemm,
        "stage2": stage2,
        "plain-gemm": lambda: cutlass_plain_gemm(weights, hidden_states),
    }
    function = functions[args.component]
    fused_gemm()
    stage2()
    function()
    torch.cuda.synchronize()
    with torch.cuda.nvtx.range("profile"):
        function()
    torch.cuda.synchronize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        required=True,
        choices=("fused-gemm", "stage2", "plain-gemm"),
    )
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--n-hidden-states", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()

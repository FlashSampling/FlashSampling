"""Single-range target used by Nsight Compute for CUTLASS greedy profiling."""

import argparse

import torch

from fused_mm_sampling.cutlass_impl import (
    cutlass_winning_plain_gemm,
    fused_mm_sample_cutlass_greedy,
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
    temperature = torch.empty((), device="cuda")

    def production_fused() -> None:
        fused_mm_sample_cutlass_greedy(
            weights,
            hidden_states,
            num_samples=1,
            temperature=temperature,
        )

    def matching_plain() -> None:
        cutlass_winning_plain_gemm(weights, hidden_states)

    functions = {
        "production-fused": production_fused,
        "matching-plain": matching_plain,
    }
    function = functions[args.component]
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
        choices=("production-fused", "matching-plain"),
    )
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--n-hidden-states", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()

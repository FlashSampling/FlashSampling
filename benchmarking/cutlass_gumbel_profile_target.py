import argparse

import torch

from fused_mm_sampling.core import get_sampler
from fused_mm_sampling.modal_lib.utils import set_volume_caches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("greedy", "gumbel"), required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--n-hidden-states", type=int, required=True)
    args = parser.parse_args()

    set_volume_caches()
    torch.manual_seed(0)
    weights = torch.randn(
        (args.vocab_size, args.hidden_size), dtype=torch.bfloat16, device="cuda"
    )
    hidden_states = torch.randn(
        (args.n_hidden_states, args.hidden_size),
        dtype=torch.bfloat16,
        device="cuda",
    )
    temperature = torch.tensor(1.0, device="cuda")
    provider = "fused-cutlass" if args.component == "gumbel" else "fused-cutlass-greedy"
    sampler = get_sampler(provider, weights=weights)
    kwargs = {
        "weights": weights,
        "hidden_states": hidden_states,
        "num_samples": 1,
        "temperature": temperature,
    }
    if args.component == "gumbel":
        kwargs["seed"] = 17
    sampler.sample(**kwargs)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("profile")
    sampler.sample(**kwargs)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

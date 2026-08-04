import argparse

import torch
from fused_mm_sampling.core import get_sampler
from fused_mm_sampling.cutlass_experiments import CUTLASS_SAMPLING_EXPERIMENTS
from fused_mm_sampling.cutlass_impl import fused_mm_sample_cutlass_experimental
from fused_mm_sampling.modal_lib.utils import set_volume_caches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        choices=("greedy", "gumbel", "triton", *CUTLASS_SAMPLING_EXPERIMENTS),
        required=True,
    )
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
    kwargs = {
        "weights": weights,
        "hidden_states": hidden_states,
        "num_samples": 1,
        "temperature": temperature,
    }
    if args.component != "greedy":
        kwargs["seed"] = 17
    if args.component in CUTLASS_SAMPLING_EXPERIMENTS:

        def sample():
            return fused_mm_sample_cutlass_experimental(
                **kwargs, variant=args.component
            )

    else:
        providers = {
            "greedy": "fused-cutlass-greedy",
            "gumbel": "fused-cutlass",
            "triton": "fused-triton",
        }
        provider = providers[args.component]
        sampler = get_sampler(provider, weights=weights)

        def sample():
            return sampler.sample(**kwargs)

    sample()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("profile")
    sample()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

"""Tensor-parallel sampling across 2 GPUs.

Each GPU holds a shard of the vocabulary weights (V/2 rows each).
The fused kernel computes local Gumbel-max values, then an all-gather
across ranks selects the global winner, so every rank gets the same
sampled token indices.

Usage (requires 2 GPUs):
    torchrun --nproc_per_node=2 examples/tensor_parallel.py
"""

import os

import torch
import torch.distributed as dist

from fused_mm_sampling import TPInfo, fused_mm_sample_triton
from fused_mm_sampling.testing import shard_weights


def main():
    # Setup distributed environment
    dist.init_process_group(backend="nccl")  # use backend=gloo for single GPU usage
    rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda")
    tp = TPInfo(rank=rank, size=world_size)

    # Create inputs
    vocab_size = 128_256  # divisible by 2 for even sharding
    hidden_size = 4096
    n_hidden_states = 4  # (batch size)
    hidden_states = torch.randn(n_hidden_states, hidden_size, dtype=torch.bfloat16, device=device)

    # Shard weights across ranks
    torch.manual_seed(42)
    full_weights = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16, device=device)
    weights = shard_weights(full_weights, tp)
    if rank == 0:
        print(f"  weight shard per rank: {list(weights.shape)}")

    samples = fused_mm_sample_triton(
        weights=weights,
        hidden_states=hidden_states,
        num_samples=1,
        temperature=torch.tensor(0.8, device=device),
        seed=rank * 1_000_000,
        tp=tp,
    )

    if rank == 0:
        print("Sample shape: ", samples.shape)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

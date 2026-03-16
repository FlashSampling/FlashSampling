import os

from fused_mm_sampling import fused_mm_sample_triton

os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

import torch

device = torch.device("cuda")

# Input dimensions (typical for large language models)
vocab_size = 256_000
hidden_size = 5120  # dimension of the hidden states
# number of vectors in the hidden states
# during batched sampling, this is the number of sequences in the batch
n_hidden_states = 16


print("Running example with:")
print(f"  vocab_size = {vocab_size:,}")
print(f"  hidden_size = {hidden_size:,}")
print(f"  n_hidden_states = {n_hidden_states:,}")

# Create random inputs. The weights are transposed for the matmul.
weights = torch.randn(vocab_size, hidden_size, dtype=torch.bfloat16, device=device)
hidden_states = torch.randn(n_hidden_states, hidden_size, dtype=torch.bfloat16, device=device)

# Sample from categorical distribution using fused Triton kernel
samples = fused_mm_sample_triton(
    weights=weights,
    hidden_states=hidden_states,
    num_samples=16,
    temperature=torch.tensor(0.8, device=device),
    seed=0,
)

print(f"\nOutput shape: {samples.shape}")
print(f"Sample values (first 10): {samples.flatten()[:10].tolist()}")
print("\n✓ Example completed successfully! The next run should re-use the cached kernel.")

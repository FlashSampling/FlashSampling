# FMMS Algorithm: Fused Matrix Multiplication & Sampling

High-performance GPU implementation of fused matrix multiplication + sampling using Triton.
This package provides an efficient kernel for sampling from categorical distributions where logits are computed on-the-fly from matrix multiplication, avoiding the need to materialize the full logit tensor in GPU main memory (GMEM).
The key insight is that in LLM decode workloads, both the matmul and the sampling are memory-bound (the matmul collapses to a matrix-vector product).
By fusing both operations, we avoid round-trips to GPU main memory (GMEM) and speed up the sampling process.

![PyTorch Sampling vs FMMS](./imgs/baseline-vs-fmms-diagram.png)

## Features

- **Bandwidth-Efficient**: Fuses matrix multiplication and sampling into a single Triton kernel, avoiding materialization of intermediate logit tensors, and preventing round-trips to GMEM.
- **Exact**: Uses Gumbel-max trick for efficient categorical sampling. No approximations.
- **Flexible**: Supports temperature scaling and multiple samples per hidden state vector.

## Installation

```bash
# Clone the repository
git clone https://github.com/tomasruizt/fused-mm-sample.git
cd fused-mm-sample

# Install the package (assumes you're in a virtual environment)
uv pip install -e ".[dev]"

# Verify installation
python examples/basic_usage.py
```

## Usage (Standalone)

For a complete working example, see [`examples/basic_usage.py`](examples/basic_usage.py).
The basic usage pattern:

```python
from fused_mm_sampling import fused_mm_sample_triton

samples = fused_mm_sample_triton(
    weights=weights,        # [vocab_size, hidden_size]
    hidden_states=hidden_states,  # [n_hidden_states, hidden_size]
    num_samples=1,
    temperature=torch.tensor(1.0, device="cuda"),  # scalar (0-d) CUDA tensor
    seed=42  # Optional: for reproducibility
)
# Returns: [n_hidden_states, num_samples]
```

**Parameters**:

- **`weights`** (Tensor): Weight matrix of shape `[vocab_size, hidden_size]`
- **`hidden_states`** (Tensor): Hidden states of shape `[n_hidden_states, hidden_size]`
- **`num_samples`** (int): Number of samples to draw per sequence position
- **`temperature`** (Tensor): Scalar (0-d) CUDA tensor for temperature scaling (higher = more random)
- **`seed`** (int, optional): Random seed for reproducibility

**Returns**: Tensor of shape `[n_hidden_states, num_samples]` containing sampled indices

## Usage in vLLM

Check out the branch `feature/fmms-sampler` from my vLLM fork and install vLLM from local sources. The [code diff](https://github.com/tomasruizt/vllm/pull/13/changes) is minimal.

```shell
git clone https://github.com/tomasruizt/vllm.git
cd vllm
git checkout feature/fmms-sampler
VLLM_USE_PRECOMPILED=1 uv pip install -e .
```

Then launch any model. Use the flags below to activate the FMMS sampler:

```shell
VLLM_USE_FMMS_SAMPLER=1 VLLM_FMMS_PROVIDER=fused-triton vllm serve Qwen/Qwen3-1.7B
```

## Algorithm

The FMMS kernel implements the Gumbel-max trick for categorical sampling:

1. **Matrix Multiplication**: Compute a tile of logits = hidden_states @ weights in SRAM
2. **Temperature Scaling**: Scale logits by temperature
3. **Gumbel Noise**: Add Gumbel noise to scaled logits tile
4. **Argmax**: Take argmax within the tile to get samples

The FMMS kernel computes these steps in blocks without materializing the full logit tensor, preventing memory accesses, and relieving the bottleneck on the memory bandwidth.

## Benchmarks

Kernel microbenchmarks across B300, B200, H200, and H100 GPUs, roofline analysis, and end-to-end vLLM integration results are in the [blog post](https://tomasruizt.github.io/tomas-blog/posts/07_fused-mm-sample/).

### Running the Benchmarks

```bash
# Benchmark all implementations
python speed_test.py

# Compare performance over many batch sizes
make triton-benchmark

# Run all microbenchmarks on Modal (B300, B200, H200, H100)
make modal-triton-benchmark-all-gpus
```

## Profiling

All profiling scripts are located in the `benchmarking/` directory.

### Memory Profiling

```bash
cd benchmarking
make profile-mem
```

This will generate a memory snapshot and HTML visualization in `benchmarking/memory/`.

### NVIDIA Nsight Compute Profiling

```bash
cd benchmarking

# Profile fused Triton kernel
make ncu-profile-fused-triton

# Profile naive compiled implementation
make ncu-profile-naive-compiled
```

### NVIDIA Nsight Systems Profiling

```bash
cd benchmarking

# Profile fused Triton kernel
make nsight-profile-fused-triton

# Profile naive compiled implementation
make nsight-profile-naive-compiled
```

## Development

### Development Environment

The dev dependencies permit running the scripts in the `benchmarking/` directory. To install them, run:

```bash
uv pip install -e ".[dev]"
```

### Modal Setup
The experiments involving many differnt GPUs were run on Modal. To install and login to Modal:

```bash
uv pip install modal
modal setup
```

Run the speed-test on modal:

```bash
make modal-speed-test
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to create an issue or submit a pull request.

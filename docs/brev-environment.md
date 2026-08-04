# NVIDIA Brev environment

## Supported provisioning path

Create the current Brev instance from the repository root:

```bash
make brev-create
```

The target requests `dmz.h100x2.pcie` and uses `nvidia/cuda:13.0.0-devel-ubuntu22.04`.
After the instance starts, run:

```bash
HOME=/home/shadeform bash scripts/brev-bootstrap.sh
```

The bootstrap script performs the supported setup:

1. It detects the highest CUDA toolkit under `/usr/local/cuda-*`.
2. It installs the current `cuda-toolkit` and `g++-12` only when Nsight Compute is unavailable.
3. It clones or fast-forwards `https://github.com/FlashSampling/FlashSampling.git`.
4. It installs `uv` when needed and runs `uv sync --all-extras`.
5. It creates the repository-local `.venv`.
6. It runs an import smoke test with `.venv/bin/python`.

The previous stock-image workflow used a global `/home/shadeform/.venv` and manually installed CUDA 12.2.
Do not apply those old instructions to an instance created by `make brev-create`.

## Environment details

- Some Shadeform non-login shells leave `HOME` unset.
  Pass `HOME=/home/shadeform` to the bootstrap and to later commands that rely on home-directory expansion.
- Use the repository `.venv` created by `uv sync`.
- Set `HF_TOKEN` before downloading gated Hugging Face models.
- Verify the selected toolkit with `nvcc --version`, `ncu --version`, and `echo "$CUDA_HOME"` before profiling or compiling JIT extensions.
- The bootstrap chooses `CUDA_HOME` from the detected toolkit and adds its `bin` directory to `PATH`.
- See [profiling.md](profiling.md) for current NCU and nsys workflows.

## Running benchmarks

Activate the environment or put its executables first on `PATH`, then run the benchmark Makefile:

```bash
cd /home/shadeform/code/FlashSampling
source .venv/bin/activate
HOME=/home/shadeform make -C benchmarking sweep-bsz-all
```

For a focused sweep:

```bash
HOME=/home/shadeform make -C benchmarking sweep-bsz-all CASE=small
HOME=/home/shadeform make -C benchmarking sweep-bsz-all CASE=large SWEEP_BSZ='1 16 64 256'
```

Save benchmark and profiler output under the configured results directory as required by `AGENTS.md`.

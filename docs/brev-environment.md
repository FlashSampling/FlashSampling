# NVIDIA Brev machine quirks

The Brev cloud GPU environment (shadeform) has several non-standard behaviors:

- **`$HOME` is unset** in non-login shells. Always pass `HOME=/home/shadeform` explicitly when running `make` or scripts that depend on `~` expansion. The Makefile's `$(HOME)` resolves to empty string otherwise.
- **Single global venv at `/home/shadeform/.venv/`**, not per-project. Both vLLM and fused-mm-sampling are installed there. The project's `.venv/` (referenced in the Makefile as `$(HOME)/code/fused-mm-sample/.venv/`) and vLLM's `venv/` (`$(HOME)/code/vllm/venv/`) do not exist.
- **vLLM binary**: `/home/shadeform/.venv/bin/vllm` (not `~/code/vllm/venv/bin/vllm`).
- **Python**: `/home/shadeform/.venv/bin/python` (Python 3.10.12).
- **GPU**: 1x NVIDIA H100 PCIe, 81,559 MiB VRAM, CUDA 13.0.
- **`datasets` / `pyarrow` conflict**: The pre-installed `datasets==2.14.4` is incompatible with `pyarrow==23.0.0` (`pa.PyExtensionType` was removed). Fix: `pip install --upgrade datasets` (upgrades to 4.5.0+).
- **HuggingFace**: Not logged in by default. Set `HF_TOKEN` env var for gated models.
- **Pip cache**: `/ephemeral/cache/pip` has wrong permissions; pip disables cache automatically (harmless warning).
- **Makefile portability**: Both `benchmarking/Makefile` and `benchmarking/vllm/Makefile` use `$(shell which python)` / `$(shell which vllm)` to discover binaries dynamically. No hard-coded paths — just activate the correct venv before running `make`. Example:
  ```bash
  HOME=/home/shadeform make -C benchmarking/vllm quick \
    MODEL=openai/gpt-oss-120b \
    HF_TOKEN=<token>
  ```

## CUDA toolkit installation on Brev (H100)

The Brev image ships with the NVIDIA driver (CUDA runtime 13.0) but **no nvcc** by default. Several components require nvcc for JIT compilation:

- **`fused-cuda` provider**: Uses `torch.utils.cpp_extension.load()` to JIT-compile a CUDA C++ extension. Needs nvcc + a compatible C++ compiler (g++-12).
- **flashinfer**: JIT-compiles CUDA kernels on first use. Requires nvcc with sm_90a support.
- **`tvm_ffi`**: The `_optional_torch_c_dlpack.py` module JIT-compiles a helper shared library. Non-fatal if it fails (just a warning).

**Installation steps:**

1. The CUDA 12.2 local repo deb is pre-cached at `/var/cuda-repo-ubuntu2204-12-2-local/`. Install from there:
   ```bash
   sudo dpkg -i /var/cuda-repo-ubuntu2204-12-2-local/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get install -y cuda-toolkit-12-2
   ```
2. Install a compatible C++ compiler:
   ```bash
   sudo apt-get install -y g++-12
   ```
3. Set environment variables:
   ```bash
   export CUDA_HOME=/usr/local/cuda-12.2
   export PATH=$CUDA_HOME/bin:$PATH
   ```

**Key pitfalls:**
- The Brev image may have CUDA 11.5 nvcc pre-installed (`/usr/local/cuda-11.5/bin/nvcc`). This is **too old for H100** (doesn't support `compute_90a` / `sm_90a`). You must use CUDA 12.0+ for H100.
- `tests/conftest.py` auto-discovers CUDA_HOME by searching `/usr/local/cuda-*` (preferring highest version) and validates that nvcc supports the current GPU's compute capability. If validation fails, it raises a clear error message.
- After installing a new CUDA toolkit, delete stale JIT caches: `rm -rf /ephemeral/cache/torch_extensions/` to force recompilation with the new nvcc.

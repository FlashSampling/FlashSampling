#!/usr/bin/env bash
# Bootstrap script for running sweep-bsz-all on a fresh Brev instance.
# Usage: curl/scp this to the Brev machine, then run:
#   HOME=/home/shadeform bash brev-bootstrap.sh
set -euo pipefail

echo "=== Brev bootstrap for fused-mm-sample ==="

# ── 1. CUDA toolkit (needed for ncu + nvcc JIT compilation) ──
if ! command -v ncu &>/dev/null; then
    echo "Installing CUDA toolkit 12.2..."
    sudo dpkg -i /var/cuda-repo-ubuntu2204-12-2-local/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
    sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get install -y cuda-toolkit-12-2 g++-12
else
    echo "ncu already available: $(which ncu)"
fi

export CUDA_HOME=/usr/local/cuda-12.2
export PATH=$CUDA_HOME/bin:$PATH

# ── 2. Clone repo ──
REPO_DIR="$HOME/code/fused-mm-sample"
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning repo..."
    mkdir -p "$HOME/code"
    git clone git@github.com:tomasruizt/fmms-kernel.git "$REPO_DIR"
else
    echo "Repo already exists at $REPO_DIR, pulling latest..."
    git -C "$REPO_DIR" pull --ff-only
fi
cd "$REPO_DIR"

# ── 3. Install uv + dependencies ──
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating venv and installing dependencies..."
uv sync --all-extras

# Clear stale JIT caches
rm -rf /ephemeral/cache/torch_extensions/ 2>/dev/null || true

# ── 4. Smoke test ──
echo "Smoke test: importing fused_mm_sampling..."
.venv/bin/python -c "import fused_mm_sampling; print('OK')"

echo ""
echo "=== Bootstrap complete ==="
echo "To run the sweep:"
echo "  cd $REPO_DIR/benchmarking"
echo "  HOME=$HOME make sweep-bsz-all"
echo ""
echo "Or for a specific case/batch sizes:"
echo "  HOME=$HOME make sweep-bsz-all CASE=small"
echo "  HOME=$HOME make sweep-bsz-all CASE=large SWEEP_BSZ='1 16 64 256'"
echo ""
echo "To copy results back (run from your local machine):"
echo "  brev copy <instance-name>:$REPO_DIR/benchmarking/profiles/sweeps/ ./sweeps-results/"

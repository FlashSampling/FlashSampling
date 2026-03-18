#!/usr/bin/env bash
# Bootstrap script for running sweep-bsz-all on a fresh Brev instance.
# Usage: curl/scp this to the Brev machine, then run:
#   HOME=/home/shadeform bash brev-bootstrap.sh
set -euo pipefail

echo "=== Brev bootstrap for fused-mm-sample ==="

# ── 1. CUDA toolkit (needed for ncu + nvcc JIT compilation) ──
# Auto-detect the highest CUDA version installed under /usr/local/cuda-*.
# Falls back to installing cuda-toolkit from apt if ncu is not found.
if ! command -v ncu &>/dev/null; then
    # Try to find an existing CUDA toolkit
    CUDA_DIR=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
    if [ -n "$CUDA_DIR" ] && [ -x "$CUDA_DIR/bin/ncu" ]; then
        echo "Found CUDA toolkit at $CUDA_DIR"
    else
        echo "No CUDA toolkit found. Installing latest cuda-toolkit..."
        sudo apt-get update
        # Install the latest available cuda-toolkit and g++
        sudo apt-get install -y cuda-toolkit g++-12
        CUDA_DIR=$(ls -d /usr/local/cuda-* 2>/dev/null | sort -V | tail -1)
    fi
else
    echo "ncu already available: $(which ncu)"
    CUDA_DIR=$(dirname "$(dirname "$(which ncu)")")
fi

export CUDA_HOME="${CUDA_DIR:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
echo "Using CUDA_HOME=$CUDA_HOME ($(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found'))"

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

# ── 4. Install Claude Code ──
if ! command -v claude &>/dev/null; then
    echo "Installing Claude Code..."
    # Install Node.js via nvm if not available
    if ! command -v node &>/dev/null; then
        curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
        export NVM_DIR="$HOME/.nvm"
        # shellcheck source=/dev/null
        [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
        nvm install --lts
    fi
    npm install -g @anthropic-ai/claude-code
else
    echo "Claude Code already installed: $(claude --version)"
fi

# ── 5. Smoke test ──
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

#!/usr/bin/env bash
# Build an anonymized supplemental zip of the FlashSampling repo for NeurIPS submission.
#
# Allowlist approach: copies only the items needed for reproduction into a clean
# staging directory, then scans for residual identifiers.
#
# Steps:
#   1. Create a clean staging directory at /tmp/FlashSampling-anon/
#   2. Copy the allowlist of files/dirs from the source repo into staging
#   3. Grep the staging directory for author names and home paths and report hits
#   4. Print next manual steps (rewrite README.md, scrub pyproject.toml, then zip)
#
# Usage: scripts/make-anon-zip.sh [SRC_DIR] [STAGE_DIR]
#   SRC_DIR    defaults to $HOME/code/FlashSampling
#   STAGE_DIR  defaults to /tmp/FlashSampling-anon

set -euo pipefail

SRC="${1:-$HOME/code/FlashSampling}"
STAGE="${2:-/tmp/FlashSampling-anon}"

if [[ ! -d "$SRC" ]]; then
  echo "error: source directory not found: $SRC" >&2
  exit 1
fi

# Allowlist of paths to copy from $SRC into $STAGE.
# Keep this minimal: only what reviewers need to reproduce the results.
INCLUDE=(
  "src"
  "tests"
  "examples"
  "Makefile"
  "pyproject.toml"
  "uv.lock"
  "LICENSE"
  ".gitignore"
  ".pre-commit-config.yaml"
  "REPRODUCTION.md"

  # benchmarking/ is allowlisted file-by-file: only the scripts referenced by the
  # Make targets in REPRODUCTION.md (Sections 3.1, 3.2, 3.3, 6) are shipped.
  # Exploratory notebooks, nsys wrappers, profile-mem, matmul_comparison, and
  # speed_test/ are deliberately omitted.
  "benchmarking/Makefile"
  "benchmarking/triton_benchmark.py"
  "benchmarking/plot-triton-bench.py"
  "benchmarking/plot_tp_scaling.py"
  "benchmarking/plot_lib.py"
  "benchmarking/plot_styles.py"
  "benchmarking/plot_bsz_sweep_runtime.py"
  "benchmarking/plot_ncu_kernel_breakdown.py"
  "benchmarking/parse_ncu_sweep.py"
  "benchmarking/parse_proton_intrakernel.py"
  "benchmarking/proton_profile.py"
  "benchmarking/insert_proton_records.py"
)

echo "==> Creating clean staging directory: $STAGE"
rm -rf "$STAGE"
mkdir -p "$STAGE"

echo "==> Copying allowlist from $SRC"
for item in "${INCLUDE[@]}"; do
  if [[ -e "$SRC/$item" ]]; then
    mkdir -p "$STAGE/$(dirname "$item")"
    cp -r "$SRC/$item" "$STAGE/$item"
    echo "  + $item"
  else
    echo "  - $item (not present in source, skipped)"
  fi
done

echo "==> Removing transient artifacts and bulky raw results"
# Caches, bytecode, build artifacts
find "$STAGE" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type d -name ".cache" -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -type f -name "*.pyc" -delete 2>/dev/null || true
find "$STAGE" -type f -name "*.log" -delete 2>/dev/null || true

# benchmarking/vllm/ is excluded by the path allowlist. modal_vllm_benchmark.py
# is kept for reviewer inspection but its fork URL, branch, and commit SHA are
# anonymized below; the workflow is illustrative since the anonymous mirror does
# not support `git clone` over the git protocol.

# Drop checked-in model weight snapshots; reviewers fetch weights from HuggingFace.
find "$STAGE/tests" -type f \( -name "weights.pt" -o -name "*.safetensors" \) -delete 2>/dev/null || true

# REPRODUCTION.md is the supplemental's README. The original README.md is not copied.
if [[ -f "$STAGE/REPRODUCTION.md" ]]; then
  mv "$STAGE/REPRODUCTION.md" "$STAGE/README.md"
  echo "  REPRODUCTION.md -> README.md"
fi

# Mask vLLM fork identifiers (URL, branch name, commit SHA). The values point
# to a private GitHub fork; reviewers see anonymized placeholders instead.
VLLM_FORK_ANON_URL="https://anonymous.4open.science/r/vllm-6E4E"
VLLM_FORK_ANON_BRANCH="<ANON-BRANCH>"
VLLM_FORK_ANON_SHA="<ANON-SHA>"
find "$STAGE" -type f \( -name "*.py" -o -name "*.md" -o -name "*.sh" \) -print0 \
  | xargs -0 sed -i \
      -e "s|https://github.com/tomasruizt/vllm[a-zA-Z.]*|${VLLM_FORK_ANON_URL}|g" \
      -e "s|feature/fmms-sampler|${VLLM_FORK_ANON_BRANCH}|g" \
      -e "s|3170fdf3da4e09a76031fa515698c86d1fcbc699|${VLLM_FORK_ANON_SHA}|g"

# Anonymize pyproject.toml: clear authors, mask GitHub URLs.
if [[ -f "$STAGE/pyproject.toml" ]]; then
  python3 - "$STAGE/pyproject.toml" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(
    r'authors\s*=\s*\[[^\]]*\]',
    'authors = [\n    { name = "Anonymous" }\n]',
    s, flags=re.DOTALL)
s = re.sub(r'https?://github\.com/tomasruizt/[^"\s]+', 'https://anonymous.example/repo', s)
open(p, 'w').write(s)
PY
fi

echo "==> Scrubbing for residual identifiers"
HITS=0
SCRUB_PATTERNS=(
  "tomasruiz"
  "Tomas Ruiz"
  "/home/tomasruiz"
  "Zhen Qin"
  "Yifan Zhang"
  "Xuyang Shen"
  "Yiran Zhong"
  "Mengdi Wang"
)
for pat in "${SCRUB_PATTERNS[@]}"; do
  if grep -rIl --binary-files=without-match "$pat" "$STAGE" >/dev/null 2>&1; then
    echo "  ! '$pat' found in:"
    grep -rIl --binary-files=without-match "$pat" "$STAGE" | sed "s|^$STAGE/|    |"
    HITS=$((HITS + 1))
  fi
done

echo
echo "==> Staging complete: $STAGE"
du -sh "$STAGE"
echo
if [[ $HITS -gt 0 ]]; then
  echo "ERROR: $HITS identifying pattern(s) still present. Fix the listed files (or extend SCRUB_PATTERNS) and re-run; aborting before zip." >&2
  exit 1
fi
echo "No identifier hits."

echo
echo "==> Creating zip"
ZIP_PATH="$(dirname "$STAGE")/$(basename "$STAGE").zip"
rm -f "$ZIP_PATH"
( cd "$(dirname "$STAGE")" && zip -rq "$(basename "$ZIP_PATH")" "$(basename "$STAGE")" )
du -h "$ZIP_PATH"
echo "Zip: $ZIP_PATH"

echo
echo "Final manual checks recommended before uploading to OpenReview:"
echo "  - Skim $STAGE/README.md for any author-revealing content not caught by SCRUB_PATTERNS."
echo "  - Skim $STAGE/benchmarking/ for the same."

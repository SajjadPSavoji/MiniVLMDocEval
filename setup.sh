#!/usr/bin/env bash
# Reproducible setup for MiniVLMDocEval.
#
# Installs VLMEvalKit as a PINNED DEPENDENCY (not forked). Custom model wrappers
# live in our own code and are registered via VLMEvalKit's plugin path; the
# installed package is never modified.
#
# Why editable (-e) and not `pip install git+...`:
#   VLMEvalKit's setup.py uses find_packages(), which drops the ~20 subdirs that
#   lack an __init__.py (e.g. megabench/parsing). Building a wheel therefore
#   produces a BROKEN install. Editable install uses the source tree directly
#   and includes everything. This matches VLMEvalKit's own install docs.
set -euo pipefail

COMMIT=2cf2a36c6e79b51faf676a7011b3a0f5b579814d   # HEAD of main @ 2026-06-26 (just ahead of v0.3rc1)
REPO=https://github.com/open-compass/VLMEvalKit.git
DIR=external/VLMEvalKit

# 1. Clone VLMEvalKit at the pinned commit (idempotent).
if [ ! -d "$DIR/.git" ]; then
  git clone "$REPO" "$DIR"
fi
git -C "$DIR" fetch --quiet origin "$COMMIT"
git -C "$DIR" checkout --quiet "$COMMIT"
echo "VLMEvalKit pinned at $(git -C "$DIR" rev-parse --short HEAD)"

# 2. Install VLMEvalKit's deps.
#    mac-arm64: `decord` (video) has no wheel and we evaluate image-only, so we
#    substitute a local stub and skip the real decord. Linux/Colab uses it as-is.
if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
  echo "mac-arm64 detected -> installing local decord stub (image-only eval)"
  pip install ./tools/decord-stub
  grep -vi '^decord' "$DIR/requirements.txt" | pip install -r /dev/stdin
else
  pip install -r "$DIR/requirements.txt"
fi

# 3. Install VLMEvalKit itself (editable, no deps — deps handled above).
pip install --no-deps -e "$DIR"

echo "Setup complete. Verify with:"
echo "  python -c 'from vlmeval.config import supported_VLM; print(len(supported_VLM), \"models\")'"

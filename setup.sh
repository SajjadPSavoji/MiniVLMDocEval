#!/usr/bin/env bash
# Reproducible setup for MiniVLMDocEval (targets Linux/Colab).
#
# Clones VLMEvalKit at a PINNED commit into external/ (gitignored), then installs
# it. VLMEvalKit is a dependency — never modified; our custom model wrappers live
# in our own code and register via its plugin path. The local clone also serves
# as a read-only reference while coding.
#
# Why editable (-e) and not a wheel:
#   VLMEvalKit's setup.py find_packages() drops subdirs lacking __init__.py
#   (e.g. megabench/parsing), so a built wheel is broken. Editable uses the
#   source tree directly. Matches VLMEvalKit's own install docs.
set -euo pipefail

COMMIT=2cf2a36c6e79b51faf676a7011b3a0f5b579814d   # HEAD of main @ 2026-06-26 (just ahead of v0.3rc1)
REPO=https://github.com/open-compass/VLMEvalKit.git
DIR=external/VLMEvalKit

# 1. Clone at the pinned commit (idempotent; replace any non-git leftover).
if [ ! -d "$DIR/.git" ]; then
  rm -rf "$DIR"
  git clone "$REPO" "$DIR"
fi
git -C "$DIR" fetch --quiet origin "$COMMIT"
git -C "$DIR" checkout --quiet "$COMMIT"
echo "VLMEvalKit pinned at $(git -C "$DIR" rev-parse --short HEAD)"

# 2. Install deps + the editable package.
pip install -r "$DIR/requirements.txt"
pip install --no-deps -e "$DIR"

echo "Setup complete. Verify with:"
echo "  python -c 'from vlmeval.config import supported_VLM; print(len(supported_VLM), \"models\")'"

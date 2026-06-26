#!/usr/bin/env bash
# Reproducible setup for MiniVLMDocEval.
#
# VLMEvalKit is VENDORED at external/VLMEvalKit (committed to this repo, pinned to
# an exact upstream commit — see external/README.md). This script does NOT clone
# anything; it just installs the vendored copy. It is never modified — our custom
# model wrappers live in our own code and register via VLMEvalKit's plugin path.
#
# Why editable (-e) and not a wheel:
#   VLMEvalKit's setup.py uses find_packages(), which drops the ~20 subdirs that
#   lack an __init__.py (e.g. megabench/parsing). Building a wheel therefore
#   produces a BROKEN install. Editable install uses the source tree directly
#   and includes everything. This matches VLMEvalKit's own install docs.
set -euo pipefail

DIR=external/VLMEvalKit   # vendored, pinned (see external/README.md)

if [ ! -f "$DIR/setup.py" ]; then
  echo "ERROR: $DIR is missing. It is vendored in this repo — make sure you cloned"
  echo "the full repository (it should be present without any extra step)." >&2
  exit 1
fi

# 1. Install VLMEvalKit's deps (targets Linux/Colab; `decord` has a Linux wheel).
pip install -r "$DIR/requirements.txt"

# 2. Install VLMEvalKit itself (editable, no deps — deps handled above).
pip install --no-deps -e "$DIR"

echo "Setup complete. Verify with:"
echo "  python -c 'from vlmeval.config import supported_VLM; print(len(supported_VLM), \"models\")'"

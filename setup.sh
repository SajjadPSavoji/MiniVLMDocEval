#!/usr/bin/env bash
# Reproducible setup for MiniVLMDocEval (targets Linux/Colab).
#
# Clones VLMEvalKit at a PINNED commit into external/ (gitignored), then installs
# it. VLMEvalKit is a dependency — never modified; our custom model wrappers live
# in our own code and register via its plugin path. The local clone also serves
# as a read-only reference while coding.
#
# Two transformers environments:
#   (default)        transformers<4.57  -> "Env A": the 3 built-ins + FastVLM-0.5B
#   --bleeding-edge  transformers@main  -> "Env B": Qwen3.5-0.8B + LFM2.5-VL-450M
#     LFM2.5-VL's processor (Lfm2VlProcessor) is native only in transformers>=4.57,
#     and Qwen3.5 upstreamed 2026-02; both need Env B.
# Both write results to the same Drive output tree, so models from either env
# merge into one comparison table.
#
# Why editable (-e) and not a wheel:
#   VLMEvalKit's setup.py find_packages() drops subdirs lacking __init__.py
#   (e.g. megabench/parsing), so a built wheel is broken. Editable uses the
#   source tree directly. Matches VLMEvalKit's own install docs.
set -euo pipefail

# Flags (combinable): --bleeding-edge (Env B transformers@main) and --train (add
# LoRA fine-tuning deps for the Part-2 PoC; should accompany --bleeding-edge since
# training Qwen3.5-0.8B needs Env B).
BLEEDING=0
TRAIN=0
for arg in "$@"; do
  case "$arg" in
    --bleeding-edge) BLEEDING=1 ;;
    --train) TRAIN=1 ;;
    "") ;;
    *) echo "unknown arg: $arg (use --bleeding-edge and/or --train)" >&2; exit 2 ;;
  esac
done
[[ "$TRAIN" == 1 && "$BLEEDING" == 0 ]] && \
  echo "warn: --train without --bleeding-edge; Qwen3.5-0.8B training needs Env B (transformers@main)."

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

# 2. Install deps, then pin transformers per environment.
pip install -r "$DIR/requirements.txt"
if [[ "$BLEEDING" == 1 ]]; then
  echo "Env B: installing transformers from main (Qwen3.5-0.8B)"
  pip install --upgrade "git+https://github.com/huggingface/transformers.git"
else
  # Env A: newer transformers changed tied-weight bookkeeping for remote-code
  # models, which breaks InternVL3 loading; pin below that.
  pip install "transformers<4.57"
fi

# 3. Install VLMEvalKit itself (editable, no deps — deps handled above).
pip install --no-deps -e "$DIR"

# 4. (optional) Training extras for the Part-2 LoRA PoC (scripts/train_lora_qwen.py).
#    bf16 LoRA only — no bitsandbytes/4-bit (gated-delta is quantization-sensitive).
if [[ "$TRAIN" == 1 ]]; then
  echo "Train add-on: installing peft + trl + accelerate (LoRA fine-tuning)"
  pip install "peft>=0.11" "trl>=0.9" "accelerate>=0.30"
  # Colab preinstalls torchao 0.10, which peft's import check rejects (wants >0.16)
  # and RAISES on get_peft_model. We don't use torchao (bf16 LoRA, no quantization),
  # so remove it -> peft's is_torchao_available() then returns False cleanly.
  pip uninstall -y torchao || true
fi

LABEL="default env"; [[ "$BLEEDING" == 1 ]] && LABEL="Env B"; [[ "$TRAIN" == 1 ]] && LABEL="$LABEL +train"
echo "Setup complete ($LABEL). Verify with:"
echo "  python -c 'from vlmeval.config import supported_VLM; print(len(supported_VLM), \"models\")'"

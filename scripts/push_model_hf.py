#!/usr/bin/env python
"""Push the trained LoRA adapter to the HuggingFace Hub (as a PEFT adapter repo).

The adapter is saved by train_lora_qwen.py under <out>/lora_adapters/<run>/ (on
Drive when run on Colab). This uploads it to <username>/<name>, skipping the bulky
intermediate trainer checkpoints, and writes an honest model card.

IMPORTANT: our released adapter is the Part-2 experimental run that FAILED the
pre-registered no-regression gate (it regressed the model). It is published for
transparency/reproducibility of the negative result, NOT as a recommended
checkpoint. The generated card says so.

Secrets are runtime-only: HF write token via $HF_TOKEN (or --token).

Usage (Colab terminal, Drive mounted):
  HF_TOKEN=hf_xxx python scripts/push_model_hf.py \
      --adapter-dir $OUT_DIR/lora_adapters/table_lora_v1 \
      --repo-id <HF_USERNAME>/qwen3.5-0.8b-tablevqa-lora [--private]
"""
import argparse
import json
import os
import sys
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
REPO_URL = "https://github.com/SajjadPSavoji/MiniVLMDocEval"


def card(repo_id, base_model, meta):
    hp = ""
    if meta:
        hp = (f"r={meta.get('r')}, alpha={meta.get('alpha')}, lr={meta.get('lr')}, "
              f"epochs={meta.get('epochs')}, batch={meta.get('batch')}x"
              f"grad_accum={meta.get('grad_accum')}")
    return f"""---
base_model: {base_model}
library_name: peft
tags:
- peft
- lora
- vision-language
- document-understanding
- tablevqa
- qwen
---

# {repo_id.split('/')[-1]}

A **LoRA adapter** for [`{base_model}`](https://huggingface.co/{base_model}), from the
[MiniVLMDocEval]({REPO_URL}) Part-2 study. It targets the model's measured weakness on
**visual table lookup** (Wikipedia-style VWTQ) via bf16 LoRA with the vision encoder frozen.

## ⚠️ This is a research artifact, not a recommended checkpoint

This adapter **failed** the project's pre-registered *no-regression* gate: instead of a
targeted gain it **catastrophically regressed** the base model
(TableVQABench 50.1 → 19.6, DocVQA 89.3 → 50.8, InfoVQA 62.3 → 27.8). The diagnosis is a
**train/evaluation answer-style mismatch** — the synthetic training source supplies verbose,
full-sentence answers while the benchmarks are scored on short cell values/spans — which
collapsed the model's output distribution. It is published for **transparency and
reproducibility of the negative result**; do not use it as an improved model. The full
diagnosis and a corrected recipe are in the technical report at {REPO_URL}.

## Usage

```python
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import PeftModel

base = AutoModelForImageTextToText.from_pretrained("{base_model}", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "{repo_id}").eval()
processor = AutoProcessor.from_pretrained("{repo_id}")
```

## Training

bf16 LoRA, vision encoder frozen, adapters on the language + gated-delta projections;
{hp or "see train_meta.json"}. One epoch on a ~7k VWTQ-weighted mixture with ~25%
general-document replay. Reproduce with [`scripts/train_lora_qwen.py`]({REPO_URL}).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True, help="dir with adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--repo-id", required=True, help="<username>/<model-name>")
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF write token; defaults to $HF_TOKEN (preferred)")
    args = ap.parse_args()

    if not args.token:
        raise SystemExit("No HF token. Set HF_TOKEN env var (role: write).")
    if "/" not in args.repo_id:
        raise SystemExit("--repo-id must be '<username>/<model-name>'.")
    adir = Path(args.adapter_dir)
    if not (adir / "adapter_config.json").exists():
        raise SystemExit(f"No adapter_config.json in {adir} — is this the adapter dir?")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    meta = {}
    mp = adir / "train_meta.json"
    if mp.exists():
        try:
            meta = json.loads(mp.read_text())
        except Exception:                                # noqa: BLE001
            pass
    (adir / "README.md").write_text(card(args.repo_id, args.base_model, meta))

    print(f"[push] uploading adapter {adir} -> {args.repo_id} (private={args.private}) ...")
    api.upload_folder(
        folder_path=str(adir), repo_id=args.repo_id, repo_type="model",
        ignore_patterns=["checkpoint-*", "*/optimizer.pt", "*/scheduler.pt", "*/rng_state*"],
    )
    print(f"[push] done -> https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

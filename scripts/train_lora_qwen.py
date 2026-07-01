#!/usr/bin/env python
"""Light bf16 LoRA fine-tune of Qwen3.5-0.8B for the TableVQA PoC (Part 2).

Targets the measured weakness (Wikipedia-style table lookup; see
tablevqa_subdomain_report.py). Design decisions (research_plan.md §6.3):
  - bf16 LoRA, NO 4-bit (gated-delta decay gates are quantization-sensitive).
  - Vision encoder FROZEN — LoRA only on language-side linear layers, including the
    gated-delta projections (75% of Qwen3.5's mixer layers are Gated-DeltaNet, so a
    standard attention-only target set would miss most of the model).
  - Prompt/answer formatting mirrors custom_models.Qwen35VL.generate_inner exactly,
    so the model trains under the same chat template it is evaluated with.

ALWAYS run `--smoke` first (H0-1 de-risk): it confirms the adapter attaches to this
bleeding-edge architecture, that gradients flow, and that outputs actually change —
before investing in a full data build / training run.

Usage (Env B: transformers@main + peft + trl + accelerate):
  python scripts/train_lora_qwen.py --smoke
  python scripts/train_lora_qwen.py --out $OUT_DIR --data $OUT_DIR/train_data/train.jsonl \
      --run-name table_lora_v1 --epochs 1
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

BASE_MODEL = "Qwen/Qwen3.5-0.8B"

# Candidate LoRA target leaf-names: attention + MLP + gated-delta projections.
# Intersected at runtime with the model's actual nn.Linear leaves (minus vision).
CANDIDATE_TARGETS = {
    "q_proj", "k_proj", "v_proj", "o_proj",            # gated attention
    "gate_proj", "up_proj", "down_proj",               # MLP
    "in_proj", "out_proj", "in_proj_qkvz", "in_proj_ba",  # gated-delta projections
}
VISION_HINTS = ("visual", "vision", "image_encoder", "vit", "merger", "patch_embed")


def log(m):
    print(m, flush=True)


def load_base(dtype_str="bf16"):
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForMultimodalLM as AutoVLM
    except Exception:                                   # noqa: BLE001
        from transformers import AutoModelForImageTextToText as AutoVLM
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_str]
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = AutoVLM.from_pretrained(BASE_MODEL, torch_dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    return model, processor


def pick_targets(model):
    """Language-side nn.Linear leaves whose names are in CANDIDATE_TARGETS and NOT
    under the vision tower. Returns a sorted list and logs the breakdown."""
    import torch.nn as nn
    found, skipped_vision = set(), set()
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        leaf = name.split(".")[-1]
        if leaf not in CANDIDATE_TARGETS:
            continue
        if any(h in name.lower() for h in VISION_HINTS):
            skipped_vision.add(leaf)
            continue
        found.add(leaf)
    log(f"[lora] target modules (language-side): {sorted(found)}")
    if skipped_vision:
        log(f"[lora] vision-side linears excluded from LoRA: {sorted(skipped_vision)}")
    if not found:
        raise SystemExit("No LoRA target modules matched — inspect model.named_modules(); "
                         "the architecture may use different projection names.")
    return sorted(found)


def build_peft(model, r, alpha, dropout):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=pick_targets(model),
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ----------------------------- data + collator ------------------------------

class JsonlDataset:
    """List-backed dataset of {image, prompt, answer, ...} records."""
    def __init__(self, path):
        self.rows = [json.loads(l) for l in open(path) if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    """Builds processor inputs mirroring Qwen35VL.generate_inner, with labels masked
    on the prompt so loss is computed on the answer only."""
    def __init__(self, processor):
        self.processor = processor
        # right-pad so answer-label alignment from position 0 holds for batch>1
        if getattr(processor, "tokenizer", None) is not None:
            processor.tokenizer.padding_side = "right"

    def _text(self, prompt, with_answer, answer):
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        msgs = [{"role": "user", "content": content}]
        if with_answer:
            msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return self.processor.apply_chat_template(
            msgs, add_generation_prompt=not with_answer, tokenize=False)

    def __call__(self, batch):
        import torch
        from PIL import Image
        input_ids_list, labels_list, images = [], [], []
        for rec in batch:
            img = Image.open(rec["image"]).convert("RGB")
            full = self._text(rec["prompt"], True, rec["answer"])
            prompt_only = self._text(rec["prompt"], False, None)
            full_ids = self.processor(text=[full], images=[img], return_tensors="pt")
            prompt_ids = self.processor(text=[prompt_only], images=[img], return_tensors="pt")
            ids = full_ids["input_ids"][0]
            plen = prompt_ids["input_ids"].shape[1]
            labels = ids.clone()
            labels[:plen] = -100                        # supervise the answer only
            input_ids_list.append(ids)
            labels_list.append(labels)
            images.append(img)

        # re-run the processor on the whole batch to get correctly-batched image tensors
        full_texts = [self._text(r["prompt"], True, r["answer"]) for r in batch]
        enc = self.processor(text=full_texts, images=images, return_tensors="pt", padding=True)
        # pad labels to enc input_ids length
        seqlen = enc["input_ids"].shape[1]
        pad_id = self.processor.tokenizer.pad_token_id or 0
        labels = torch.full((len(batch), seqlen), -100, dtype=torch.long)
        for i, lab in enumerate(labels_list):
            n = min(len(lab), seqlen)
            labels[i, :n] = lab[:n]
        enc["labels"] = labels
        return enc


# ----------------------------------- smoke ----------------------------------

def smoke():
    """H0-1 de-risk: attach adapter, 1 fwd+bwd, save+reload, confirm outputs change."""
    import torch
    from peft import PeftModel
    log("[smoke] loading base ...")
    model, processor = load_base()
    model = build_peft(model, r=8, alpha=16, dropout=0.0)
    model.train()

    # tiny synthetic batch: a blank image + a trivial QA
    from PIL import Image
    rec = {"image": "_smoke.png", "prompt": "What color is the box?", "answer": "blue"}
    Image.new("RGB", (64, 64), (0, 0, 255)).save("_smoke.png")
    batch = Collator(processor)([rec])
    batch = {k: v.cuda() if hasattr(v, "cuda") else v for k, v in batch.items()}

    out = model(**batch)
    log(f"[smoke] forward OK, loss={float(out.loss):.4f}")
    out.loss.backward()
    gnorm = sum(p.grad.norm().item() for p in model.parameters() if p.requires_grad and p.grad is not None)
    log(f"[smoke] backward OK, adapter grad-norm={gnorm:.4e}")
    assert gnorm > 0, "adapter received NO gradient — LoRA not attached to active layers"

    tmp = Path("_smoke_adapter")
    model.save_pretrained(tmp)
    log(f"[smoke] saved adapter -> {tmp} ; reloading on fresh base ...")
    base2, proc2 = load_base()
    reloaded = PeftModel.from_pretrained(base2, tmp)
    log("[smoke] reload OK")
    log("\n[smoke] PASS — adapter attaches, grads flow, save/reload works on this arch.")


# ----------------------------------- train ----------------------------------

def train(args):
    import torch
    from transformers import TrainingArguments, Trainer
    model, processor = load_base()
    model = build_peft(model, r=args.r, alpha=args.alpha, dropout=args.dropout)
    model.config.use_cache = False
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    ds = JsonlDataset(args.data)
    log(f"[train] {len(ds)} examples from {args.data}")

    out_dir = Path(args.out) / "lora_adapters" / args.run_name
    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    trainer = Trainer(
        model=model, args=targs, train_dataset=ds, data_collator=Collator(processor),
    )
    trainer.train()
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    (out_dir / "train_meta.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log(f"\n[train] adapter saved -> {out_dir}")
    log(f"[train] evaluate with: MVDE_LORA_ADAPTER={out_dir} "
        f"python scripts/run_eval.py --out {args.out} --models Qwen3.5-0.8B-TableLoRA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="H0-1 adapter de-risk; no training")
    ap.add_argument("--out", help="results base dir (adapter saved under <out>/lora_adapters/<run>)")
    ap.add_argument("--data", help="train.jsonl from build_table_sft.py")
    ap.add_argument("--run-name", default="table_lora_v1")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--grad-ckpt", action="store_true", default=True)
    args = ap.parse_args()

    if args.smoke:
        smoke()
        return
    if not (args.out and args.data):
        ap.error("training needs --out and --data (or use --smoke)")
    train(args)


if __name__ == "__main__":
    main()

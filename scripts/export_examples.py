#!/usr/bin/env python
"""Export curated qualitative examples (full-res images + question/gold/prediction)
for the technical-report appendix.

Why a separate Colab step: our prediction .xlsx truncate large images at Excel's
32,767-char cell limit, so authentic images cannot be recovered from them. This
script instead pulls full-res images straight from the VLMEvalKit datasets and
joins them with the model's answers from the prediction .xlsx (whose text is
intact). It needs NO model load, so it runs fast in the eval env (either env).

Selection (matches the report design):
  * representative: a few CORRECT examples per dataset (the diagnostic ladder),
  * vwtq_failure:   TableVQA VWTQ/VWTQ-Syn rows the model got WRONG (the gap).

Writes <out>/examples/{<dataset>_<idx>.png, manifest.json}.

Usage (Colab terminal, after a completed run for --model):
  python scripts/export_examples.py --out $OUT_DIR --model Qwen3.5-0.8B
Then pull locally:
  rclone copy gdrive:MiniVLMDocEval/outputs/examples drive_sync/examples -P
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from minivlmdoceval.config import DATASETS, SAMPLE_SEED, DEFAULT_N, PREDICTIONS_SUBDIR

MAX_DIM = 1400  # downscale very large images so the pulled folder + PDF stay light


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def is_correct(pred, gold):
    """Lenient normalized match — a curation proxy, not the official scorer."""
    p, g = norm(pred), norm(gold)
    if not g or p in ("", "no answer"):
        return False
    return p == g or g in p or p in g


def safe(idx):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(idx)).strip("_")[:60]


def load_preds(out, model, dataset):
    """{index: (question, gold, prediction, split)} from the prediction xlsx."""
    import pandas as pd
    f = Path(out) / PREDICTIONS_SUBDIR / model / f"{dataset}_n{DEFAULT_N}.xlsx"
    if not f.exists():
        print(f"  [skip] no prediction file: {f}")
        return {}
    df = pd.read_excel(f)
    sp = df["split"] if "split" in df.columns else [""] * len(df)
    return {row["index"]: (row.get("question"), row.get("answer"),
                           row.get("prediction"), s)
            for (_, row), s in zip(df.iterrows(), sp)}


def save_image(ds, line, dest):
    from PIL import Image
    img = ds.dump_image(line)
    img = img[0] if isinstance(img, list) else img
    im = Image.open(img).convert("RGB")
    if max(im.size) > MAX_DIM:
        im.thumbnail((MAX_DIM, MAX_DIM))
    im.save(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="results base dir (predictions live here; examples written here)")
    ap.add_argument("--model", default="Qwen3.5-0.8B")
    ap.add_argument("--per-dataset", type=int, default=2, help="correct 'representative' examples per dataset")
    ap.add_argument("--failures", type=int, default=6, help="VWTQ/VWTQ-Syn failures to export")
    args = ap.parse_args()

    from vlmeval.dataset import build_dataset
    from run_eval import subset_data

    out_dir = Path(args.out) / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []

    # 1) representative correct examples per dataset (the diagnostic ladder)
    for ds_name in DATASETS:
        print(f"[representative] {ds_name}")
        preds = load_preds(args.out, args.model, ds_name)
        if not preds:
            continue
        ds = build_dataset(ds_name)
        data = subset_data(ds.data, DEFAULT_N, SAMPLE_SEED)
        taken = 0
        for _, line in data.iterrows():
            idx = line["index"]
            if idx not in preds:
                continue
            q, gold, pred, split = preds[idx]
            if not is_correct(pred, gold):
                continue
            fname = f"{ds_name}_{safe(idx)}.png"
            try:
                save_image(ds, line, out_dir / fname)
            except Exception as exc:                     # noqa: BLE001
                print(f"  [img skip {idx}] {type(exc).__name__}")
                continue
            manifest.append({"dataset": ds_name, "split": split, "index": str(idx),
                             "question": str(q), "gold": str(gold), "prediction": str(pred),
                             "correct": True, "role": "representative", "image": fname})
            taken += 1
            if taken >= args.per_dataset:
                break

    # 2) VWTQ / VWTQ-Syn failures (the gap centerpiece)
    print("[vwtq_failure] TableVQABench")
    preds = load_preds(args.out, args.model, "TableVQABench")
    if preds:
        ds = build_dataset("TableVQABench")
        data = subset_data(ds.data, DEFAULT_N, SAMPLE_SEED)
        n = 0
        for _, line in data.iterrows():
            idx = line["index"]
            if idx not in preds:
                continue
            q, gold, pred, split = preds[idx]
            if split not in ("vwtq", "vwtq_syn") or is_correct(pred, gold):
                continue
            fname = f"TableVQA_{safe(idx)}.png"
            try:
                save_image(ds, line, out_dir / fname)
            except Exception as exc:                     # noqa: BLE001
                print(f"  [img skip {idx}] {type(exc).__name__}")
                continue
            manifest.append({"dataset": "TableVQABench", "split": split, "index": str(idx),
                             "question": str(q), "gold": str(gold), "prediction": str(pred),
                             "correct": False, "role": "vwtq_failure", "image": fname})
            n += 1
            if n >= args.failures:
                break

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[done] {len(manifest)} examples -> {out_dir}")
    by_role = {}
    for m in manifest:
        by_role[m["role"]] = by_role.get(m["role"], 0) + 1
    print(f"[done] by role: {by_role}")
    print("[done] pull with: rclone copy gdrive:MiniVLMDocEval/outputs/examples drive_sync/examples -P")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # skip HF/torch interpreter-teardown segfault; work is saved above

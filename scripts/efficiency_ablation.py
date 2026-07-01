#!/usr/bin/env python
"""Reproduce the inference-efficiency ablations (fp16 speedup + max_new_tokens cap).

Runs small, fixed-seed subsets via run_eval.py under different settings and prints
score + generation time, so two claims are reproducible:
  1. fp16 (Lever C) speeds up the fp32 model materially.
  2. The max_new_tokens cap (Lever A) is accuracy-neutral on short-answer document QA.

This is the scripted version of the manual A/B checks done during development.

Usage:
  python scripts/efficiency_ablation.py --out /content/drive/MyDrive/MiniVLMDocEval/ablation
  python scripts/efficiency_ablation.py --out ./ablation --n 100
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_EVAL = REPO_ROOT / "scripts" / "run_eval.py"
MODEL = "SmolVLM2-500M"  # the only fp32 wrapper -> the model both levers affect most


def run_cfg(out, data, n, flags):
    cmd = [sys.executable, str(RUN_EVAL), "--out", str(out), "--models", MODEL,
           "--data", *data, "--n", str(n), *flags]
    print("  $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_scores(out, data, n):
    """{dataset: (value, seconds)} from the per-pair score.json files."""
    res = {}
    preds = Path(out) / "predictions" / MODEL
    for ds in data:
        f = preds / f"{ds}_n{n}_score.json"
        if f.exists():
            r = json.loads(f.read_text())
            res[ds] = (r.get("value"), r.get("seconds"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="ablation output base dir")
    ap.add_argument("--n", type=int, default=100, help="samples per dataset")
    args = ap.parse_args()
    base = Path(args.out)
    n = args.n

    # Ablation 1: fp16 vs fp32 (OCRBench) -- speed + score. Both cap at 128.
    print("\n[Ablation 1] fp16 vs fp32 -- SmolVLM2-500M x OCRBench")
    run_cfg(base / "fp32", ["OCRBench"], n, ["--max-new-tokens", "128"])
    run_cfg(base / "fp16", ["OCRBench"], n, ["--max-new-tokens", "128", "--fp16"])
    s32 = read_scores(base / "fp32", ["OCRBench"], n)
    s16 = read_scores(base / "fp16", ["OCRBench"], n)

    # Ablation 2: max_new_tokens 128 vs 2048 (DocVQA, InfoVQA) -- accuracy-neutrality. fp16 fixed.
    d2 = ["DocVQA_VAL", "InfoVQA_VAL"]
    print("\n[Ablation 2] max_new_tokens 128 vs 2048 -- SmolVLM2-500M x DocVQA,InfoVQA (fp16)")
    run_cfg(base / "cap128", d2, n, ["--fp16", "--max-new-tokens", "128"])
    run_cfg(base / "cap2048", d2, n, ["--fp16", "--max-new-tokens", "0"])
    c128 = read_scores(base / "cap128", d2, n)
    c2048 = read_scores(base / "cap2048", d2, n)

    # ---- report ----
    print(f"\n================ EFFICIENCY ABLATIONS (n={n}) ================")

    print("\n## Ablation 1 -- fp16 vs fp32 (OCRBench)")
    print(f"{'config':8} {'score(%)':>10} {'gen_time(s)':>12}")
    for tag, s in (("fp32", s32), ("fp16", s16)):
        v, sec = s.get("OCRBench", (None, None))
        print(f"{tag:8} {v:>10.2f} {sec:>12.1f}" if v is not None else f"{tag:8}  (missing)")
    if s32.get("OCRBench") and s16.get("OCRBench") and s16["OCRBench"][1]:
        print(f"-> fp16 speedup: {s32['OCRBench'][1] / s16['OCRBench'][1]:.2f}x  "
              f"(score delta: {s16['OCRBench'][0] - s32['OCRBench'][0]:+.2f} pts)")

    print("\n## Ablation 2 -- max_new_tokens 128 vs 2048 (accuracy-neutrality)")
    print(f"{'dataset':14} {'cap128':>10} {'cap2048':>10} {'delta':>8}")
    for ds in d2:
        v1 = c128.get(ds, (None,))[0]
        v2 = c2048.get(ds, (None,))[0]
        if v1 is not None and v2 is not None:
            print(f"{ds:14} {v1:>10.3f} {v2:>10.3f} {v1 - v2:>8.3f}")
    print("(identical scores => the 128 cap truncates nothing -> accuracy-neutral)")


if __name__ == "__main__":
    main()

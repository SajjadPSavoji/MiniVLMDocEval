#!/usr/bin/env python
"""Report TableVQABench accuracy broken down by its four sub-domains.

The eval pipeline (run_eval.py) collapses TableVQABench to a single mean in the
summary table, but VLMEvalKit already writes the per-sub-domain numbers to the
intermediate `predictions/<model>/TableVQABench_n{N}_acc.csv` file (columns
`split`, `average_scores`). This script just reads those and prints them — no GPU,
no model load, no training deps — so it works on already-completed runs.

Sub-domains: vwtq / vwtq_syn (Wikipedia-style visual table lookup),
vtabfact (true/false fact verification), fintabnetqa (financial tables; its
average_scores carries [relaxed, exact]).

The headline "mean" is reconciled to run_eval.extract_primary: average each
split's score list, then average across the four splits.

Usage:
  # baseline diagnosis on the synced local mirror
  python scripts/tablevqa_subdomain_report.py --models Qwen3.5-0.8B
  # before/after comparison
  python scripts/tablevqa_subdomain_report.py --models Qwen3.5-0.8B Qwen3.5-0.8B-TableLoRA
  # point at a different results tree (e.g. Drive on Colab)
  python scripts/tablevqa_subdomain_report.py --out $OUT_DIR --models Qwen3.5-0.8B
"""
import argparse
import ast
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from minivlmdoceval.config import PREDICTIONS_SUBDIR

# Canonical sub-domain order (weakest-first, per our baseline diagnosis).
SUBDOMAINS = ["vwtq", "vwtq_syn", "vtabfact", "fintabnetqa"]


def find_acc_csv(out_dir, model):
    """Locate predictions/<model>/TableVQABench_n*_acc.csv (newest if several)."""
    mdir = Path(out_dir) / PREDICTIONS_SUBDIR / model
    hits = sorted(mdir.glob("TableVQABench_n*_acc.csv"))
    return hits[-1] if hits else None


def read_subdomains(acc_csv):
    """Return {split: [floats]} from a TableVQABench *_acc.csv."""
    scores = {}
    with open(acc_csv, newline="") as f:
        for row in csv.DictReader(f):
            split = (row.get("split") or "").strip()
            raw = (row.get("average_scores") or "").strip()
            try:
                vals = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                continue
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            scores[split] = [float(v) for v in vals]
    return scores


def headline_mean(scores):
    """Reconcile to run_eval.extract_primary: mean of per-split list-means."""
    per_split = [sum(v) / len(v) for v in scores.values() if v]
    return sum(per_split) / len(per_split) if per_split else float("nan")


def split_value(scores, split):
    """Single comparable number per split: list-mean (matches the headline calc)."""
    vals = scores.get(split)
    return sum(vals) / len(vals) if vals else None


def fmt(x):
    return "  --  " if x is None else f"{x:6.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="drive_sync",
                    help="results tree containing predictions/ (default: drive_sync; use $OUT_DIR on Colab)")
    ap.add_argument("--models", nargs="+", required=True,
                    help="model keys to report; pass two for a before/after diff")
    args = ap.parse_args()

    loaded = {}
    for model in args.models:
        acc = find_acc_csv(args.out, model)
        if acc is None:
            print(f"[warn] no TableVQABench *_acc.csv for {model!r} under {args.out}/{PREDICTIONS_SUBDIR}/")
            continue
        loaded[model] = read_subdomains(acc)

    if not loaded:
        raise SystemExit("No sub-domain data found. Sync results first (bash scripts/sync_results.sh).")

    models = [m for m in args.models if m in loaded]
    width = 13
    header = f"{'sub-domain':<14}" + "".join(f"{m[:width]:>{width}}" for m in models)
    if len(models) == 2:
        header += f"{'Δ':>9}"
    print("\n=== TableVQABench per-sub-domain (list-mean per split, 0–100) ===")
    print(header)
    print("-" * len(header))

    for split in SUBDOMAINS:
        line = f"{split:<14}"
        vals = []
        for m in models:
            v = split_value(loaded[m], split)
            vals.append(v)
            line += f"{fmt(v):>{width}}"
        if len(models) == 2 and None not in vals:
            line += f"{vals[1] - vals[0]:>+9.1f}"
        print(line)

    # headline mean row
    line = f"{'MEAN':<14}"
    means = []
    for m in models:
        mv = headline_mean(loaded[m])
        means.append(mv)
        line += f"{mv:>{width}.1f}"
    if len(models) == 2:
        line += f"{means[1] - means[0]:>+9.1f}"
    print("-" * len(header))
    print(line)

    # fintabnetqa carries [relaxed, exact] — show both so it isn't misread
    print("\nnote: fintabnetqa average_scores = [relaxed, exact]:")
    for m in models:
        raw = loaded[m].get("fintabnetqa")
        if raw and len(raw) == 2:
            print(f"  {m}: relaxed={raw[0]:.1f}  exact={raw[1]:.1f}")


if __name__ == "__main__":
    main()

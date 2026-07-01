#!/usr/bin/env python
"""Decide the Part-2 ship gate: baseline vs LoRA-tuned Qwen3.5-0.8B.

Pre-declared rule (research_plan.md §6.5):
  PASS iff  ΔTableVQA(mean) >= +3.0  AND  no protected benchmark drops by > 1.5.
We also print the per-sub-domain TableVQA deltas (the gain should land in vwtq /
vwtq_syn), since the ~1500-item mean is noisy.

Read-only: reads the *_score.json the eval pipeline already wrote for both model
keys. Run scripts/sync_results.sh first if comparing on the local mirror.

Usage:
  python scripts/regression_gate.py                       # drive_sync, default keys
  python scripts/regression_gate.py --out $OUT_DIR \
      --baseline Qwen3.5-0.8B --finetuned Qwen3.5-0.8B-TableLoRA
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from minivlmdoceval.config import DATASETS, PREDICTIONS_SUBDIR
from run_eval import aggregate
from tablevqa_subdomain_report import find_acc_csv, read_subdomains, split_value, SUBDOMAINS

PROTECTED = ["OCRBench", "DocVQA_VAL", "ChartQA_TEST", "InfoVQA_VAL"]
TABLE = "TableVQABench"


def values_by_model(out_dir):
    out = aggregate(Path(out_dir) / PREDICTIONS_SUBDIR)
    if out is None:
        raise SystemExit(f"No scores under {out_dir}/{PREDICTIONS_SUBDIR}/ — run eval / sync first.")
    df, _ = out
    table = {}
    for _, r in df.iterrows():
        table.setdefault(r["model"], {})[r["benchmark"]] = r["value"]
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="drive_sync")
    ap.add_argument("--baseline", default="Qwen3.5-0.8B")
    ap.add_argument("--finetuned", default="Qwen3.5-0.8B-TableLoRA")
    ap.add_argument("--table-gain", type=float, default=3.0, help="required ΔTableVQA to pass")
    ap.add_argument("--max-drop", type=float, default=1.5, help="max allowed drop on a protected benchmark")
    args = ap.parse_args()

    vals = values_by_model(args.out)
    for m in (args.baseline, args.finetuned):
        if m not in vals:
            raise SystemExit(f"No scores for model {m!r} under {args.out} (have: {sorted(vals)})")
    base, ft = vals[args.baseline], vals[args.finetuned]

    print(f"\n=== Part-2 gate: {args.finetuned} vs {args.baseline} ===")
    print(f"{'benchmark':<16}{'base':>8}{'tuned':>8}{'Δ':>8}   status")
    print("-" * 52)

    table_delta = None
    protected_ok = True
    for ds in DATASETS:
        b, t = base.get(ds), ft.get(ds)
        if b is None or t is None:
            print(f"{ds:<16}{'--':>8}{'--':>8}{'--':>8}   missing")
            continue
        d = t - b
        if ds == TABLE:
            table_delta = d
            status = "TARGET ↑" if d >= args.table_gain else "below gain"
        else:
            ok = d >= -args.max_drop
            protected_ok = protected_ok and ok
            status = "ok" if ok else "REGRESSION"
        print(f"{ds:<16}{b:>8.1f}{t:>8.1f}{d:>+8.1f}   {status}")

    # per-sub-domain TableVQA detail
    print("\n-- TableVQA sub-domains (Δ) --")
    sb = read_subdomains(find_acc_csv(args.out, args.baseline)) if find_acc_csv(args.out, args.baseline) else {}
    sf = read_subdomains(find_acc_csv(args.out, args.finetuned)) if find_acc_csv(args.out, args.finetuned) else {}
    for split in SUBDOMAINS:
        b, t = split_value(sb, split), split_value(sf, split)
        if b is None or t is None:
            print(f"  {split:<12}  (n/a)")
        else:
            print(f"  {split:<12}{b:>8.1f}{t:>8.1f}{t - b:>+8.1f}")

    table_pass = table_delta is not None and table_delta >= args.table_gain
    verdict = "PASS" if (table_pass and protected_ok) else "FAIL"
    print("\n" + "=" * 52)
    print(f"ΔTableVQA = {table_delta:+.1f} (need ≥ +{args.table_gain})  |  "
          f"protected within -{args.max_drop}: {protected_ok}")
    print(f"GATE: {verdict}")
    if verdict == "FAIL":
        if not table_pass:
            print("  → table gain short: add an epoch or up-weight vwtq data.")
        if not protected_ok:
            print("  → regression: raise replay to ~35% and/or drop LR to 5e-5, re-run the dropped pairs (--no-reuse).")
    raise SystemExit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()

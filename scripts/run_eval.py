#!/usr/bin/env python
"""Run the built-in models on the FULL document benchmark suite via VLMEvalKit,
then aggregate per-(model, dataset) primary metrics into a comparison table.

- Inference + scoring is delegated to VLMEvalKit's run.py (handles resume via
  --reuse). Heavy predictions/status go to --work-dir; point it at Google Drive
  on Colab so runs survive session disconnects.
- The light comparison table is written to results/ (git-tracked) and refreshed
  after each model, so partial results are always available.

Usage:
  python scripts/run_eval.py                         # all built-ins x all datasets (full)
  python scripts/run_eval.py --models SmolVLM2-500M --data OCRBench
  python scripts/run_eval.py --work-dir /content/drive/MyDrive/MiniVLMDocEval/outputs
  python scripts/run_eval.py --aggregate-only        # just rebuild the table from existing outputs
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # make `minivlmdoceval` importable when run as a script
from minivlmdoceval.config import BUILTIN_MODELS, DATASETS, DEFAULT_WORK_DIR, SUMMARY_SUBDIR

RUNPY = REPO_ROOT / "external" / "VLMEvalKit" / "run.py"


def run_one_model(model, datasets, work_dir, reuse):
    cmd = [sys.executable, str(RUNPY), "--model", model, "--data", *datasets, "--work-dir", work_dir]
    if reuse:
        cmd.append("--reuse")
    print(f"\n=== RUN {model} | {datasets} ===\n{' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, check=True)


def aggregate(work_dir):
    """Scan work_dir for VLMEvalKit status.json files; build model x benchmark table."""
    from vlmeval.smp import collect_run_benchmark_report  # noqa: import here (needs vlmeval)
    import pandas as pd

    rows = []
    for status in Path(work_dir).rglob("status.json"):
        run_dir = status.parent
        model = run_dir.parent.name  # layout: work_dir/<model>/<eval_id>/status.json
        for r in collect_run_benchmark_report(run_dir):
            rows.append({
                "model": model,
                "benchmark": r.get("benchmark"),
                "metric": r.get("primary_metric"),
                "value": r.get("primary_metric_value"),
                "infer_failed": r.get("infer_failed"),
                "infer_total": r.get("infer_total"),
            })
    if not rows:
        return None
    df = pd.DataFrame(rows)
    pivot = df.pivot_table(index="model", columns="benchmark", values="value", aggfunc="first")
    return df, pivot


def write_results(work_dir):
    out = aggregate(work_dir)
    if out is None:
        print("No status.json found yet — nothing to aggregate.")
        return
    df, pivot = out
    rdir = Path(work_dir) / SUMMARY_SUBDIR  # summary lives under the (Drive) work-dir
    rdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(rdir / "scores_long.csv", index=False)
    pivot.to_csv(rdir / "comparison.csv")
    (rdir / "comparison.md").write_text(pivot.to_markdown())
    print("\n=== comparison (model x benchmark, primary metric) ===")
    print(pivot.to_string())
    print(f"\nwrote summary tables to {rdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=BUILTIN_MODELS)
    ap.add_argument("--data", nargs="+", default=DATASETS)
    ap.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    ap.add_argument("--no-reuse", action="store_true", help="disable VLMEvalKit --reuse (resume)")
    ap.add_argument("--aggregate-only", action="store_true", help="skip running; just rebuild the table")
    args = ap.parse_args()

    if not args.aggregate_only:
        for model in args.models:
            run_one_model(model, args.data, args.work_dir, reuse=not args.no_reuse)
            write_results(args.work_dir)  # refresh table after each model
    else:
        write_results(args.work_dir)


if __name__ == "__main__":
    main()

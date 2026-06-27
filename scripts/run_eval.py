#!/usr/bin/env python
"""Run the built-in models on the FULL document benchmark suite via VLMEvalKit,
capturing per-run logs and aggregating per-(model, dataset) primary metrics.

All artifacts go under a single output base (--out), so the code runs anywhere
(local, or a Google Drive path on Colab that persists across sessions):

  <out>/predictions/<model>/<eval_id>/...   VLMEvalKit work-dir (status.json, preds)
  <out>/summary/comparison.{csv,md}, scores_long.csv
  <out>/logs/<model>_<timestamp>.log        tee of each run (resume/debug)

The summary table is refreshed after each model. VLMEvalKit --reuse resumes from
<out>/predictions after a disconnect.

Usage:
  python scripts/run_eval.py --out /content/drive/MyDrive/MiniVLMDocEval/outputs
  python scripts/run_eval.py --models SmolVLM2-500M --data OCRBench --out outputs
  python scripts/run_eval.py --out outputs --aggregate-only
"""
import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # make `minivlmdoceval` importable when run as a script
from minivlmdoceval.config import (
    BUILTIN_MODELS, DATASETS, DEFAULT_OUT,
    PREDICTIONS_SUBDIR, SUMMARY_SUBDIR, LOGS_SUBDIR,
)

RUNPY = REPO_ROOT / "external" / "VLMEvalKit" / "run.py"


def _tee(cmd, log_path):
    """Run cmd, streaming merged stdout/stderr to the console AND a log file."""
    with open(log_path, "w") as logf:
        logf.write(f"# {dt.datetime.now().isoformat()}\n# {' '.join(map(str, cmd))}\n\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        proc.wait()
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_one_model(model, datasets, preds_dir, logs_dir, reuse):
    cmd = [sys.executable, str(RUNPY), "--model", model, "--data", *datasets,
           "--work-dir", str(preds_dir)]
    if reuse:
        cmd.append("--reuse")
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = logs_dir / f"{model}_{ts}.log"
    print(f"\n=== RUN {model} | {datasets} ===\nlog: {log_path}\n", flush=True)
    _tee(cmd, log_path)


def aggregate(preds_dir):
    """Scan preds_dir for VLMEvalKit status.json files; build model x benchmark table."""
    status_files = list(Path(preds_dir).rglob("status.json"))
    if not status_files:
        return None  # nothing to aggregate yet — avoid importing vlmeval

    from vlmeval.smp import collect_run_benchmark_report, load_run_status  # needs vlmeval (Colab)
    import pandas as pd

    # Read the model name from the status content (not the dir path): VLMEvalKit
    # writes status.json at more than one nesting level, so a path-based parent
    # name is unreliable (it can surface as the literal "predictions"). Dedupe by
    # (model, benchmark), preferring an entry that actually has a metric value.
    best = {}
    for status in status_files:
        run_dir = status.parent
        model = load_run_status(run_dir).get("model_name") or run_dir.parent.name
        for r in collect_run_benchmark_report(run_dir):
            key = (model, r.get("benchmark"))
            row = {
                "model": model,
                "benchmark": r.get("benchmark"),
                "metric": r.get("primary_metric"),
                "value": r.get("primary_metric_value"),
                "infer_failed": r.get("infer_failed"),
                "infer_total": r.get("infer_total"),
            }
            prev = best.get(key)
            if prev is None or (prev["value"] in (None, "") and row["value"] not in (None, "")):
                best[key] = row
    rows = list(best.values())
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["model", "benchmark"]).reset_index(drop=True)
    pivot = df.pivot_table(index="model", columns="benchmark", values="value", aggfunc="first")
    return df, pivot


def write_results(preds_dir, summary_dir):
    out = aggregate(preds_dir)
    if out is None:
        print("No status.json found yet — nothing to aggregate.")
        return
    df, pivot = out
    df.to_csv(summary_dir / "scores_long.csv", index=False)
    pivot.to_csv(summary_dir / "comparison.csv")
    (summary_dir / "comparison.md").write_text(pivot.to_markdown())
    print("\n=== comparison (model x benchmark, primary metric) ===")
    print(pivot.to_string())
    print(f"\nwrote summary tables to {summary_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT, help="output base dir (point at Drive on Colab)")
    ap.add_argument("--models", nargs="+", default=BUILTIN_MODELS)
    ap.add_argument("--data", nargs="+", default=DATASETS)
    ap.add_argument("--no-reuse", action="store_true", help="disable VLMEvalKit --reuse (resume)")
    ap.add_argument("--aggregate-only", action="store_true", help="skip running; just rebuild the table")
    args = ap.parse_args()

    out = Path(args.out)
    preds_dir = out / PREDICTIONS_SUBDIR
    summary_dir = out / SUMMARY_SUBDIR
    logs_dir = out / LOGS_SUBDIR
    for d in (preds_dir, summary_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"output tree: {out}/ {{{PREDICTIONS_SUBDIR}, {SUMMARY_SUBDIR}, {LOGS_SUBDIR}}}")

    if not args.aggregate_only:
        for model in args.models:
            run_one_model(model, args.data, preds_dir, logs_dir, reuse=not args.no_reuse)
            write_results(preds_dir, summary_dir)  # refresh table after each model
    else:
        write_results(preds_dir, summary_dir)


if __name__ == "__main__":
    main()

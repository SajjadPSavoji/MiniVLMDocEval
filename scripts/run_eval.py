#!/usr/bin/env python
"""Run the built-in models on a CAPPED subset (N samples/dataset) of the document
benchmark suite, with per-pair resume and logs, storing everything under --out.

Why a subset loop instead of VLMEvalKit's run.py: run.py has no sample limit, and
truncating the cached dataset TSV triggers an md5 re-download. So we build each
dataset, take a fixed-seed N-row sample, run inference, and call dataset.evaluate
on that subset — reusing VLMEvalKit's models, datasets, and scorers on N rows.

Tree under --out (point at Google Drive on Colab so it persists):
  predictions/<model>/<dataset>_n{N}.xlsx        predictions
  predictions/<model>/<dataset>_n{N}_score.json  per-pair score (resume marker)
  summary/comparison.{csv,md}, scores_long.csv
  logs/<model>_<timestamp>.log                   per-model run log

Resume is per (model, dataset, N) pair: a pair with a *_score.json is skipped.
The summary table is refreshed after each pair.

Usage:
  python scripts/run_eval.py --out <dir>                       # all built-ins, N=1000
  python scripts/run_eval.py --out <dir> --n 1000 --models SmolVLM2-500M --data OCRBench
  python scripts/run_eval.py --out <dir> --aggregate-only
"""
import argparse
import datetime as dt
import gc
import json
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # make `minivlmdoceval` importable when run as a script
from minivlmdoceval.config import (
    BUILTIN_MODELS, DATASETS, DEFAULT_OUT, DEFAULT_N, SAMPLE_SEED, DEFAULT_MAX_NEW_TOKENS,
    PREDICTIONS_SUBDIR, SUMMARY_SUBDIR, LOGS_SUBDIR,
)


NO_ANS = "<no_answer>"  # placeholder for empty/None predictions (scores as wrong; avoids NaN in xlsx)


def clean_preds(values):
    """Coerce predictions to non-empty strings. Empty strings round-trip to NaN
    in xlsx, which breaks some VLMEvalKit scorers (e.g. TableVQABench .lower())."""
    return [v if (isinstance(v, str) and v.strip()) else NO_ANS for v in values]


def log(msg, logf=None):
    print(msg, flush=True)
    if logf is not None:
        logf.write(msg + "\n")
        logf.flush()


def subset_data(data, n, seed):
    """Fixed-seed N-row sample (full data if it already has <= N rows)."""
    if len(data) <= n:
        return data.reset_index(drop=True)
    return data.sample(n=n, random_state=seed).sort_index().reset_index(drop=True)


def build_struct(model, dataset, dataset_name, line):
    """Replicate vlmeval/inference.py prompt construction."""
    if getattr(dataset, "force_use_dataset_prompt", False):
        return dataset.build_prompt(line)
    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
        return model.build_prompt(line, dataset=dataset_name)
    return dataset.build_prompt(line)


def extract_primary(res, n):
    """Pull a single headline metric on a 0-100 (accuracy %) scale out of a
    dataset.evaluate() result. `n` = number of samples evaluated.

    Note: OCRBench's 'Final Score' is a raw count of correct answers, and its
    'Final Score Norm' (=Final Score/10) is hardcoded for the full 1000-question
    set — wrong for subsets. So we normalize by the actual N: correct/N*100.
    ImageVQA's 'Overall' and TableVQA's 'average_scores' are already per-subset.
    """
    import pandas as pd
    if isinstance(res, dict) and isinstance(res.get("Final Score"), (int, float)):  # OCRBench
        return "acc(%)", float(res["Final Score"]) / max(n, 1) * 100
    if isinstance(res, pd.DataFrame):
        if "Overall" in res.columns and len(res):
            return "Overall(%)", float(res["Overall"].iloc[0])
        if "average_scores" in res.columns and len(res):   # TableVQABench: cells are lists like [acc]
            per_split = []
            for cell in res["average_scores"]:
                if isinstance(cell, (list, tuple)):
                    nums = [v for v in cell if isinstance(v, (int, float))]
                elif isinstance(cell, (int, float)):
                    nums = [cell]
                else:
                    nums = []
                if nums:
                    per_split.append(sum(nums) / len(nums))
            if per_split:
                m = sum(per_split) / len(per_split)  # mean across the table sub-domains
                return "acc(%)", m * 100 if m <= 1 else m
        if len(res):
            row = res.iloc[0].to_dict()
            for k in ("Overall", "Final Score", "score", "acc"):
                if isinstance(row.get(k), (int, float)):
                    return k, float(row[k])
        return None, None
    if isinstance(res, dict):
        if isinstance(res.get("Overall"), (int, float)):
            return "Overall(%)", float(res["Overall"])
        if isinstance(res.get("average_scores"), list):             # TableVQABench
            vals = [v for v in res["average_scores"] if isinstance(v, (int, float))]
            if vals:
                m = sum(vals) / len(vals)
                return "acc(%)", m * 100 if m <= 1 else m
        for k, v in res.items():
            if isinstance(v, (int, float)):
                return k, float(v)
    return None, None


def apply_gen_cap(model, max_new_tokens, logf):
    """Lever A: cap generation length on the loaded model (wrappers default 2048).
    All built-in wrappers store gen config in self.kwargs['max_new_tokens']."""
    if max_new_tokens is None:
        return
    kw = getattr(model, "kwargs", None)
    if isinstance(kw, dict):
        old = kw.get("max_new_tokens")
        kw["max_new_tokens"] = max_new_tokens
        log(f"  max_new_tokens: {old} -> {max_new_tokens}", logf)
    else:
        log(f"  [warn] {type(model).__name__} has no .kwargs dict; max_new_tokens not applied", logf)


def apply_fp16(model, enabled, logf):
    """Lever C (selective): downcast fp32 models to fp16. bf16/fp16 models are left
    at their native precision — avoids bf16->fp16 instability (e.g. InternVL). With
    our 3 built-ins this converts only SmolVLM (the lone fp32 wrapper)."""
    if not enabled:
        return
    import torch
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "parameters"):
        log("  [fp16] no .model module to convert; skipped", logf)
        return
    try:
        dtype = next(inner.parameters()).dtype
    except StopIteration:
        return
    if dtype == torch.float32:
        inner.half()
        log(f"  [fp16] {type(model).__name__}: fp32 -> fp16", logf)
    else:
        log(f"  [fp16] {type(model).__name__}: left at {dtype} (already 16-bit)", logf)


def run_pair(model, model_name, dataset_name, n, preds_dir, logf):
    from vlmeval.dataset import build_dataset
    from vlmeval.smp import dump

    mdir = preds_dir / model_name
    mdir.mkdir(parents=True, exist_ok=True)
    pred_file = mdir / f"{dataset_name}_n{n}.xlsx"
    score_file = mdir / f"{dataset_name}_n{n}_score.json"

    log(f"\n--- {model_name} | {dataset_name} (n={n}) ---", logf)
    dataset = build_dataset(dataset_name)
    if hasattr(model, "set_dump_image"):
        model.set_dump_image(dataset.dump_image)
    data = subset_data(dataset.data, n, SAMPLE_SEED)
    log(f"running {len(data)} samples", logf)

    # Lever B: no per-sample torch.cuda.empty_cache() — it forces a sync + cache
    # churn every iteration for no benefit at batch-1 with these tiny models.
    preds = []
    t0 = dt.datetime.now()
    for i in range(len(data)):
        struct = build_struct(model, dataset, dataset_name, data.iloc[i])
        preds.append(model.generate(message=struct, dataset=dataset_name))
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(data)}  ({(dt.datetime.now() - t0).total_seconds():.0f}s)", logf)

    gen_seconds = (dt.datetime.now() - t0).total_seconds()  # generation-only wall time
    sub = data.copy()
    sub["prediction"] = clean_preds(preds)
    dump(sub, str(pred_file))
    res = dataset.evaluate(str(pred_file))
    metric, value = extract_primary(res, len(data))
    rec = {"model": model_name, "benchmark": dataset_name, "n": int(len(data)),
           "metric": metric, "value": value,
           "seconds": round(gen_seconds, 1),
           "s_per_sample": round(gen_seconds / max(len(data), 1), 3)}
    score_file.write_text(json.dumps(rec, indent=2, default=str))
    log(f"  -> {metric} = {value}", logf)
    return rec


def aggregate(preds_dir):
    score_files = list(Path(preds_dir).rglob("*_score.json"))
    if not score_files:
        return None  # nothing yet — avoid importing pandas

    import pandas as pd
    recs = []
    for sf in score_files:
        try:
            recs.append(json.loads(sf.read_text()))
        except Exception:
            pass
    if not recs:
        return None
    df = pd.DataFrame(recs).sort_values(["model", "benchmark"]).reset_index(drop=True)
    pivot = df.pivot_table(index="model", columns="benchmark", values="value", aggfunc="first")
    return df, pivot


def rescore(preds_dir, summary_dir):
    """Re-evaluate saved prediction files and rewrite score.json — no inference.
    Use after a scoring fix. Preserves the recorded timing."""
    import re
    from vlmeval.dataset import build_dataset
    from vlmeval.smp import load, dump

    pat = re.compile(r"^(.*)_n(\d+)\.xlsx$")
    ds_cache = {}
    for pred in sorted(Path(preds_dir).rglob("*_n*.xlsx")):
        m = pat.match(pred.name)
        if not m:
            continue
        ds_name, n = m.group(1), int(m.group(2))
        model = pred.parent.name
        score_file = pred.with_name(f"{pred.stem}_score.json")
        if ds_name not in ds_cache:
            ds_cache[ds_name] = build_dataset(ds_name)
        # repair empty/NaN predictions in-place (older files written before clean_preds)
        df = load(str(pred))
        if "prediction" in df.columns:
            fixed = clean_preds(list(df["prediction"]))
            if fixed != list(df["prediction"]):
                df["prediction"] = fixed
                dump(df, str(pred))
        try:
            metric, value = extract_primary(ds_cache[ds_name].evaluate(str(pred)), n)
        except Exception:
            print(f"  rescore FAIL {model} | {ds_name}\n{traceback.format_exc()}")
            continue
        rec = {"model": model, "benchmark": ds_name, "n": n, "metric": metric, "value": value}
        if score_file.exists():  # preserve timing
            try:
                old = json.loads(score_file.read_text())
                rec["seconds"] = old.get("seconds")
                rec["s_per_sample"] = old.get("s_per_sample")
            except Exception:
                pass
        score_file.write_text(json.dumps(rec, indent=2, default=str))
        print(f"  rescored {model} | {ds_name}: {metric} = {value}")
    write_results(preds_dir, summary_dir)


def pair_complete(score_file):
    """A pair is 'done' only if it has BOTH a score value and a runtime, so a
    re-run backfills pairs that are missing either (e.g. timing lost in a rescore)."""
    if not score_file.exists():
        return False
    try:
        r = json.loads(score_file.read_text())
    except Exception:
        return False
    return r.get("value") is not None and r.get("seconds") is not None


def write_results(preds_dir, summary_dir):
    out = aggregate(preds_dir)
    if out is None:
        print("No per-pair scores yet — nothing to aggregate.")
        return
    df, pivot = out
    df.to_csv(summary_dir / "scores_long.csv", index=False)
    pivot.to_csv(summary_dir / "comparison.csv")
    (summary_dir / "comparison.md").write_text(pivot.to_markdown())
    print("\n=== comparison (model x benchmark, primary metric, 0-100) ===")
    print(pivot.to_string())
    print(f"\nwrote summary tables to {summary_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT, help="output base dir (point at Drive on Colab)")
    ap.add_argument("--models", nargs="+", default=BUILTIN_MODELS)
    ap.add_argument("--data", nargs="+", default=DATASETS)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="samples per dataset")
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
                    help="cap generation length (Lever A); set 0 to leave the wrapper default")
    ap.add_argument("--fp16", action="store_true",
                    help="downcast fp32 models to fp16 (Lever C; auto-targets SmolVLM, leaves bf16/fp16 as-is)")
    ap.add_argument("--no-reuse", action="store_true", help="recompute pairs even if a score exists")
    ap.add_argument("--aggregate-only", action="store_true", help="skip running; just rebuild the table")
    ap.add_argument("--rescore", action="store_true",
                    help="re-evaluate saved prediction files (no inference) and rebuild the table")
    args = ap.parse_args()

    out = Path(args.out)
    preds_dir = out / PREDICTIONS_SUBDIR
    summary_dir = out / SUMMARY_SUBDIR
    logs_dir = out / LOGS_SUBDIR
    for d in (preds_dir, summary_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)
    print(f"output tree: {out}/ {{{PREDICTIONS_SUBDIR}, {SUMMARY_SUBDIR}, {LOGS_SUBDIR}}}  N={args.n}")

    if args.aggregate_only:
        write_results(preds_dir, summary_dir)
        return
    if args.rescore:
        rescore(preds_dir, summary_dir)
        return

    from vlmeval.config import supported_VLM
    import torch

    for model_name in args.models:
        ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        with open(logs_dir / f"{model_name}_{ts}.log", "w") as logf:
            if model_name not in supported_VLM:
                log(f"SKIP {model_name}: not in VLMEvalKit registry", logf)
                continue
            log(f"=== loading {model_name} ===", logf)
            try:
                model = supported_VLM[model_name]()
            except Exception:
                log(f"FAIL load {model_name}\n{traceback.format_exc()}", logf)
                continue
            apply_gen_cap(model, args.max_new_tokens or None, logf)  # Lever A
            apply_fp16(model, args.fp16, logf)                       # Lever C (selective)

            for dataset_name in args.data:
                score_file = preds_dir / model_name / f"{dataset_name}_n{args.n}_score.json"
                if pair_complete(score_file) and not args.no_reuse:
                    log(f"reuse {model_name} | {dataset_name} (complete)", logf)
                    continue
                try:
                    run_pair(model, model_name, dataset_name, args.n, preds_dir, logf)
                    write_results(preds_dir, summary_dir)  # refresh after each pair
                except Exception:
                    log(f"FAIL {model_name} | {dataset_name}\n{traceback.format_exc()}", logf)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    write_results(preds_dir, summary_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""End-to-end smoke test for the VLMEvalKit eval path on small subsets.

By default this runs every built-in model from minivlmdoceval.config across every
configured dataset, taking 10 rows from each dataset. It mirrors
vlmeval/inference.py prompt construction, prints predictions, and optionally
scores each subset. Purpose: confirm the kit runs end-to-end on Colab GPU before
committing to a full run. Device-agnostic so wrapper plumbing can also be checked
on mac.

Note: VLMEvalKit's built-in model wrappers hardcode CUDA, so for those this
script is Colab/GPU-only. Our own custom wrappers (FastVLM, Qwen3.5) will be
written device-agnostic and can be smoke-tested on mac (mps/cpu).

Usage:
  python scripts/smoke_test.py --score
  python scripts/smoke_test.py --model SmolVLM2-500M --data OCRBench --n 5 --score
"""
import argparse
import gc
import os
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from minivlmdoceval.config import BUILTIN_MODELS, DATASETS


def build_struct(model, dataset, dataset_name, line):
    """Replicate vlmeval/inference.py prompt-construction logic exactly."""
    if getattr(dataset, "force_use_dataset_prompt", False):
        return dataset.build_prompt(line)
    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
        return model.build_prompt(line, dataset=dataset_name)
    return dataset.build_prompt(line)


def subset_data(data, n):
    """Take a small subset, preserving split coverage when a benchmark has splits."""
    if "split" not in data.columns or data["split"].nunique(dropna=False) <= 1:
        return data.head(n).reset_index(drop=True)

    groups = list(data.groupby("split", sort=False, dropna=False))
    per_split = n // len(groups)
    remainder = n % len(groups)
    parts = []
    for idx, (_, group) in enumerate(groups):
        take = per_split + (1 if idx < remainder else 0)
        if take:
            parts.append(group.head(take))

    subset = pd.concat(parts) if parts else data.head(0)
    if len(subset) < n:
        fill = data.loc[~data.index.isin(subset.index)].head(n - len(subset))
        subset = pd.concat([subset, fill])
    return subset.head(n).reset_index(drop=True)


def describe_splits(data):
    if "split" not in data.columns:
        return ""
    counts = data["split"].value_counts(dropna=False).sort_index()
    return " | splits=" + ", ".join(f"{split}:{count}" for split, count in counts.items())


def run_one_dataset(model, model_name, dataset_name, n, score, show_tracebacks):
    from vlmeval.dataset import build_dataset

    print(f"\n--- dataset: {dataset_name} | n={n} ---")
    print(f"[2/3] building dataset: {dataset_name}")
    dataset = build_dataset(dataset_name)
    if hasattr(model, "set_dump_image"):
        model.set_dump_image(dataset.dump_image)

    data = subset_data(dataset.data, n)
    print(f"[3/3] running inference on {len(data)} samples{describe_splits(data)}\n")

    preds = []
    for i in range(len(data)):
        line = data.iloc[i]
        struct = build_struct(model, dataset, dataset_name, line)
        resp = model.generate(message=struct, dataset=dataset_name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        q = str(line.get("question", ""))[:70]
        print(f"  [{i}] Q={q!r}\n      PRED={resp!r}")
        preds.append(resp)

    print(f"\nSMOKE_OK: {model_name} | {dataset_name} | {len(data)} samples")

    if score:
        try:
            from vlmeval.smp import dump

            sub = data.copy()
            sub["prediction"] = preds
            tmp = os.path.join(tempfile.gettempdir(), f"smoke_{model_name}_{dataset_name}.xlsx")
            dump(sub, tmp)
            print("\n[score] running dataset.evaluate on the subset (indicative only):")
            res = dataset.evaluate(tmp)
            print(res)
        except Exception:
            print("\n[score] scoring step failed (non-fatal for the smoke test):")
            print(traceback.format_exc(limit=1).strip())
            if show_tracebacks:
                traceback.print_exc()

    return len(data)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="single VLMEvalKit model key to run")
    ap.add_argument("--models", nargs="+", help="model keys to run (default: all built-in models)")
    ap.add_argument("--data", nargs="+", help="dataset names to run (default: all configured datasets)")
    ap.add_argument("--n", type=int, default=10, help="number of samples per dataset")
    ap.add_argument("--score", action="store_true", help="also run dataset.evaluate on each subset (indicative only)")
    ap.add_argument("--fail-fast", action="store_true", help="stop on the first model/dataset failure")
    ap.add_argument("--tracebacks", action="store_true", help="print full tracebacks for failures")
    args = ap.parse_args()
    if args.model and args.models:
        ap.error("use either --model or --models, not both")
    return args


def main():
    args = parse_args()

    from vlmeval.config import supported_VLM
    from minivlmdoceval.custom_models import register_custom_models
    register_custom_models()  # make FastVLM / Qwen3.5 / LFM2.5-VL smoke-testable

    models = args.models or ([args.model] if args.model else BUILTIN_MODELS)
    datasets = args.data or DATASETS

    print(f"[env] cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()}")
    print(f"[plan] models={models}")
    print(f"[plan] datasets={datasets}")
    print(f"[plan] samples_per_dataset={args.n} score={args.score}\n")

    failures = []
    completed = []
    for model_name in models:
        if model_name not in supported_VLM:
            msg = f"{model_name!r} not in VLMEvalKit registry"
            print(f"\nSMOKE_FAIL: {msg}")
            failures.append((model_name, "*", msg))
            if args.fail_fast:
                break
            continue

        print(f"\n=== model: {model_name} ===")
        print(f"[1/3] loading model: {model_name}")
        try:
            model = supported_VLM[model_name]()
        except Exception as exc:
            print(f"\nSMOKE_FAIL: {model_name} failed to load")
            if args.tracebacks:
                traceback.print_exc()
            else:
                print(f"{type(exc).__name__}: {exc}")
            failures.append((model_name, "*", repr(exc)))
            if args.fail_fast:
                break
            continue

        for dataset_name in datasets:
            try:
                n_ran = run_one_dataset(model, model_name, dataset_name, args.n, args.score, args.tracebacks)
                completed.append((model_name, dataset_name, n_ran))
            except Exception as exc:
                print(f"\nSMOKE_FAIL: {model_name} | {dataset_name}")
                if args.tracebacks:
                    traceback.print_exc()
                else:
                    print(f"{type(exc).__name__}: {exc}")
                failures.append((model_name, dataset_name, repr(exc)))
                if args.fail_fast:
                    break

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        if failures and args.fail_fast:
            break

    print("\n=== smoke summary ===")
    for model_name, dataset_name, n_ran in completed:
        print(f"OK   {model_name} | {dataset_name} | {n_ran} samples")

    if failures:
        print("\nFailures:")
        for model_name, dataset_name, err in failures:
            print(f"FAIL {model_name} | {dataset_name} | {err}")
        raise SystemExit(1)

    print("\nSMOKE_OK: all requested model/dataset subsets ran end-to-end.")


if __name__ == "__main__":
    main()

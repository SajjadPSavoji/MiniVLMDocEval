#!/usr/bin/env python
"""End-to-end smoke test for the VLMEvalKit eval path on a few samples.

Loads ONE model, builds ONE dataset, runs inference on the first N rows
(mirroring vlmeval/inference.py), prints predictions, and optionally scores the
subset. Purpose: confirm the kit runs end-to-end on Colab GPU before committing
to a full run. Device-agnostic so wrapper plumbing can also be checked on mac.

Note: VLMEvalKit's built-in model wrappers hardcode CUDA, so for those this
script is Colab/GPU-only. Our own custom wrappers (FastVLM, Qwen3.5) will be
written device-agnostic and can be smoke-tested on mac (mps/cpu).

Usage:
  python scripts/smoke_test.py --model SmolVLM2-500M --data OCRBench --n 5 --score
"""
import argparse
import os
import tempfile
import traceback

import torch


def build_struct(model, dataset, dataset_name, line):
    """Replicate vlmeval/inference.py prompt-construction logic exactly."""
    if getattr(dataset, "force_use_dataset_prompt", False):
        return dataset.build_prompt(line)
    if hasattr(model, "use_custom_prompt") and model.use_custom_prompt(dataset_name):
        return model.build_prompt(line, dataset=dataset_name)
    return dataset.build_prompt(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="SmolVLM2-500M", help="key in vlmeval supported_VLM registry")
    ap.add_argument("--data", default="OCRBench", help="VLMEvalKit dataset name")
    ap.add_argument("--n", type=int, default=5, help="number of samples to run")
    ap.add_argument("--score", action="store_true", help="also run dataset.evaluate on the subset (indicative only)")
    args = ap.parse_args()

    from vlmeval.config import supported_VLM
    from vlmeval.dataset import build_dataset

    print(f"[env] cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()}")
    assert args.model in supported_VLM, f"{args.model!r} not in registry"

    print(f"[1/3] loading model: {args.model}")
    model = supported_VLM[args.model]()

    print(f"[2/3] building dataset: {args.data}")
    dataset = build_dataset(args.data)
    if hasattr(model, "set_dump_image"):
        model.set_dump_image(dataset.dump_image)

    data = dataset.data.head(args.n).reset_index(drop=True)
    print(f"[3/3] running inference on {len(data)} samples\n")

    preds = []
    for i in range(len(data)):
        line = data.iloc[i]
        struct = build_struct(model, dataset, args.data, line)
        resp = model.generate(message=struct, dataset=args.data)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        q = str(line.get("question", ""))[:70]
        print(f"  [{i}] Q={q!r}\n      PRED={resp!r}")
        preds.append(resp)

    print("\nSMOKE_OK: model loaded, dataset built, inference ran end-to-end.")

    if args.score:
        try:
            from vlmeval.smp import dump
            sub = data.copy()
            sub["prediction"] = preds
            tmp = os.path.join(tempfile.gettempdir(), f"smoke_{args.model}_{args.data}.xlsx")
            dump(sub, tmp)
            print("\n[score] running dataset.evaluate on the subset (indicative only):")
            res = dataset.evaluate(tmp)
            print(res)
        except Exception:
            print("\n[score] scoring step failed (non-fatal for the smoke test):")
            traceback.print_exc()


if __name__ == "__main__":
    main()

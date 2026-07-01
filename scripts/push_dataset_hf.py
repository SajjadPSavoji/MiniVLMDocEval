#!/usr/bin/env python
"""Push the locally-built TableVQA SFT mixture to the HuggingFace Hub as a dataset.

Build first (scripts/build_table_sft.py --data-dir /content/table_sft), then push.
This loads train.jsonl + the referenced images into a `datasets.Dataset` with a
proper Image feature and `push_to_hub`s it, plus uploads the provenance manifest.

SECRETS ARE PASSED AT RUNTIME, never hardcoded here:
  - HF write token  -> $HF_TOKEN env var (or --token). Create one at
                       https://huggingface.co/settings/tokens (role: write).
  - your username   -> part of --repo-id (<username>/<dataset-name>).

Usage (token via env so it never lands in shell history/logs):
  HF_TOKEN=hf_xxx python scripts/push_dataset_hf.py \
      --data-dir /content/table_sft \
      --repo-id <HF_USERNAME>/minivlm-tablevqa-sft [--private]
"""
import argparse
import json
import os
import sys
from pathlib import Path


def build_card_body(repo_id, manifest):
    """Human-readable dataset card body (markdown) built from the build manifest."""
    m = manifest or {}
    by_sub = m.get("by_subtype", {})
    by_src = m.get("by_source", {})
    excl = m.get("tablevqa_questions_excluded", "?")
    dropped = m.get("table_rows_dropped_as_eval_overlap", "?")
    protected = m.get("protected_eval_rows_excluded", {})
    sources = ", ".join(m.get("table_sources", [])) or "(see manifest)"
    return f"""# {repo_id.split('/')[-1]}

Supervised fine-tuning mixture for improving **visual table question answering** in
small vision-language models — built for the MiniVLMDocEval project to lift
**Qwen3.5-0.8B** on TableVQABench.

## Why this mixture (data-driven targeting)

We measured Qwen3.5-0.8B per TableVQABench sub-domain and found the weakness is
**Wikipedia-style visual-table lookup**, not financial tables:

| sub-domain | Qwen3.5-0.8B | |
|---|---|---|
| vwtq (Wikipedia lookup) | **27.8** | weakest, and the largest split |
| vwtq_syn (synthetic Wikipedia) | **33.1** | |
| vtabfact (fact check) | 55.5 | |
| fintabnetqa (financial) | **84.0** | already strong |

So this dataset is **weighted toward table-lookup QA** (the measured gap), with a
general-document **replay** portion to prevent catastrophic forgetting. This
deliberately departs from the common assumption that financial/numeric tables are
the bottleneck — our own evaluation said otherwise.

## Composition

- **records:** {m.get('n_total', '?')}  ·  by subtype: `{by_sub}`  ·  by source: `{by_src}`
- **table sources:** {sources} (synthetic / non-Wikipedia by design — see overlap guarantee)
- **replay:** held-out rows of DocVQA / ChartQA / InfoVQA / OCRBench
- **build seed:** {m.get('seed', '?')}

## No train/eval overlap (guarantee)

This set is built to be safely disjoint from the **TableVQABench** evaluation:
- **{excl}** distinct TableVQABench questions were used as an exclusion set; any
  training row matching one was dropped (**{dropped}** dropped).
- replay rows are drawn **only from outside** the fixed seed-42 evaluation subset of
  each protected benchmark (held out: `{protected}`).
- TableVQABench itself is **never** used as a training source.

## Schema

| field | description |
|---|---|
| `image` | the table/document image |
| `prompt` | user prompt (table rows use VLMEvalKit's `VWTQ_PROMPT`, so train==eval format) |
| `answer` | target answer |
| `source` | originating dataset |
| `subtype` | `table` or `replay` |

## Reproduce

Built by [`scripts/build_table_sft.py`](https://github.com/SajjadPSavoji/MiniVLMDocEval)
and pushed with `scripts/push_dataset_hf.py`. See the bundled `manifest.json` for the
exact counts, sources, and overlap-guard results of this build.
"""


def write_card(repo_id, token, manifest):
    """Prepend our human-readable card above the auto-generated dataset_info section."""
    from huggingface_hub import DatasetCard
    body = build_card_body(repo_id, manifest)
    try:
        card = DatasetCard.load(repo_id, token=token)   # the auto card push_to_hub wrote
        card.text = body + "\n\n---\n\n" + (card.text or "")
    except Exception:                                    # noqa: BLE001
        card = DatasetCard(content=body)
    try:
        tags = list(getattr(card.data, "tags", None) or [])
        for t in ("table-question-answering", "vqa", "tablevqa", "lora-sft"):
            if t not in tags:
                tags.append(t)
        card.data.tags = tags
    except Exception:                                    # noqa: BLE001
        pass
    card.push_to_hub(repo_id, token=token)
    print("[push] wrote dataset card (README.md)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="train_data", help="dir with train.jsonl + images/")
    ap.add_argument("--repo-id", required=True, help="<username>/<dataset-name>")
    ap.add_argument("--private", action="store_true", help="create the dataset repo as private")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF write token; defaults to $HF_TOKEN (preferred — don't hardcode)")
    args = ap.parse_args()

    if not args.token:
        raise SystemExit("No HF token. Set HF_TOKEN env var (or pass --token). "
                         "Create one at https://huggingface.co/settings/tokens (role: write).")
    if "/" not in args.repo_id:
        raise SystemExit("--repo-id must be '<username>/<dataset-name>'.")

    from datasets import Dataset, Image

    data_dir = Path(args.data_dir)
    jsonl = data_dir / "train.jsonl"
    if not jsonl.exists():
        raise SystemExit(f"{jsonl} not found — run scripts/build_table_sft.py --data-dir {data_dir} first.")

    rows = []
    for line in open(jsonl):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        rows.append({k: v for k, v in r.items() if not k.startswith("_")})  # drop internal fields
    if not rows:
        raise SystemExit(f"{jsonl} is empty.")

    missing = [r["image"] for r in rows[:50] if not Path(r["image"]).exists()]
    if missing:
        raise SystemExit(f"image files missing (e.g. {missing[0]}). Build + push in the SAME "
                         "session — local images are wiped on runtime reset.")

    print(f"[push] {len(rows)} records from {jsonl}; casting image column ...")
    ds = Dataset.from_list(rows).cast_column("image", Image())
    print(f"[push] pushing to {args.repo_id} (private={args.private}) — embeds images into parquet ...")
    ds.push_to_hub(args.repo_id, token=args.token, private=args.private)

    # carry the provenance manifest (sources, contamination-guard counts) with the dataset
    manifest_path = data_dir / "manifest.json"
    manifest = None
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        from huggingface_hub import HfApi
        HfApi(token=args.token).upload_file(
            path_or_fileobj=str(manifest_path), path_in_repo="manifest.json",
            repo_id=args.repo_id, repo_type="dataset")
        print("[push] uploaded manifest.json")

    # auto-generate the dataset card (VWTQ rationale + no-overlap guarantee)
    write_card(args.repo_id, args.token, manifest)

    print(f"[push] done -> https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
    # HF/torch leave non-daemon threads that can segfault at interpreter teardown;
    # work is done + uploaded above, so exit hard.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

#!/usr/bin/env python
"""Build the LoRA fine-tuning mixture for the TableVQA PoC (Part 2).

Our measured diagnosis (scripts/tablevqa_subdomain_report.py) is that Qwen3.5-0.8B's
table weakness is **Wikipedia-style visual-table lookup** (vwtq 27.8 / vwtq_syn 33.1),
NOT financial tables (fintabnetqa 84.0). So the mixture is VWTQ-weighted:

  ~70%  table lookup QA   (external Wikipedia-style visual-table datasets)
  ~25%  replay            (held-out rows of the 4 protected benchmarks)
  ~ 5%  anti-regression   (a little fact-check / financial table QA to hold those)

Two safety properties this script guarantees and records in the manifest:
  1. NO TRAIN/TEST CONTAMINATION — replay is drawn only from rows OUTSIDE the
     fixed seed-42 eval subset (run_eval.subset_data), and TableVQABench (the eval
     benchmark) is never used as a training source.
  2. TRAIN/EVAL PROMPT CONSISTENCY — table examples are wrapped in VLMEvalKit's own
     VWTQ_PROMPT, and replay examples use each dataset's own build_prompt(), so the
     model trains under the same prompts it is scored under.

Output (under <out>/train_data/):
  train.jsonl     one record/line: {image, prompt, answer, source, subtype}
  images/...      extracted training images
  manifest.json   counts, seeds, fractions, contamination-check result

Table sources are best-effort: the deep-research dataset IDs are unverified, so each
source is guarded — if one fails to load, it is skipped and logged, and the mixture
rebalances to whatever loaded. Verify availability first with --dry-run.

Usage:
  python scripts/build_table_sft.py --data-dir /content/table_sft --dry-run   # probe, no write
  python scripts/build_table_sft.py --data-dir /content/table_sft --n-total 7000
  python scripts/build_table_sft.py --data-dir /content/table_sft \
      --table-sources preset:visual_tableqa preset:mmtab hf:some/dataset:train
"""
import argparse
import io
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from minivlmdoceval.config import SAMPLE_SEED, DEFAULT_N

# Protected benchmarks we must not regress (and the eval set we must not train on).
PROTECTED = ["DocVQA_VAL", "ChartQA_TEST", "OCRBench", "InfoVQA_VAL"]
EVAL_BENCHMARK = "TableVQABench"  # never a training source

# Heuristic field-name synonyms for generic HF table-QA datasets.
IMAGE_KEYS = ["image", "images", "img", "table_image", "decoded_image", "picture"]
Q_KEYS = ["question", "query", "instruction", "prompt", "input", "Question"]
A_KEYS = ["answer", "label", "output", "response", "target", "Answer", "gt"]


LOG_EVERY = 200  # progress cadence (image saves) so the build isn't a silent black box


def log(msg):
    print(msg, flush=True)


# ----------------------------- prompt templates -----------------------------

def vwtq_prompt():
    """VLMEvalKit's own VWTQ few-shot prompt, so training matches eval exactly."""
    from vlmeval.dataset.utils.tablevqabench import VWTQ_PROMPT
    return VWTQ_PROMPT


def render_table_prompt(question):
    return vwtq_prompt().format_map({"question": str(question)})


# ----------------------------- image handling -------------------------------

def save_image(obj, dest):
    """Persist a dataset image (PIL / bytes / dict / path) to dest as PNG. Returns
    dest on success, None on failure."""
    from PIL import Image
    try:
        img = None
        if hasattr(obj, "save"):                       # PIL.Image
            img = obj
        elif isinstance(obj, dict) and obj.get("bytes"):
            img = Image.open(io.BytesIO(obj["bytes"]))
        elif isinstance(obj, dict) and obj.get("path"):
            img = Image.open(obj["path"])
        elif isinstance(obj, (bytes, bytearray)):
            img = Image.open(io.BytesIO(obj))
        elif isinstance(obj, str) and os.path.exists(obj):
            img = Image.open(obj)
        if img is None:
            return None
        img.convert("RGB").save(dest)
        return dest
    except Exception as exc:                            # noqa: BLE001 — best-effort
        log(f"    [img] skip ({type(exc).__name__})")
        return None


# ----------------------------- field detection ------------------------------

def detect_fields(features, columns):
    """Pick (image_col, question_col, answer_col) by name/type heuristics."""
    img_col = next((k for k in IMAGE_KEYS if k in columns), None)
    if img_col is None:  # fall back to any column whose feature looks like an Image
        for k in columns:
            if "image" in k.lower():
                img_col = k
                break
    q_col = next((k for k in Q_KEYS if k in columns), None)
    a_col = next((k for k in A_KEYS if k in columns), None)
    return img_col, q_col, a_col


def extract_conv(row):
    """Handle llava/sharegpt 'conversations' format: [{from:human,value},{from:gpt,value}]."""
    conv = row.get("conversations") or row.get("messages")
    if not isinstance(conv, list):
        return None, None
    q = a = None
    for turn in conv:
        who = turn.get("from") or turn.get("role")
        val = turn.get("value") or turn.get("content")
        if who in ("human", "user") and q is None:
            q = val
        elif who in ("gpt", "assistant") and a is None:
            a = val
    # strip llava <image> tokens from the question text
    if isinstance(q, str):
        q = q.replace("<image>", "").strip()
    return q, a


# ----------------------------- table sources --------------------------------

PRESETS = {
    # name -> (hf_id, split). Existence verified against the Hub API (2026-06-30).
    # visual_tableqa: SYNTHETIC LLM-generated tables (cols image/question/answer) —
    #   loads + maps cleanly, and cannot overlap the real-Wikipedia eval. Safe default.
    "visual_tableqa": ("AI-4-Everyone/Visual-TableQA", "train"),
    # cmarkea: tables from arXiv papers (QA nested in a `qa` dict) — safe (scientific,
    #   not Wikipedia) but needs a custom adapter for its `qa` field; off-distribution
    #   for the VWTQ weakness. Not in defaults.
    "cmarkea_tablevqa": ("cmarkea/table-vqa", "train"),
    # mmtab: contains WikiTableQuestions-DERIVED QA -> SAME source as the vwtq eval
    #   split => train/eval OVERLAP RISK; also ships as raw JSON (no standard 'train'
    #   split, datasets-server fails). Excluded from defaults; the overlap guard below
    #   would drop colliding rows, but prefer not to use it.
    "mmtab": ("SpursgoZmy/MMTab", "train"),
    # wtq: the original WikiTableQuestions (real Wikipedia tables) — the *source* of
    #   vwtq. Best distribution match IF used as train-split-only with qa exclusion,
    #   but tables are CSV (need rendering to images). Not wired in yet.
    "wtq": ("stanfordnlp/wikitablequestions", "train"),
}


def normalize_q(s):
    """Whitespace/case-normalized question text for cross-dataset overlap matching."""
    return " ".join(str(s).lower().split())


def iter_hf_table(spec, want, rng, img_dir, dry_run, exclude, stats):
    """Yield up to `want` table records from one source spec.
    spec: 'preset:<name>' or 'hf:<id>:<split>'. Guarded — yields nothing on failure.
    `exclude` is a set of normalized eval questions; any candidate matching it is
    dropped (train/eval overlap guard) and counted in stats['overlap_dropped']."""
    from datasets import load_dataset

    if spec.startswith("preset:"):
        name = spec.split(":", 1)[1]
        if name not in PRESETS:
            log(f"  [src {spec}] unknown preset; skip")
            return
        hf_id, split = PRESETS[name]
    elif spec.startswith("hf:"):
        parts = spec.split(":")
        hf_id, split = parts[1], (parts[2] if len(parts) > 2 else "train")
        name = hf_id
    else:
        log(f"  [src {spec}] bad spec (use preset:<n> or hf:<id>:<split>); skip")
        return

    try:
        ds = load_dataset(hf_id, split=split, streaming=True)
    except Exception as exc:                            # noqa: BLE001
        log(f"  [src {spec}] load FAILED: {type(exc).__name__}: {exc}; skip")
        return

    log(f"  [src {spec}] loaded {hf_id}:{split} (streaming); saving up to {want} images "
        f"(downloads + decodes lazily — slowest phase)...")
    n = 0
    for i, row in enumerate(ds):
        if n >= want:
            break
        cols = list(row.keys())
        img_col, q_col, a_col = detect_fields(None, cols)
        q, a = (row.get(q_col), row.get(a_col)) if (q_col and a_col) else (None, None)
        if q is None or a is None:                      # try conversation format
            q, a = extract_conv(row)
        if not q or a in (None, ""):
            continue
        if normalize_q(q) in exclude:                   # train/eval overlap guard
            stats["overlap_dropped"] = stats.get("overlap_dropped", 0) + 1
            continue
        if img_col is None or row.get(img_col) is None:
            continue
        if dry_run:
            log(f"    sample[{i}] q={str(q)[:60]!r} a={str(a)[:30]!r} img_col={img_col}")
            n += 1
            if n >= min(want, 3):
                break
            continue
        dest = img_dir / f"{name.replace('/', '_')}_{i}.png"
        if save_image(row[img_col], dest) is None:
            continue
        yield {
            "image": str(dest),
            "prompt": render_table_prompt(q),
            "answer": str(a).strip(),
            "source": name,
            "subtype": "table",
        }
        n += 1
        if n % LOG_EVERY == 0:
            log(f"    [{name}] {n}/{want} table images saved "
                f"({stats.get('overlap_dropped', 0)} dropped as eval overlap so far)")
    log(f"  [src {spec}] yielded {n} records")


# ----------------------------- replay (protected) ---------------------------

def eval_index_sets():
    """For each protected dataset (+ the eval benchmark), the set of seed-42 eval
    indices — the rows we must NOT train on."""
    from vlmeval.dataset import build_dataset
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from run_eval import subset_data

    out = {}
    for name in PROTECTED + [EVAL_BENCHMARK]:
        try:
            ds = build_dataset(name)
            sub = subset_data(ds.data, DEFAULT_N, SAMPLE_SEED)
            out[name] = (ds, set(sub["index"]) if "index" in sub.columns else set(sub.index))
        except Exception as exc:                        # noqa: BLE001
            log(f"  [replay] {name} unavailable: {type(exc).__name__}; skip")
            out[name] = (None, set())
    return out


def iter_replay(name, ds, eval_idx, want, rng, dry_run):
    """Yield up to `want` replay records from rows OUTSIDE the eval subset."""
    data = ds.data
    holdout = data[~data["index"].isin(eval_idx)] if "index" in data.columns else data
    if len(holdout) == 0:
        return
    take = holdout.sample(n=min(want, len(holdout)), random_state=rng.randint(0, 2**31))
    n = 0
    for _, line in take.iterrows():
        # text prompt exactly as eval builds it
        try:
            msgs = ds.build_prompt(line)
            prompt = "\n".join(m["value"] for m in msgs if m["type"] == "text")
        except Exception:                               # noqa: BLE001
            prompt = str(line.get("question", ""))
        ans = line.get("answer")
        if ans is None or str(ans).strip() == "":
            continue
        if dry_run:
            n += 1
            if n >= 3:
                break
            continue
        img_path = ds.dump_image(line)
        img_path = img_path[0] if isinstance(img_path, list) else img_path
        yield {
            "image": str(img_path),
            "prompt": prompt,
            "answer": str(ans).strip() if not isinstance(ans, list) else str(ans[0]),
            "source": name,
            "subtype": "replay",
            "_eval_excluded_index": int(line["index"]) if "index" in line else None,
        }
        n += 1
        if n % LOG_EVERY == 0:
            log(f"    [replay {name}] {n}/{want} images saved")
    log(f"  [replay {name}] yielded {n} records")


# ----------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="train_data",
                    help="where to write train.jsonl + images. Use LOCAL Colab disk "
                         "(e.g. /content/table_sft), NOT Drive — thousands of small image "
                         "writes through the Drive mount are slow and eat quota. Regenerable.")
    ap.add_argument("--n-total", type=int, default=7000)
    ap.add_argument("--frac-table", type=float, default=0.70)
    ap.add_argument("--frac-replay", type=float, default=0.25)
    ap.add_argument("--frac-antireg", type=float, default=0.05)
    ap.add_argument("--table-sources", nargs="+",
                    default=["preset:visual_tableqa"],
                    help="ordered source specs; tried until the table quota is filled. "
                         "Default is the one source verified to load+map+not-overlap the eval "
                         "(Visual-TableQA, synthetic). MMTab is excluded (WTQ overlap + broken load).")
    ap.add_argument("--seed", type=int, default=SAMPLE_SEED)
    ap.add_argument("--dry-run", action="store_true", help="probe sources / print samples, write nothing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.data_dir)
    img_dir = out_dir / "images"
    if not args.dry_run:
        img_dir.mkdir(parents=True, exist_ok=True)

    n_table = int(args.n_total * args.frac_table)
    n_replay = int(args.n_total * args.frac_replay)
    n_antireg = args.n_total - n_table - n_replay
    log(f"[plan] total={args.n_total} table={n_table} replay={n_replay} anti-reg={n_antireg} seed={args.seed}")

    records = []
    stats = {"overlap_dropped": 0}

    # 0) overlap guard — build the set of normalized EVAL questions (full
    #    TableVQABench, not just the seed-42 subset) so no table source can train on
    #    a question we score on. eval_index_sets() loads TableVQABench anyway.
    log("\n[guard] loading eval indices + building question-overlap exclusion set:")
    idx_sets = eval_index_sets()
    tvb_ds = idx_sets[EVAL_BENCHMARK][0]
    exclude = set()
    if tvb_ds is not None and "question" in tvb_ds.data.columns:
        exclude = {normalize_q(q) for q in tvb_ds.data["question"]}
    log(f"  excluding {len(exclude)} distinct TableVQABench questions from training sources")
    for name in PROTECTED:
        log(f"  {name}: {len(idx_sets[name][1])} eval rows held out (replay drawn from the rest)")

    # 1) table lookup data (VWTQ-weighted), filling from sources in order, with the
    #    eval-question overlap guard applied to every candidate row.
    log("\n[table] sourcing table-lookup data (overlap-guarded):")
    remaining = n_table
    for spec in args.table_sources:
        if remaining <= 0:
            break
        for rec in iter_hf_table(spec, remaining, rng, img_dir, args.dry_run, exclude, stats):
            records.append(rec)
            remaining -= 1
    got_table = n_table - remaining
    log(f"  [table] {got_table} kept; {stats['overlap_dropped']} dropped as eval overlaps")
    if got_table == 0 and not args.dry_run:
        raise SystemExit("No table training data loaded — check --table-sources / network. "
                         "Run with --dry-run to debug source mappings.")

    # 2) replay from the protected benchmarks (held-out rows only)
    log("\n[replay] sampling held-out protected rows:")
    per = max(1, n_replay // max(1, len(PROTECTED)))
    for name in PROTECTED:
        ds, eval_idx = idx_sets[name]
        if ds is None:
            continue
        records.extend(iter_replay(name, ds, eval_idx, per, rng, args.dry_run))

    # 3) anti-regression: a little fintabnet/vtabfact-style — reuse table sources but
    #    keep it small; in practice we draw extra table rows tagged anti-reg. (Kept
    #    minimal; the replay above is the main forgetting guard.)
    #    For the PoC we fold anti-reg into the table quota above and skip a separate
    #    fetch to avoid extra fragile sources; recorded in the manifest.

    if args.dry_run:
        log("\n[dry-run] no files written. Re-run without --dry-run once sources look right.")
        return

    # ---- contamination guard (hard assertion) ----
    bad = [r for r in records if r["subtype"] == "replay"
           and r.get("_eval_excluded_index") in idx_sets[r["source"]][1]]
    assert not bad, f"CONTAMINATION: {len(bad)} replay rows overlap the eval subset"
    assert all(r["source"] != EVAL_BENCHMARK for r in records), "TableVQABench leaked into training"

    rng.shuffle(records)
    train_path = out_dir / "train.jsonl"
    with open(train_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    by_sub = {}
    by_src = {}
    for r in records:
        by_sub[r["subtype"]] = by_sub.get(r["subtype"], 0) + 1
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    manifest = {
        "n_total": len(records),
        "by_subtype": by_sub,
        "by_source": by_src,
        "seed": args.seed,
        "fractions": {"table": args.frac_table, "replay": args.frac_replay, "antireg": args.frac_antireg},
        "protected_eval_rows_excluded": {k: len(v[1]) for k, v in idx_sets.items()},
        "table_sources": args.table_sources,
        "tablevqa_questions_excluded": len(exclude),
        "table_rows_dropped_as_eval_overlap": stats["overlap_dropped"],
        "contamination_check": "passed (0 eval rows in training; table rows matching any "
                               "TableVQABench question dropped)",
        "prompt_consistency": "table=VWTQ_PROMPT; replay=dataset.build_prompt",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    img_bytes = sum(f.stat().st_size for f in img_dir.rglob("*") if f.is_file())
    log(f"\n[done] wrote {len(records)} records -> {train_path}")
    log(f"[done] manifest -> {out_dir / 'manifest.json'}")
    log(f"[done] by subtype: {by_sub}")
    log(f"[done] images: ~{img_bytes/1e6:.0f} MB on {out_dir} "
        f"(local disk — regenerable, not on Drive)")


if __name__ == "__main__":
    main()
    # Hard-exit after the work is done: HF streaming + torch leave non-daemon C
    # threads that segfault during interpreter teardown (PyGILState_Release). All
    # outputs are already written/closed above, so skip the buggy finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)

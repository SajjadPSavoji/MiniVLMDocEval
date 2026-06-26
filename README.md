# MiniVLMDocEval

Systematic evaluation of **sub-1B vision-language models** on **document understanding**, using [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) as the evaluation engine — plus a knowledge-gap analysis and an improvement strategy for the best model.

- **Task brief:** [task.md](task.md)
- **Strategy, model/benchmark/metric choices, and step-by-step plan:** [research_plan.md](research_plan.md)

## Models under evaluation (all ≤1B)

`SmolVLM2-500M` · `FastVLM-0.5B` · `LFM2.5-VL-450M` · `InternVL3-1B` · `LLaVA-OneVision-0.5B` · `Qwen3.5-0.8B` — 6 generalist VLMs across 6 labs and 5 encoder paradigms. See [research_plan.md](research_plan.md) §Step 1 for the full rationale.

## Benchmarks

`OCRBench` · `DocVQA` · `ChartQA` · `InfoVQA` · `TableVQABench` — a diagnostic ladder from raw OCR to layout/table reasoning. See §Step 2A.

## Repo layout (script-first; one notebook)

```
setup.sh                  Idempotent bootstrap (clone VLMEvalKit @ pinned commit, install editable)
requirements.txt          Dependency record (pinned commit; install via setup.sh)
scripts/                  Thin entrypoints you run on Colab (all logic is in .py)
  smoke_test.py             1 model x 1 dataset x N samples, end-to-end
notebooks/
  colab.ipynb             The ONLY notebook — a thin "terminal": git pull + run a script
external/VLMEvalKit/      The engine — cloned by setup.sh at a pinned commit (gitignored; see external/README.md)
```

All real code is `.py`. The notebook is just a stable terminal — each experiment
is one `!git pull && python scripts/<x>.py ...` line. No per-experiment notebooks.

## Where things run

VLMEvalKit's model wrappers **hardcode CUDA**, so the eval runs on **GPU (Colab)**, not mac.
The mac is for **development**: editing scripts and plumbing-testing wrapper logic.

### Workflow (VS Code ⇄ Colab over git)

The Colab GPU is driven from VS Code via the Google Colab extension. Code crosses
the local↔cloud boundary through **git** (Claude edits local `.py` → push → Colab pulls):

1. `REPO_URL` in `notebooks/colab.ipynb` already points at this repo.
2. Set runtime to **T4 GPU**; run the two one-time cells (clone, `setup.sh`).
3. Iterate: re-run the **Run cell** (`git pull && python scripts/...`) after each push.
   (Or use the extension's terminal panel and type the same command directly.)

> Setup targets **Linux/Colab**. The eval needs CUDA, so it does not run on mac
> (VLMEvalKit's wrappers are CUDA-bound); local mac is not a supported run target.

## Running the full evaluation

`scripts/run_eval.py` runs the built-in models (`minivlmdoceval/config.py`) on
the **full** benchmark suite via VLMEvalKit, then aggregates per-(model, dataset)
primary metrics into a comparison table (refreshed after each model).

**Everything is written under the `--work-dir`** — both the heavy predictions and
the `summary/comparison.{csv,md}` tables. Point it at **Google Drive** so all of
it persists across Colab sessions (and VLMEvalKit `--reuse` can resume a run).
Full runs are large (~13k samples/model) and span multiple sessions.

In the Colab T4 terminal (after mounting Drive in a notebook cell:
`from google.colab import drive; drive.mount('/content/drive')`):

```bash
cd /content/MiniVLMDocEval && git pull
python scripts/run_eval.py --work-dir /content/drive/MyDrive/MiniVLMDocEval/outputs
# table is printed to the terminal AND saved to <work-dir>/summary/comparison.md
```

## Status

Active development — see the **Execution Steps** in [research_plan.md](research_plan.md) for current progress.

# MiniVLMDocEval

**Are sub-billion-parameter vision–language models a reliable foundation for document
understanding — and where do they break?**

A systematic, reproducible evaluation of **six open-source sub-1B VLMs** across **five
document benchmarks**, followed by an evidence-driven **knowledge-gap analysis** and a
**targeted improvement study** on the best model. Built on
[VLMEvalKit](https://github.com/open-compass/VLMEvalKit) as a pinned dependency.

📄 **Technical report:** [`technical_report/report.pdf`](technical_report/report.pdf) ·
🧪 **Reproduce:** [Colab notebook](notebooks/colab.ipynb) or [`scripts/reproduce.sh`](scripts/reproduce.sh) ·
🧭 **Design & decisions:** [`research_plan.md`](research_plan.md) · 📋 **Brief:** [`task.md`](task.md)

🤗 **Dataset:** [savoji/minivlm-tablevqa-sft](https://huggingface.co/datasets/savoji/minivlm-tablevqa-sft) ·
🤗 **LoRA adapter:** [savoji/qwen3.5-0.8b-tablevqa-lora](https://huggingface.co/savoji/qwen3.5-0.8b-tablevqa-lora)
*(the Part-2 experimental adapter — published for transparency; it failed the gate, see below)*

📂 **Full result artifacts** (predictions, per-pair scores, logs, summary tables): [Google Drive](https://drive.google.com/drive/folders/1xDkoQLUG3d-ymxNvvLpKryRsGe8xpLvc?usp=sharing)

---

## Key results

Six generalist VLMs (0.45–0.9B params) under a matched protocol — greedy decoding,
uniform 16-bit precision, a 128-token generation cap, and fixed **N=1000 seed-42** subsets.
Scores are each benchmark's native metric, normalized to 0–100.

| Rank | Model | Params | ChartQA | DocVQA | InfoVQA | OCRBench | TableVQA | **Mean** |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | **Qwen3.5-0.8B** | 0.8B | 70.8 | **89.3** | **62.3** | 79.2 | **50.1** | **70.3** |
| 2 | InternVL3-1B | 0.9B | 68.8 | 80.8 | 54.2 | **79.4** | 33.5 | 63.3 |
| 3 | LFM2.5-VL-450M | 0.45B | **73.1** | 77.2 | 41.3 | 67.8 | 40.2 | 59.9 |
| 4 | LLaVA-OneVision-0.5B | 0.9B | 60.0 | 70.4 | 39.0 | 60.2 | 34.4 | 52.8 |
| 5 | SmolVLM2-500M | 0.5B | 60.4 | 67.8 | 27.4 | 61.0 | 30.3 | 49.4 |
| 6 | FastVLM-0.5B | 0.8B | 44.3 | 63.6 | 32.5 | 26.2 | 17.5 | 36.8 |

**Findings.** (1) **Qwen3.5-0.8B is the strongest foundation** (mean 70.3, wins 3/5).
(2) **Architecture beats parameter count** — the smallest model (0.45B LFM2.5-VL) outranks
both 0.9B models and wins ChartQA. (3) The **image-resolution strategy** — not scale — is the
dominant driver of both accuracy and latency. (4) **Visual table QA is the frontier**: no model
exceeds 50.1.

**The gap, localized.** Decomposing TableVQA into its sub-domains shows the best model's
weakness is **Wikipedia-style visual lookup** (VWTQ 27.8 / VWTQ-Syn 33.1), *not* financial or
numeric tables (FinTabNetQA 84.0) — correcting a common assumption in the table-QA literature.

**Part 2 (improvement, reported faithfully).** A targeted LoRA fine-tune to close the lookup gap
**failed a pre-registered no-regression gate** (catastrophic forgetting: TableVQA 50.1→19.6,
DocVQA 89.3→50.8). The diagnosis is a train/eval **answer-style mismatch** (verbose training
answers vs. short-answer scoring); the report gives a concrete corrected recipe. The gate
behaving as designed — rejecting a degraded model — is itself the intended outcome of the
protocol. See report §8.

---

## Reproduce

Six models, two incompatible `transformers` versions → reproduction runs in **two phases**
(a runtime restart between them), then a light figures phase. Results persist under a single
`OUT_DIR` and merge into one comparison table.

- **Env A** (`transformers<4.57`): SmolVLM2, InternVL3, LLaVA-OneVision, FastVLM
- **Env B** (`transformers@main`): Qwen3.5-0.8B, LFM2.5-VL-450M, and the Part-2 study

### Option 1 — Colab (recommended)
Open [`notebooks/colab.ipynb`](notebooks/colab.ipynb) and run **top to bottom**: Phase A →
**restart runtime** (banner in the notebook) → Phase B → Phase C. Point outputs at Google
Drive so they persist. T4 suffices for Phase A; prefer **A100/L4** for Phase B.

### Option 2 — script, on any CUDA machine
```bash
git clone https://github.com/SajjadPSavoji/MiniVLMDocEval.git && cd MiniVLMDocEval

OUT_DIR=outputs bash scripts/reproduce.sh env-a     # Phase A
#   --> start a FRESH shell / environment (new transformers) <--
OUT_DIR=outputs bash scripts/reproduce.sh env-b     # Phase B (+ Part 2)
OUT_DIR=outputs bash scripts/reproduce.sh figures   # report plots
```
Outputs land in `outputs/`: `summary/comparison.{md,csv}` (the table), `predictions/<model>/…`
(per-pair scores), and `logs/`. Everything is seeded and pinned; a re-run reproduces the numbers.

> **Reproducibility knobs.** Fixed seed (42), fixed N (1000), greedy decoding, pinned VLMEvalKit
> commit, per-model 16-bit precision, and pinned training deps. All configurable in
> [`minivlmdoceval/config.py`](minivlmdoceval/config.py).

---

## How it works

- **Evaluation engine:** [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) at a pinned
  commit (cloned by `setup.sh`, never forked). Adding a model = one `generate_inner` method.
- **Runner:** [`scripts/run_eval.py`](scripts/run_eval.py) — a single config-driven entry point
  scores *any* registered model on a fixed-seed N-sample subset, with per-pair caching/resume,
  and refreshes the comparison table after each pair.
- **Model registry:** [`minivlmdoceval/custom_models.py`](minivlmdoceval/custom_models.py) —
  we contribute custom wrappers that add three models the harness lacks natively
  (**FastVLM-0.5B, Qwen3.5-0.8B, LFM2.5-VL-450M**) plus the fine-tuned `Qwen3.5-0.8B-TableLoRA`.
- **Benchmarks (a diagnostic ladder):** OCRBench → DocVQA → ChartQA → InfoVQA → TableVQABench
  (recognition → extraction → numeric → layout → tabular).

## Repository layout

```
setup.sh                        Env bootstrap: clone VLMEvalKit @ pinned commit; Env A / --bleeding-edge (Env B) / --train
scripts/
  reproduce.sh                  One-command reproduction driver (env-a | env-b | figures)
  run_eval.py                   Config-driven eval runner (scores any registered model)
  smoke_test.py                 Fast end-to-end sanity check
  efficiency_ablation.py        fp16 + generation-cap ablations
  tablevqa_subdomain_report.py  TableVQA per-sub-domain breakdown (the gap diagnosis)
  build_table_sft.py            Part-2 training mixture (contamination-guarded)
  train_lora_qwen.py            Part-2 bf16 LoRA fine-tune (+ --smoke de-risk)
  regression_gate.py            Pre-registered no-regression gate
  export_examples.py            Qualitative examples for the report
  make_report_figures.py        Figures: Pareto, per-benchmark, sub-domains
  make_examples_figure.py       Figure: qualitative-example gallery
  push_dataset_hf.py            Publish the training set to the HuggingFace Hub
minivlmdoceval/
  config.py                     Models, benchmarks, N, seed, output tree
  custom_models.py              Custom VLMEvalKit wrappers + LoRA variant
notebooks/colab.ipynb           Reproducible two-phase runner
technical_report/               NeurIPS-style report (report.tex → report.pdf) + figures + references.bib
research_plan.md                Strategy, model/benchmark/metric rationale, and Part-2 results
task.md                         The assignment brief
external/VLMEvalKit/            The engine — cloned by setup.sh at a pinned commit (gitignored)
```

## Setup notes

`setup.sh` targets **Linux/Colab** and requires a CUDA GPU (VLMEvalKit's wrappers are
CUDA-bound). It clones VLMEvalKit at a pinned commit and installs it editable. Flags:
`--bleeding-edge` (Env B, `transformers@main`) and `--train` (adds `peft`/`trl`/`accelerate`
for Part 2). The mac is for development only; the eval runs on GPU.

---

*Author: Sajjad Pakdamansavoji · [sj.pakdaman.edu@gmail.com](mailto:sj.pakdaman.edu@gmail.com)*

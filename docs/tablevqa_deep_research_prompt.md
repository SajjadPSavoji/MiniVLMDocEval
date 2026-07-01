# Deep-Research Request — Improving Visual Table QA in a Sub-1B Vision-Language Model

> Paste the section below into a deep-research agent. It is self-contained.

---

## Role
You are a senior multimodal-ML research assistant. I need a rigorous, **literature-grounded** (prioritize 2023–2026) analysis and a concrete, feasible plan. Cite sources (papers, datasets, repos) with links wherever possible, and prefer methods demonstrated on **small / edge** vision-language models.

## Project context (short)
We are running a systematic study of **small (≤1B parameter) open-source vision-language models (VLMs) for document understanding**. The plan is: (1) benchmark the top small VLMs on document tasks, (2) **pick the strongest foundation**, and (3) **improve its weakest capability** without regressing the rest. We have now completed the full benchmark of **all six** shortlisted models — and the strongest model's weakest task is **visual table QA**. Evaluation uses the open-source **VLMEvalKit** toolkit on free GPUs (Google Colab; NVIDIA T4 16 GB / L4 24 GB).

## Models evaluated (all ≤1B params, open-source) — full shortlist of 6
- **Qwen3.5-0.8B** (~0.8B): Qwen-native vision encoder + gated-delta/MoE backbone, **native dynamic resolution (NaViT-style)**. **Best model overall in our study, and also the best on TableVQA (50.1) — this is our improvement target** (see "Improvement target" below): we want to push the strongest foundation's own weakest task higher.
- **InternVL3-1B** (~0.9B): InternViT-300M vision encoder + Qwen2.5-0.5B LLM, **dynamic high-resolution tiling**. Second overall (mean 63.3); a useful contrastive reference — a strong reader (OCRBench 79, DocVQA 81) that nonetheless collapses on tables (33.5).
- **LFM2.5-VL-450M** (~0.45B): SigLIP2-86M + LFM2-350M, native ≤512² + **tunable patch-tiling**. Smallest model, ~5× the fastest, and 3rd overall — the efficiency front-runner.
- **LLaVA-OneVision-0.5B** (~0.9B): SigLIP encoder + Qwen2-0.5B, AnyRes multi-crop.
- **SmolVLM2-500M** (~0.5B): SigLIP + SmolLM2-360M, aggressive visual-token compression.
- **FastVLM-0.5B** (~0.8B): FastViTHD (hybrid CNN-ViT) + Qwen2-0.5B, high-res / few-token encoder-side compression. Weakest overall in our setup.

## Benchmarks & metrics (document understanding; 1000-sample fixed-seed subsets, greedy decoding, 16-bit)
| Benchmark | Skill | Metric (0–100) |
|---|---|---|
| OCRBench | text/OCR recognition | accuracy |
| DocVQA (val) | dense document QA | ANLS |
| ChartQA (test) | chart numeric reasoning | relaxed accuracy (±5%) |
| InfoVQA (val) | infographic layout + multi-hop | ANLS |
| **TableVQABench** | **visual table QA** over 4 sub-domains (VWTQ, VWTQ-Syn, VTabFact, FinTabNet) | accuracy |

## Key results (accuracy, 0–100; higher is better) — full 6-model run, sorted by mean
| Model | ChartQA | DocVQA | InfoVQA | OCRBench | **TableVQA** | Mean |
|---|---|---|---|---|---|---|
| **Qwen3.5-0.8B** *(target)* | 70.8 | 89.3 | 62.3 | 79.2 | **50.1** | 70.3 |
| InternVL3-1B | 68.8 | 80.8 | 54.2 | 79.4 | **33.5** | 63.3 |
| LFM2.5-VL-450M | 73.1 | 77.2 | 41.3 | 67.8 | **40.2** | 59.9 |
| LLaVA-OV-0.5B | 60.0 | 70.4 | 39.0 | 60.2 | 34.4 | 52.8 |
| SmolVLM2-500M | 60.4 | 67.8 | 27.4 | 61.0 | 30.3 | 49.4 |
| FastVLM-0.5B | 44.3 | 63.6 | 32.5 | 26.2 | 17.5 | 36.8 |

## The knowledge gap (the problem to solve)
**Visual table QA is the weakest task for every model in the study** — it is the lowest-scoring column for 5 of 6 models and the single hardest capability overall (range 17.5–50.1). Models that *read* text well still **fail at structured-table reasoning**: cell lookup, row/column cross-referencing, multi-cell aggregation, and table fact-verification. Even our best model, **Qwen3.5-0.8B, tops the field at just 50.1** — that is *its own* weakest task, **~12 pts below its next-worst** (InfoVQA 62.3) and **~39 pts below its DocVQA** (89.3). So structured-table reasoning is the **frontier capability even for the strongest sub-1B VLM**, and the obvious target for a focused improvement.

The rest of the field sits far lower (InternVL3-1B 33.5, LFM2.5-VL-450M 40.2, others ~30–34), and encoder strength that helps on OCR/DocVQA does **not** transfer to tables. The cross-model spread is itself a clue: Qwen's native dynamic-resolution encoding + stronger/MoE backbone already buy **~+17 pts over InternVL** — yet still leave **roughly half the table questions wrong**. The problem to solve is to push **Qwen3.5-0.8B beyond 50.1** on TableVQABench, without regressing its other four (already-strong) document scores.

## Improvement target (Qwen3.5-0.8B)
We scope the improvement study on **Qwen3.5-0.8B** because: (a) it is the **best overall foundation** *and* already the **best on TableVQA (50.1)** — improving the strongest base's own weakest task is the cleanest, most deployable result (no point hardening a weaker model); (b) it belongs to the **widely-adopted Qwen-VL family**, which has a **mature open fine-tuning ecosystem** (e.g. LLaMA-Factory, ms-swift, well-trodden PEFT/LoRA recipes, and larger same-family siblings usable as teachers); and (c) TableVQA is unambiguously its weakest task (50.1 vs 62–89 elsewhere), so there is clear headroom to target without cannibalizing strengths. **Practical caveat to address up front:** this is a recent release on bleeding-edge `transformers` (gated-delta/MoE) — pin exact versions and **verify PEFT/LoRA adapter support against this specific architecture** before committing to a recipe; call out any tooling gaps.

## Core research question
**How can we improve Qwen3.5-0.8B on visual table question answering (TableVQABench — currently 50.1, its weakest task) while preserving its strong performance on the other four document benchmarks (OCRBench 79, DocVQA 89, ChartQA 71, InfoVQA 62), under limited compute (a single free/low-cost GPU, T4/L4)? It is already the best sub-1B model on tables — so what is its *residual* table-reasoning failure mode, and how do we push it meaningfully higher without regressions?**

## Constraints
- Model must stay **≤1B params** and edge-deployable; inference cost should not balloon.
- Compute is limited: a single **T4 (16 GB)** or **L4 (24 GB)**; **parameter-efficient fine-tuning (LoRA/QLoRA/DoRA) strongly preferred** over full fine-tuning.
- **Must avoid catastrophic forgetting** — the other four benchmarks must stay high.
- Methods and data should be **open-source and reproducible**.

## What I need from you (deliverable)
Please produce a structured report covering:

1. **Diagnosis** — Why do small VLMs fail at visual table reasoning? Discuss candidate causes with evidence: input resolution / table legibility, absence of table-structure-aware pretraining, limited LLM reasoning capacity at this scale, linearization/tokenization of 2-D tabular layout, train-data scarcity, etc. Use the same-scale spread in our data as a lever: **Qwen3.5-0.8B already leads at 50.1** (vs LFM2.5-VL-450M 40.2 and InternVL3-1B/others ~30–34) — attribute that lead to concrete factors (native dynamic-resolution vs fixed tiling, backbone capacity/MoE, table-richer instruction data, tokenization). Then go **further into our target specifically: what does Qwen3.5-0.8B *itself* still get wrong** to be stuck at 50.1 — which table sub-skills (structure parsing, cell lookup, numeric aggregation, fact-verification) and which of the 4 sub-domains (VWTQ, VWTQ-Syn, VTabFact, FinTabNet) drive its residual errors? State which causes are **fixable by post-training** vs **baked into pretraining/architecture**.

2. **Ranked methodologies to improve TableVQA** — ranked by expected impact × feasibility under our constraints. For each: the approach, key papers/links, expected gain, compute cost, and risks. At minimum consider:
   - **PEFT instruction-tuning** (LoRA/QLoRA/DoRA) on table-QA data — what configurations work for small VLMs.
   - **Table-specific training data** — existing datasets (real + synthetic table-image QA, e.g. WikiTableQuestions/TabFact/FinTabNet-derived, PubTables, synthetic table renderers) and how to construct/augment more; HTML/markdown-table ↔ image pairs.
   - **Input/representation interventions** — higher-resolution tiling, table-structure/layout tokens, OCR- or HTML-assisted prompting, table linearization strategies, grounding.
   - **Architectural tweaks** compatible with small VLMs.
   - **Knowledge distillation** from a larger table-expert model (or table-structure-recognition model) into the sub-1B VLM.
   - **In-family distillation / data generation** — exploit the Qwen-VL family: use a **larger Qwen-VL table expert** as a teacher for Qwen3.5-0.8B (response/rationale distillation, or to synthesize targeted table-QA data). Same tokenizer/chat-format and prompt conventions lower integration friction vs a cross-family teacher; weigh cost/quality against a non-Qwen table expert.
   - **Error-targeted data** — use Qwen3.5-0.8B's own failure analysis (the sub-domains/sub-skills it misses at 50.1) to construct or oversample exactly the table-QA examples that address its residual errors, rather than generic table data.

3. **Catastrophic-forgetting mitigation** — concrete strategies to specialize on tables without regressing OCR/Doc/Chart/Info: replay/rehearsal data mixtures and ratios, multi-task instruction blends, LoRA module isolation/merging, regularization, learning-rate/scheduling choices.

4. **Recommended recipe** — one concrete, end-to-end plan we could run on a **single T4/L4** to demonstrate a measurable TableVQA gain with no regressions: dataset(s) + size, method + key hyperparameters, training budget, and an evaluation protocol.

5. **Evaluation of table sub-skills** — how to measure and separate table sub-capabilities (structure parsing vs cell lookup vs numeric aggregation vs fact verification) and how to detect regressions on the other benchmarks.

6. **Starter resources** — the most relevant papers, datasets, and open-source repos to begin from, with links.

Be specific and cite sources. Where evidence exists for ≤2B or edge VLMs specifically, prioritize it.

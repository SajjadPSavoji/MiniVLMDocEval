# A Lightweight PoC to Improve Visual Table QA in Qwen3.5-0.8B

## TL;DR
- **Build this:** A bf16 (not 4-bit) LoRA fine-tune of Qwen3.5-0.8B on a ~10K-example mixture of MMTab-instruct + Visual-TableQA + TABLET financial tables, with a ~25–30% replay of general document data (DocVQA/ChartQA/OCR) to prevent forgetting — runnable in a few hours on a single free Colab T4/L4 via Unsloth's official Qwen3.5-0.8B Vision notebook. Target: push TableVQABench past 50.1 with no regression on the other four benchmarks.
- **Why it works:** TableVQABench's residual errors concentrate in FinTabNetQA (dense financial tables, merged cells, numeric aggregation) and the harder VTabFact (semantic fact verification) items — not in plain OCR. Small VLMs fail on table *structure parsing and numeric reasoning*, which are exactly the skills targetable by structure-aware instruction data, while LoRA "learns less and forgets less," protecting the already-strong scores.
- **Critical gotcha:** Unsloth explicitly warns NOT to use QLoRA 4-bit on Qwen3.5 (dense or MoE) due to abnormally large quantization error; the 0.8B model only needs ~3GB for bf16 LoRA, so 4-bit is unnecessary anyway. Use bf16/16-bit LoRA and pin transformers v5.

## Key Findings

1. **The weakness is reasoning/structure, not OCR.** TableVQABench (arXiv 2404.19205; HF `terryoo/TableVQA-Bench`) has 1,500 QA pairs split VWTQ (750), VWTQ-Syn (250), VTabFact (250), FinTabNetQA (250). 2025 studies (ExpliCIT-QA arXiv 2507.11694; TALENT arXiv 2510.07098) show VWTQ/VWTQ-Syn (Wikipedia-style lookup) and FinTabNetQA (multi-row headers, merged cells) behave very differently across models. TALENT's error analysis states verbatim: "Most errors occur not in data retrieval but in subsequent numerical calculations, indicating that the mathematical reasoning of smaller LLMs remains a bottleneck even with accurate contextual inputs." ExpliCIT-QA similarly reports FinTabNetQA "structures remain challenging and prone to misinterpretation" while VTabFact fact verification is "less compatible with… code-based reasoning."

2. **Resolution/legibility is a real but partly-fixable cause.** In MirageTVQA (Singh et al., "Lost in Translation and Noise," arXiv 2511.17238v1, Nov 21 2025), the top model degraded sharply under visual noise: "The top-performing Qwen2.5-VL-72B model, which scores 25.52% EM on clean images, sees its performance degrade to just 16.50% EM on the noisy set with drop of over 35%. This trend is consistent across all evaluated models." Qwen3.5-0.8B's native dynamic resolution (NaViT-style) mitigates this somewhat, but very dense financial tables still lose legibility.

3. **Small-VLM table fine-tuning works — with a forgetting caveat.** Visual-TableQA (Lompo & Haraoui, arXiv 2509.07966, Table 5) improved LLaVA-Next-Llama3-8B substantially (VWTQ 23.6→32.93, VTabFact 42.04→52.0) but *degraded* an already-strong Qwen2.5-VL-7B-Instruct (VTabFact 84.4→77.25, VWTQ 68.99→59.6, VWTQ-Syn 77.92→73.82). This is the central risk for the user's strong baseline and dictates a replay-heavy, conservative recipe. The protective evidence: Biderman et al., "LoRA Learns Less and Forgets Less" (TMLR 2024, arXiv 2405.09673): "LoRA better maintains the base model's performance on tasks outside the target domain. We show that LoRA mitigates forgetting more than common regularization techniques such as weight decay and dropout."

4. **Tooling is ready.** Unsloth ships official free Colab Qwen3.5-0.8B Vision LoRA notebooks (bf16 LoRA = ~3GB VRAM; it lists "0.8B: 3GB" explicitly). LLaMA-Factory and ms-swift both list Qwen3-VL support; ms-swift had a LoRA-not-learning bug on Qwen3-VL MoE variants (issue #6207) — a reason to prefer the dense 0.8B and verify adapters actually change outputs.

## Details

### 1. Diagnosis: why small VLMs fail at tables

| Candidate cause | Evidence (2025–26) | Fixable by post-training? |
|---|---|---|
| Table-structure-unaware pretraining | Table-LLaVA/MMTab (arXiv 2406.08100) shows a dedicated table-recognition pretraining stage is what unlocks tabular reasoning | **Yes** — add structure tasks (cell extraction, TSR) to SFT mix |
| Limited LLM numeric reasoning at <1B | TALENT: numeric calculation is the bottleneck even with perfect OCR | **Partially** — rationale/CoT supervision helps; hard ceiling at 0.8B |
| 2-D layout linearization loss | TALENT: Markdown/HTML conversion loses merged-cell/header relations | **Partially** — structure-aware data + narration |
| Resolution/legibility | MirageTVQA: Qwen2.5-VL-72B fell 25.52%→16.50% EM under noise | **Partially** — tiling/higher-res input |
| Train-data scarcity | MMTab (232K instruct), TABLET (~4M), Visual-TableQA (6K rationale QA) now fill this | **Yes** |

Hardest sub-skills, in order: multi-cell numeric aggregation > structure parsing of merged/hierarchical headers > semantic fact verification > single-cell lookup. Hardest sub-domains: **FinTabNetQA** and the harder **VTabFact** items drive residual error; VWTQ-style lookup is comparatively addressable. Note that even strong models can be weak on Wikipedia retrieval — Visual-TableQA Table 5 shows GPT-4o scoring just 66.5 on VWTQ vs 89.6 on VTabFact — so do not assume VWTQ is "solved" for a 0.8B model; measure it. **Baked-into-architecture** (not fully fixable by PEFT): the 0.8B LLM's arithmetic ceiling on multi-step numeric aggregation.

### 2. THE RECOMMENDED PoC RECIPE (core deliverable)

**Method — bf16 LoRA via Unsloth (NOT QLoRA):**
- Base: `unsloth/Qwen3.5-0.8B` loaded with `load_in_16bit=True`, `load_in_4bit=False`. Pin `transformers` v5, plus latest `peft`, `trl`, `unsloth`.
- LoRA: rank `r=16`, `lora_alpha=16` (Unsloth recommends alpha ≥ r), `lora_dropout=0`, `bias="none"`, target modules `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`.
- **Freeze the vision encoder and merger** (`finetune_vision_layers=False`), adapt language layers only — this saves VRAM and protects visual features the four strong benchmarks rely on. (The widely-cited Qwen3-VL fine-tuning default is "Freeze the ViT… Apply LoRA to the LLM decoder only.")
- Optimizer `adamw_8bit`; **conservative LR 1e-4** (below Unsloth's 2e-4 default) with cosine schedule, warmup ratio 0.05.
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=8–16`, `max_seq_length=2048`, gradient checkpointing on (`use_gradient_checkpointing="unsloth"`).
- **1–2 epochs only** (limited epochs is itself a forgetting mitigation).
- VRAM: ~3GB model + activations; fits T4 (16GB) comfortably, L4 (24GB) with headroom for a larger batch. Expect a few hours wall-clock for ~10K examples on T4 — note Qwen3.5's custom Mamba/Gated-DeltaNet Triton kernels compile slowly on T4 on the first run.

**Datasets + sizes (~10–12K examples total):**
- **MMTab-instruct** (`SpursgoZmy/MMTab`, file `MMTab-instruct_sft_data_llava_format_232K.json`, 232K samples across 82K table images): sample ~4,000, weighted toward table-structure-understanding tasks (cell extraction/location, merged-cell detection) and WTQ/TabFact/financial QA.
- **Visual-TableQA** (`AI-4-Everyone/Visual-TableQA`, 2.5K tables / 6K reasoning QA *with rationales*, generated for under USD 100): use ~2,500 for multi-step reasoning supervision.
- **TABLET** (`alonsoapp/TABLET-Small` / `-Medium` / `-Large`; arXiv 2509.21205, accepted ICLR 2026; ~4.07M examples / ~2.03M unique tables, 88% preserving original visualizations): sample ~2,000 financial/dense original-visualization tables (its TAT-QA, HiTab, FinTabNet-derived tasks) to directly target the FinTabNetQA weakness. On the held-out VisualTableQA benchmark, fine-tuning Qwen2.5-VL-7B on TABLET-Medium lifted exact-match accuracy from 42.4% (zero-shot) and 41.1% (MMTab) to 47.8% — and on AIT-QA (financial/airline tables) TABLET-Large reached 70.8 vs 51.7 zero-shot — evidence that original-visualization financial data transfers.
- **Replay / rehearsal (~2,500–3,000, i.e. 25–30%)**: general document data the model is already strong on — DocVQA, ChartQA, and OCR-style samples — to anchor the four protected benchmarks.

**Error-targeted oversampling:** over-represent (a) FinTabNetQA-style multi-header financial tables with numeric-aggregation questions, and (b) VTabFact-style True/False fact-verification items, since these drive residual error. Keep VWTQ-style lookup at baseline proportion but still include it (do not assume it is saturated).

**Data formatting (match Qwen3.5 chat template):** Use the ShareGPT/`messages` format consumed by Unsloth/TRL/LLaMA-Factory/ms-swift: a system message, a user message with `{"type":"image","image":...}` + `{"type":"text","text": question}`, and an assistant message with the answer; then apply `processor.apply_chat_template(...)`. Adopt TABLET's trick of wrapping the gold answer in a JSON object (`{"answer": "..."}`) for a subset to stabilize output parsing, but — following TABLET's own design — keep a portion in free-form/rationale style "to increase instruction diversity and mitigate catastrophic forgetting… reduces overfitting to a single output style."

**Catastrophic-forgetting mitigations baked in:** (1) 25–30% general-doc replay (a 1:3 to 1:2 replay-to-new ratio, consistent with rehearsal-based continual-learning literature, e.g. arXiv 2403.01244); (2) LoRA isolation (Biderman et al. 2024, above); (3) conservative LR (1e-4) + ≤2 epochs; (4) include MMTab-derived instruction-following items for instruction diversity; (5) keep the adapter un-merged at inference so the base model can be restored, and merge only after confirming no regression.

### 3. Ranked alternatives (impact × feasibility)

1. **Add table-structure-aware SFT tasks (highest impact — already folded into the recipe).** Table-LLaVA/MMTab proves structure-recognition supervision is the key unlock; its two-stage design first trains the connector on a table-recognition task (HTML output) before tabular instruction tuning. Repo: `github.com/SpursGoZmy/Table-LLaVA`. Expected gain: largest single lever. Compute: same as recipe. Risk: low.
2. **In-family rationale distillation from Qwen3.5-2B / Qwen3-VL-4B.** Use a larger same-tokenizer sibling to generate CoT rationales / synthetic FinTabNet QA, then SFT the 0.8B on them. Visual-TableQA already shows rationale-rich synthetic data transfers (and improved Qwen2.5-VL-7B on ReachQA 49.23%→60.95%, MATH-Vision 25.10%→49.77%). Expected gain: moderate-high on numeric items. Compute: teacher inference on L4 or via API. Risk: hallucinated rationales — filter with an LLM jury.
3. **OCR/HTML + narration prompting (TALENT-style, inference-time, no training).** Prompt the VLM to emit a Markdown table *and* a natural-language narration, then reason over both. TALENT (arXiv 2510.07098; `github.com/Melodramma080727/TALENT-Table-VQA-via-Augmented-Language-Enhanced-Natural-text-Transcription`) shows a small VLM + LLM can match a single large VLM at lower cost. Expected gain: moderate. Risk: adds a second model/latency and partly violates the single-edge-model constraint — best as a comparison baseline.
4. **Higher-resolution tiling for FinTabNetQA (architecture-light).** Raise max image tokens / tile dense tables. Cheap. Expected gain: small-moderate on dense financial tables (motivated by the MirageTVQA noise/legibility result). Risk: inference cost creep — bounds the edge constraint.

### 4. Evaluation protocol

- **TableVQABench via VLMEvalKit:** run `python run.py --model <your_model> --data TableVQABench` (open-compass/VLMEvalKit). Report the four sub-domains separately (VWTQ, VWTQ-Syn, VTabFact, FinTabNetQA) — both VLMEvalKit and the upstream `naver-ai/tablevqabench` repo score them independently with strict exact-match (True/False for VTabFact; the upstream `evaluate.py` takes `--evaluation_datasets vwtq vwtq_syn vtabfact fintabnetqa`). Qwen3.5-0.8B defaults to non-thinking; set `SPLIT_THINK=True` only if you enable thinking mode.
- **Sub-skill measurement:** tag eval items by skill (structure parse / cell lookup / numeric aggregation / fact verification) and compute per-skill accuracy. FinTabNetQA ≈ numeric aggregation + multi-header parsing; VTabFact = fact verification; VWTQ/VWTQ-Syn = lookup/retrieval. Watch the normalization trap that TALENT/ExpliCIT-QA flag (e.g., "$44,517" vs "44517", "$54,800" vs "$54,800 thousand") — use the benchmark's own matcher, and add explicit unit/format instructions in training.
- **Regression detection:** evaluate OCRBench, DocVQA, ChartQA, InfoVQA on the *same* fixed 1,000-sample seed subsets, greedy decoding, 16-bit, before and after. A drop >1.5–2 points on any = regression; raise the replay ratio.
- **Minimal before/after table:**

| Benchmark | Baseline | After PoC (target) |
|---|---|---|
| TableVQABench (mean) | 50.1 | >53 |
| — VWTQ | (measure) | ≥ baseline |
| — VWTQ-Syn | (measure) | ≥ baseline |
| — VTabFact | (measure) | ↑ |
| — FinTabNetQA | (measure) | ↑↑ |
| OCRBench | 79.2 | ≥ 78 |
| DocVQA | 89.3 | ≥ 88 |
| ChartQA | 70.8 | ≥ 70 |
| InfoVQA | 62.3 | ≥ 61 |

### 5. Starter resources
- **Benchmark:** TableVQA-Bench — arXiv 2404.19205; HF `terryoo/TableVQA-Bench`; `github.com/naver-ai/tablevqabench`.
- **Datasets:** MMTab `SpursgoZmy/MMTab` (arXiv 2406.08100); Visual-TableQA `AI-4-Everyone/Visual-TableQA` (arXiv 2509.07966); TABLET `alonsoapp/TABLET-Small`/`-Medium`/`-Large`/`-test` (arXiv 2509.21205); `cmarkea/table-vqa`.
- **Models/methods:** Table-LLaVA `github.com/SpursGoZmy/Table-LLaVA`; TALENT arXiv 2510.07098; SynTab-LLaVA (CVPR 2025); MirageTVQA arXiv 2511.17238.
- **Tooling:** Unsloth Qwen3.5-0.8B Vision Colab notebook (`unslothai/notebooks`, `Qwen3_5_(0_8B)_Vision.ipynb`); LLaMA-Factory `github.com/hiyouga/LLaMA-Factory` (lists Qwen3-VL); ms-swift `github.com/modelscope/ms-swift` (lists Qwen3-VL); VLMEvalKit `github.com/open-compass/VLMEvalKit`; HF PEFT.
- **Forgetting:** Biderman et al., "LoRA Learns Less and Forgets Less," TMLR 2024 (arXiv 2405.09673); SSR rehearsal arXiv 2403.01244.

## Recommendations
1. **Stage 1 (day 1) — establish control.** Reproduce the baseline TableVQABench sub-domain scores and the fixed 1,000-sample subsets of the four protected benchmarks in VLMEvalKit. Confirm where the 50.1 mean actually leaks (expect FinTabNetQA + VTabFact lowest).
2. **Stage 2 (days 2–3) — run the recipe.** Build the ~10K mixture, run bf16 LoRA (1 epoch, LR 1e-4, vision frozen, r=16). Evaluate. If TableVQA improves but a protected benchmark drops >2 pts, raise replay to 35–40% and/or drop to LR 5e-5; if no movement on tables, add a second epoch or up-weight FinTabNet/VTabFact data.
3. **Stage 3 — close the numeric gap.** If FinTabNetQA still lags, add Stage-2 alternative (rationale distillation from Qwen3.5-2B). If still capped, you have hit the 0.8B arithmetic ceiling — fall back to TALENT-style narration at inference for numeric questions only.
4. **Ship/stop thresholds:** ship (and merge the adapter) if TableVQABench mean >53 **and** no protected benchmark down >1.5 pts. Otherwise keep the adapter swappable and iterate; do not merge.

## Caveats
- **QLoRA 4-bit is contraindicated for Qwen3.5** (Unsloth: "not recommended… due to higher than normal quantization differences"). Use bf16 LoRA — the 0.8B fits in ~3GB anyway.
- The Visual-TableQA result is a hard warning that fine-tuning can *hurt* an already-strong model on VTabFact/VWTQ (Qwen2.5-VL-7B regressed there). Replay and conservative LR are non-negotiable for this user's strong baseline.
- ms-swift had a LoRA-ineffective bug on Qwen3-VL MoE (issue #6207); even on the dense 0.8B, verify the adapter actually changes outputs before trusting eval deltas. Also confirm LoRA target-module compatibility with the hybrid Gated-DeltaNet/MoE backbone empirically (the user's own caution).
- TableVQABench is only 1,500 items (the literature itself notes "moderate" statistical power); small swings may be noise — use fixed seeds and always report per-subdomain, not just the mean.
- Exact-match normalization (units, currency, "thousand") is a recurring silent failure mode; align training output format and the eval matcher to avoid mistaking format mismatches for reasoning errors.
- Some figures here are model-specific and from larger models (e.g., the Qwen2.5-VL-7B Visual-TableQA deltas, the TABLET 47.8% on a 7B backbone); they indicate *direction and risk*, not guaranteed 0.8B magnitudes. Treat the target table as hypotheses to validate in Stage 1.
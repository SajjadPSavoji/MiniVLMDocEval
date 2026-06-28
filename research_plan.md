# Research Plan

> Companion to [task.md](task.md). That file holds the full brief; this file stays compact and focuses on *strategy* — what we must nail to deliver a strong submission.

## 1. The Task in Brief

Evaluate **small VLMs (<1B params)** on **document understanding**, then propose a **justified strategy** to improve the best one.

- **Part 1 — Evaluation:** Pick a few sub-1B VLMs, select/design a document-understanding benchmark (from VLMEvalKit or our own), and compare them on metrics that go *beyond accuracy*.
- **Part 2 — Improvement:** Use our own results to diagnose where the best model fails, then propose a concrete, literature-backed strategy to fix it.
- **Deliverables:** reusable eval pipeline + benchmark scripts, a comparison table with written analysis, and a technical report (PDF).
- **Constraints:** reproducible, free/own GPUs, 5 days, must defend it live in an interview.

**Underlying thesis to answer:** *Can sub-1B VLMs be a reliable foundation for document-understanding domain adaptation?* The report should answer this directly with evidence.

## 2. Strategic Parts to Nail

The task repeats "justify / rationale / with reference to literature" everywhere — it tests **research judgment**, not benchmark-running. Ranked by leverage:

### Tier 1 — where we win or lose
1. **Knowledge-gap analysis (the centerpiece).** Diagnose *where and why* the best model fails — by task type, modality, reasoning skill — using *our own numbers* as evidence. Specific failure modes (e.g. multi-hop table reasoning, dense small-font text), not vague "it struggles with hard questions."
2. **Metrics beyond accuracy.** The brief explicitly names **calibration** and **robustness to domain terminology** — twice. At minimum: one calibration metric (e.g. ECE / confidence-vs-correctness) + one robustness probe. Accuracy-only = ignored instruction.
3. **Improvement strategy tied to the gap + literature.** Connect directly to the diagnosed gap; weigh fine-tune vs. PEFT/LoRA vs. distillation; pick one; justify with citations; address document-domain challenges (high-res input, layout, OCR-free vs OCR-assisted).

### Tier 2 — table stakes (must be solid, won't differentiate)
4. **Reusable eval pipeline.** One script that loads *any* candidate model and scores it with minimal changes — extensibility/config-driven design is the real bar.
5. **Reproducibility.** Pinned env, fixed seeds/decoding params, documented hardware. "Another engineer runs it and gets the same numbers."
6. **Model + benchmark selection rationale.** A clear justified paragraph each. Verify param counts — the <1B constraint is hard.

### Tier 3 — don't over-invest
7. Software-stack justification (tie to edge/sub-1B deployment) and expected outcomes — a few sentences each.

## 3. Traps to Avoid
- **Scope creep on models.** 3–4 well-chosen models analyzed deeply > 8 shallow ones.
- **Over-building the PoC.** Improvement is a *proposal*; a small LoRA demo on a slice or a runnable skeleton is the realistic 5-day target. State our chosen scope explicitly rather than silently under-delivering.
- **Undefendable claims.** It must survive live questioning — don't write anything we can't justify out loud.

## 4. Execution Steps

### Step 1 — Lock the model shortlist (no GPU; pure desk research)

Everything downstream (benchmark, pipeline loader, metrics) depends on *which models we commit to*, so this goes first. Output is a small table of **3–4 sub-1B VLMs** plus a couple of alternates, each with a one-line rationale. For every candidate verify:

1. **Params genuinely < 1B** — read from the model card/config, not the name. Count the vision encoder too.
2. **Real, loadable HF checkpoint** — and note *how* it loads (transformers `AutoModel` vs. custom `trust_remote_code`, processor class, chat template). This is what most often breaks on Colab.
3. **Can do document/OCR-style VQA** — accepts a reasonable input resolution; reject pure 224px natural-image captioners (document images are dense).
4. **VLMEvalKit support status** — already has a wrapper, or we write our own.

**Evaluation framework:** we use **[VLMEvalKit](https://github.com/open-compass/VLMEvalKit)**. It supports open-source HuggingFace models directly (220+ LMMs), and plugging in a model means implementing a single `generate_inner()` method on a model class — the kit handles data download, preprocessing, inference, and metrics. Candidate sub-1B families *already supported* there: **SmolVLM / SmolVLM2** (256M, 500M), **InternVL2 / 2.5 / 3 -1B** (~0.9B, borderline — verify), **LLaVA-OneVision-0.5B**, **Moondream**. Note: several "small" models exceed 1B once the encoder is counted (Moondream2 ≈1.8B, Qwen2-VL-2B, PaliGemma-3B) — flag and likely exclude.

**Big-tech reality check (verified):** Google (Gemma 3 *and* Gemma 4) and Microsoft (Phi-4) ship **no** vision model under 1B — Gemma's floor is ~2B (E2B), Phi-4-multimodal is 5.6B; Kimi-VL is a 16B MoE; OpenAI ships no open VLM. The sub-1B VLM space belongs to efficiency-focused labs + OCR specialists. (Microsoft's only sub-1B vision entry is **Florence-2**, a task-token OCR/grounding model, not a chat VLM.)

#### How the shortlist was selected

We combined a **principled** and a **practical** lens so the set is defensible on both:

1. **Filter the HF Hub** by the `Image-Text-to-Text` task tag (the tag the candidate VLMs actually carry) and restrict to **≤1B total parameters**.
2. **Sort by downloads** (most-used on the platform) — this grounds selection in real-world adoption, not just paper claims, and gives the report a practitioner's perspective.
3. **Keep generalist VLMs only** — drop OCR/parser specialists (Florence-2, GOT-OCR, Nemotron-Parse, etc.), which output transcription rather than answers and don't fit a VQA comparison.
4. **Remove Hub noise** — test fixtures (`trl-internal-testing/tiny-*`, random-weight repos) and quantized re-packages (`-GGUF`) that pollute download-sorted lists.
5. **Diversify across families** — from the surviving popular generalists, pick one strong representative per distinct architecture/lab so the comparison isolates *design* differences rather than re-testing one recipe.

#### Locked shortlist — 6 generalist VLMs

Diversified across **6 labs**, **5 encoder paradigms** (SigLIP / FastViTHD / SigLIP2 / InternViT / Qwen-native), **4 backbone lineages** (SmolLM2 / Qwen2-2.5 / LFM2 / gated-delta-MoE), and **distinct resolution strategies** — the axis that matters most for dense document images.

| # | Model (HF repo) | Lab | Total | Vision encoder + LLM backbone | Image / resolution handling | HF downloads/mo | Loading + license | VLMEvalKit support |
|---|---|---|---|---|---|---|---|---|
| 1 | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | Hugging Face | 0.5B | SigLIP + SmolLM2-360M | Image-splitting + pixel-shuffle token compression | ~695k | native · Apache-2.0 | ✅ built-in (`smolvlm_series`) |
| 2 | `apple/FastVLM-0.5B` | Apple | 0.8B | **FastViTHD** (hybrid CNN-ViT) + Qwen2-0.5B | High-res native, *few* tokens (encoder-side compression) | — (Apple) | `trust_remote_code` · apple-amlr | ❌ **custom wrapper** |
| 3 | `LiquidAI/LFM2.5-VL-450M` | Liquid AI | 0.45B | SigLIP2-86M + **LFM2-350M** (liquid/hybrid) | Native ≤512², patch-tiling for larger; tunable speed/quality | ~772k | native · LFM open license | ⚠️ series in (`lfm2vl_series`); add 450M key |
| 4 | `OpenGVLab/InternVL3-1B-hf` | OpenGVLab | 0.9B | **InternViT-300M** + Qwen2.5-0.5B | Dynamic high-res tiling (448² tiles) | ~315k | native (`-hf`) · MIT | ✅ built-in (`internvl3`) |
| 5 | `llava-hf/llava-onevision-qwen2-0.5b-ov-hf` | LMMs-Lab | 0.9B | SigLIP + Qwen2-0.5B | AnyRes multi-crop | ~964k | native · Apache-2.0 | ✅ built-in (`llava_series`) |
| 6 | `Qwen/Qwen3.5-0.8B` | Alibaba | 0.8B | Qwen-native vision enc. + **gated-delta/MoE** | Native dynamic resolution (NaViT-style) | ~2.48M | native (`AutoModelForMultimodalLM`) · Apache-2.0 | ❌ **custom wrapper** (registry has only large Qwen3.5 MoE *text* models) |

**Implementation implication:** 3 models run out-of-the-box (SmolVLM2, InternVL3, LLaVA-OV), LFM2.5-VL-450M needs a one-line config key, and **FastVLM + Qwen3.5-0.8B need a custom `generate_inner()` wrapper** — that's the bulk of the pipeline integration work, and the two `trust_remote_code`-style models are also where Colab friction concentrates. Budget Step-3 effort accordingly.

**Why this makes the comparison robust:** every model brings a different answer to the two questions that decide document performance — *how do you encode a high-resolution page* (encoder + resolution strategy) and *how much language capacity backs the reading* (backbone). Holding the size budget roughly fixed (0.45–0.9B) turns those differences into the independent variables, while download counts keep at least one foot in real deployment reality.

Notes: prefer `-hf` variants (native `transformers`, no `trust_remote_code`) wherever they exist. Excluded as >1B once the encoder is counted: Moondream2 (~1.8B), Qwen2-VL-2B, PaliGemma-3B, Gemma 4 E2B (~2B). Specialists (Florence-2, GOT-OCR, Nemotron-Parse) excluded as non-generalist. Param counts and download figures are estimates pending exact `config.json`/safetensors + live Hub verification. If Colab compute gets tight, the core comparison can shrink to #1–4.

*Output of Step 1:* the table above, locked with exact param counts + verified load signatures → feeds the "Model Survey & Selection" deliverable.

### Step 2A — Benchmark suite selection (no GPU; design)

From VLMEvalKit's document-relevant benchmarks (DocVQA, InfoVQA, ChartQA, TextVQA, OCRVQA, OCRBench/v2, TableVQABench, AI2D, ChartQAPro, CharXiv, OmniDocBench, WildDoc, …), we select **5** that form a *diagnostic capability ladder* rather than five interchangeable accuracy numbers. All are locally scorable (no leaderboard submission server), which the reproducibility requirement demands.

#### The 5 chosen benchmarks — metadata & stats

| # | Benchmark (VLMEvalKit key) | Source / image type | Split size (≈) | Answer format | Core sub-skill probed | Reasoning depth | Metric |
|---|---|---|---|---|---|---|---|
| 1 | **OCRBench** | Mixed: scene, scanned docs, handwriting, formulas (5 sub-tasks) | 1,000 Q | Short text string | **Raw text recognition** — *can it read at all?* | Recognition only | Accuracy (score/1000) |
| 2 | **DocVQA_VAL** | Scanned industry documents — forms, letters, reports (UCSF library) | ≈5,349 Q / ≈1,286 imgs | Extractive span | Read **+** locate answer in dense printed text | Single-hop extraction | **ANLS** |
| 3 | **ChartQA_TEST** | Real charts — bar/line/pie (Statista etc.); 1,250 human + 1,250 machine | 2,500 Q | Number / short phrase | Chart data extraction **+** arithmetic | Numeric / multi-step | **Relaxed acc** (±5%) |
| 4 | **InfoVQA_VAL** | High-res web infographics, extreme aspect ratios | ≈2,801 Q / ≈500 imgs | Span / number | Joint **text + graphics + layout** reasoning | Multi-hop + arithmetic | **ANLS** |
| 5 | **TableVQABench** *(optional 5th, included)* | Rendered table images (VWTQ 750 / VWTQ-Syn 250 / VTabFact 250 / FinTabNetQA 250) | 1,500 Q | Cell value / yes-no | Structured **tabular** lookup + aggregation | Row/col cross-reference | Accuracy / exact-match |

*(Counts are the standard published splits; exact loaded sizes verified at runtime.)*

#### Why these five — rationale

- **It's a diagnostic ladder, not a grab-bag.** OCRBench is the *anchor*: if a model fails DocVQA, OCRBench disambiguates a **reading** failure from a **reasoning** failure. That single contrast is what turns Part 2's gap analysis into evidence instead of speculation — the highest-value design decision in this step.
- **Escalating difficulty & reasoning type:** recognition (OCRBench) → single-hop extraction (DocVQA) → numeric reasoning (ChartQA) → multi-hop layout reasoning (InfoVQA) → structured tabular reasoning (TableVQA). Where each small model breaks *on this gradient* is the per-skill profile we need.
- **Built-in metric diversity** (Accuracy / ANLS / Relaxed-acc) satisfies "go beyond generic VQA accuracy" *structurally*, before calibration/robustness are added in Step 2B.
- **Image-diversity stresses different encoders** — exactly the axis our model set varies on (Step 1). Dense scanned text (DocVQA) rewards high native resolution (FastVLM/InternVL tiling); extreme-aspect-ratio infographics (InfoVQA) punish fixed-square encoders; rendered tables (TableVQA) probe fine grid alignment. This couples the benchmark suite to the model-diversity story.
- **Practical / reproducible:** all five are in VLMEvalKit, all locally scorable, all single-image (no video/multi-turn complexity), and small enough to subsample for free GPUs.

#### Practical experiment note (Colab budget)

Full suite ≈ **13.1k questions** × 6 models ≈ 79k generations — too heavy for free Colab end-to-end. Plan: run a **fixed-seed stratified subset (≈500 Q/benchmark)** for the headline comparison table (reproducible, ~15k generations total), and optionally run full splits for the single best/worst model to confirm the subset is representative. Subset size + seed are pinned in config for reproducibility.

**Deliberately excluded:** TextVQA/OCRVQA (natural-image scene text — drifts from "documents"), AI2D (science diagrams — different domain), and heavy parsing benchmarks (OmniDocBench, WildDoc — suit OCR *specialists*, not generalist VQA, and are compute-heavy).

*Output of Step 2A:* the locked 5-benchmark suite above → feeds Step 2B (metrics) and the evaluation pipeline.

### Step 2B — Metrics

**Approach:** start with the **native VLMEvalKit metrics** (correctness only), run the full comparison, then analyze results and failure modes *before* deciding whether to invest in additional metrics. We do **not** build new metrics up front — the baseline numbers tell us where they're worth adding.

#### Metrics we will compute now (native to VLMEvalKit)

VLMEvalKit is *generation-based*: each model emits an answer string that is scored against ground truth. The metrics for our 5 benchmarks:

| Metric | What it represents | Range | Our benchmarks |
|---|---|---|---|
| **Accuracy / exact-match** | Answer string exactly correct after normalization (default for yes-no & multi-choice). | 0–1 | TableVQABench (per-subset) |
| **OCRBench score** | Accuracy summed across OCRBench's 5 OCR sub-tasks, reported /1000. | 0–1000 | OCRBench |
| **ANLS** (Avg. Normalized Levenshtein Similarity) | `1 − normalized edit distance`, averaged, with a 0.5 floor → 0. Tolerates minor OCR/spelling slips when *reading* text. | 0–1 | DocVQA_VAL, InfoVQA_VAL |
| **Relaxed accuracy** | Correct if within **±5%** of the numeric ground truth (exact match for text); accommodates reading values off a chart. | 0–1 | ChartQA_TEST |

The suite already spans **3 distinct correctness metrics** (Accuracy / ANLS / Relaxed-acc), partly satisfying "go beyond generic VQA accuracy" through suite design alone.

**Answer-extraction fallback (decide at run-time):** small models are *chatty* and may wrap the answer in a sentence, breaking naïve string matching. VLMEvalKit can use an **LLM judge** (OpenAI key or local LMDeploy model) to extract/grade the final answer. We'll enable it *only if* heuristic scoring is visibly under-counting correct-but-verbose answers — to keep the default run reproducible and API-free.

#### Deferred ideas — revisit after baseline analysis

VLMEvalKit has **no** built-in calibration or robustness metrics. Both are explicitly valued by the task and are good ideas, but we **defer them by design** until baseline results + failure modes justify the investment:

- **Calibration (ECE):** capture model confidence — via *verbalized confidence* (answer + 0–100%, uniform across all 6 models) or *sequence logprobs* (`generate(output_scores=True)`, more precise but uneven across `trust_remote_code` models) — then bin into Expected Calibration Error. *Worth adding if* models are accurate-but-overconfident, or if calibration separates otherwise-close models.
- **Robustness (Δ-accuracy):** build perturbed slices — *image degradation* (blur/JPEG/downscale, directly probing the Step-1 resolution-strategy differences) and/or *domain-terminology paraphrase* — and report the accuracy drop vs. clean. *Worth adding if* the gap analysis points to brittleness (e.g. a model strong on clean DocVQA but fragile on low-res scans).

This sequencing is itself a defensible choice: **let the data decide which "beyond-accuracy" axis matters**, rather than building metrics speculatively.

*Output of Step 2B:* native metric set locked for the first run; calibration + robustness specced but parked pending baseline analysis.

### Step 3 — Pipeline: add VLMEvalKit as a pinned dependency *(in progress)*

**Decisions (settled):** use VLMEvalKit as a **dependency, not a fork**; **deterministic decoding** (greedy); **subsample** benchmarks for a light first iteration (full runs later); per-model **prompt/chat templates** in custom wrappers. First sub-step = install the engine and smoke-test the 3 already-supported models before writing any wrappers.

**Done — engine vendored & importable:**
- **VLMEvalKit cloned at a pinned commit** by `setup.sh` into `external/VLMEvalKit` (gitignored; a local clone doubles as a read-only source reference), pinned to upstream `2cf2a36c6e79b51faf676a7011b3a0f5b579814d` (HEAD of main @ 2026-06-26, just ahead of `v0.3rc1`), **never modified** (see `external/README.md`). *(We tried vendoring/committing the source, but VLMEvalKit ships its own `.gitignore` that drops ~35 force-tracked files like `api/bailingmm.py` → broken committed copy. Cloning gets the complete tree.)*
- **Editable install** (`pip install --no-deps -e external/VLMEvalKit`), *not* a wheel: VLMEvalKit's `setup.py` `find_packages()` drops ~20 subdirs lacking `__init__.py` (e.g. `megabench/parsing`), so a built **wheel is broken** — editable uses the source tree and matches upstream's own install docs. *(Holds on Colab too.)*
- **Colab/Linux only:** we dropped local-mac running (eval needs CUDA anyway). `decord` (video, imported eagerly by `dsrbench`/`stibench`) has a Linux wheel, so install is the plain `pip install -r requirements.txt` — no mac stub needed. (A mac-arm64 `decord` stub existed briefly for local import; removed once we committed to Colab-only.)
- **Reproducibility artifacts:** `setup.sh` (install vendored copy: deps → editable), `requirements.txt` (records the recipe), `external/README.md` (vendoring provenance).
- **Verified:** `import vlmeval` OK; registry = **552 models**; our 3 freebies present — `SmolVLM2-500M`, `InternVL3-1B`, `llava_onevision_qwen2_0.5b_si`.

**Local vs. Colab verdict (tested):** the VLMEvalKit *eval pipeline* must run on **Colab (CUDA)**, not mac, because (1) **88 wrapper files hardcode CUDA** (`device_map="cuda"`, `.to("cuda")`) and this mac has `cuda: False` — fixing = forking; (2) `run.py` has **no `--limit`** (runs whole datasets); (3) even patched, MPS is **too slow** (~20s/gen × ~15k-gen subset ≈ 80+h vs. a few hours on a T4). **However, the models run fine on mac:** a direct-transformers test loaded `SmolVLM2-256M` on **MPS** and correctly read a document image (`'INVOICE 2026'`, load 26s / gen 20s). → **mac = wrapper-logic development; Colab = actual eval.** Design consequence: write our custom **FastVLM/Qwen3.5 wrappers device-agnostic** (`cuda→mps→cpu`) so their plumbing is smoke-testable locally before Colab.

**Workflow + Colab scaffolding (ready to run):** repo on GitHub (`SajjadPSavoji/MiniVLMDocEval`, public). **Script-first, one notebook:** `notebooks/colab.ipynb` is a thin *terminal* (GPU-check + clone, `setup.sh`, then a Run cell = `git pull && python scripts/...`); all logic is `.py`. `scripts/smoke_test.py` loads 1 model + builds 1 dataset + runs N samples (mirrors `inference.py`, optional subset scoring). **Loop:** Claude edits `.py` locally → push (Claude via `gh`) → user runs in the Colab **T4 terminal** (`git pull && python scripts/...`) → pastes output. (Claude's Bash runs locally only — cannot drive the Colab kernel/terminal; git is the bridge.)

**✅ Smoke test PASSED on Colab T4 (2026-06-26):** `SmolVLM2-500M` × `OCRBench` × 5 samples ran end-to-end — `cuda=True`, model + 114MB dataset downloaded, inference produced answers, scorer ran (4/5 text-recognition). Confirmed: VLMEvalKit's built-in wrappers default to **greedy** (`do_sample=False`) — matches our determinism decision. The setup recipe (clone @ pinned commit → editable install) works on Colab.

*Next:* (a) confirm the other 2 built-ins run (`InternVL3-1B`, `llava_onevision_qwen2_0.5b_si`) — same one-line loop; (b) write the 2 custom wrappers (FastVLM, Qwen3.5) + add the LFM2.5-VL-450M config key; (c) build the config-driven runner + seeded ~500/benchmark subset + results aggregation for the full 6×5 comparison table.

### Step 3.1 — Inference efficiency & precision

Free-T4 inference was the bottleneck (full datasets ≈ 10 h). We capped each dataset to **N=1000** (fixed seed 42) and applied the speedups below. The GPU was **compute/power-bound** (100% util, pinned at the 70 W cap) with **12 GB VRAM free** — so the wins came from *more efficient work*, not from filling memory (parallel jobs / bigger batches don't help under a power cap; vLLM batching isn't supported for these wrappers).

| Lever | Change (our code, no fork) | Measured effect | Consequence / status |
|---|---|---|---|
| **A — generation cap** | `max_new_tokens` 2048→**128** via `model.kwargs` | **Accuracy-neutral** (identical scores, see ablation); ~0 latency on SmolVLM (137 vs 138 s) — it emits EOS early | Kept as a cheap **guardrail** vs. runaway generation |
| **B — no per-sample `empty_cache`** | removed `torch.cuda.empty_cache()` from the loop | Small (~5–15%, not isolated) — drops a per-iteration GPU sync | Safe at batch-1 with sub-1B models |
| **C — fp16 (selective)** | downcast **fp32** wrappers → fp16 (`model.model.half()`); targets SmolVLM only | **~1.8× faster**: OCRBench n=100 **209 s → 116 s** | Validated; bf16/fp16 models left at native precision |

**Ablations / sanity checks (SmolVLM2-500M, n=100):**
- **fp16 vs fp32** (OCRBench): **209 s → 116 s = 1.8×**; score 57% → 62% on the *same* 100 (precision variation, ≈1 SE at n=100; shrinks at n=1000). fp16 produced valid outputs (no NaN/crash — inputs auto-cast).
- **max_new_tokens 128 vs 2048** (DocVQA + InfoVQA): **identical** scores (DocVQA 76.597, InfoVQA 32.933) *and* identical time (137 vs 138 s) → the 128 cap truncates nothing and is the real-world latency floor only when a model over-generates.

**Precision for comparability (our position):** precision is a confound, so it should be **held constant** — the target common precision is **16-bit (fp16)**, which is also the realistic edge-deployment format (no one deploys sub-1B VLMs in fp32). Current state: SmolVLM **fp16**, InternVL **bf16**, LLaVA-OV **fp16** — all 16-bit. We do *not* force InternVL bf16→fp16 because fp16's narrow range can destabilize a bf16-native model (NaNs); strict uniform-fp16 would require validating InternVL's fp16 stability first. The **per-model precision is reported** in the methodology, and we log per-pair `seconds`/`s_per_sample` to support the latency comparison.

*Reproducibility:* `scripts/efficiency_ablation.py` re-runs both ablations and prints the score + gen-time tables; the Colab notebook runs it right after the smoke test.

## 5. Part 1 — Results & Analysis

### 5.1 Experimental setup (summary)

- **Models (3 of 6 shortlisted):** the VLMEvalKit-native built-ins — `InternVL3-1B` (InternViT-300M + Qwen2.5-0.5B), `LLaVA-OneVision-0.5b-ov-hf` (SigLIP + Qwen2-0.5B), `SmolVLM2-500M` (SigLIP + SmolLM2-360M). Custom-wrapper models (FastVLM, Qwen3.5-0.8B, LFM2.5-VL-450M) are deferred to a later iteration.
- **Benchmarks (5):** OCRBench, DocVQA (val), ChartQA (test), InfoVQA (val), TableVQABench — **N = 1000 fixed-seed (42) subsample** each (OCRBench has exactly 1000 → evaluated in full).
- **Metrics, normalized to 0–100:** OCRBench = accuracy (correct/N·100); DocVQA & InfoVQA = ANLS·100; ChartQA = relaxed accuracy (±5%); TableVQA = accuracy (mean over its 4 sub-domains). Each column is a single metric, so columns are internally comparable.
- **Inference:** single NVIDIA **T4** (Colab); **greedy** decoding (`do_sample=False`); `max_new_tokens=128` (ablated accuracy-neutral, §3.1); **16-bit** precision (SmolVLM/LLaVA fp16, InternVL bf16). Per-pair generation wall-time recorded.

### 5.2 Results

**Accuracy (0–100; higher better). Bold = best per column.**

| Model | ChartQA | DocVQA | InfoVQA | OCRBench | TableVQA | **Mean** |
|---|---|---|---|---|---|---|
| **InternVL3-1B** | **68.8** | **80.8** | **54.2** | **79.4** | 33.5 | **63.3** |
| LLaVA-OV-0.5b | 60.0 | 70.4 | 39.0 | 60.2 | **34.4** | 52.8 |
| SmolVLM2-500M | 60.4 | 67.8 | 27.4 | 61.0 | 30.3 | 49.4 |

**Latency (seconds/sample on T4; lower better).**

| Model | ChartQA | DocVQA | InfoVQA | OCRBench | TableVQA | **Mean** |
|---|---|---|---|---|---|---|
| InternVL3-1B | 0.69 | 1.47 | 1.84 | 1.46 | 0.47 | **1.19** |
| SmolVLM2-500M | 0.72 | 0.96 | 0.70 | 0.83 | 0.66 | 0.77 |
| LLaVA-OV-0.5b | 0.46 | 1.35 | 0.83 | 0.95 | 0.57 | 0.83 |

### 5.3 Analysis

**By benchmark (the diagnostic ladder):**
- **OCRBench (raw reading):** InternVL **79.4** vs ~60 — a ~19-pt gap. Reading fidelity scales with vision-encoder capacity; the two 0.5B models lag well behind the InternViT-300M encoder.
- **DocVQA (read + locate):** 80.8 vs 68–70. ANLS is edit-distance tolerant, so all three "read" documents adequately; InternVL leads.
- **ChartQA (numeric reasoning):** the **tightest** spread (60–69) — chart QA is the least encoder-bound task.
- **InfoVQA (layout + multi-hop):** the **most discriminating** task — 27 → 39 → 54, the widest spread. High resolution and layout reasoning separate the models; SmolVLM's aggressive token compression is the clear liability (27.4).
- **TableVQA (structured tables):** the **equalizer** — all 30–34, and InternVL is *not* best (LLaVA 34.4 ≥ InternVL 33.5). Encoder strength does not transfer to structured-table reasoning.

**By model:**
- **InternVL3-1B** — best on 4/5, mean **63.3**. Strong reading/document/chart/infographic understanding; weak on tables (33.5). Cost: slowest (mean 1.19 s/sample).
- **LLaVA-OV-0.5b** — mean 52.8; mid-tier; best on tables; AnyRes multi-crop helps layout vs SmolVLM (InfoVQA 39 vs 27).
- **SmolVLM2-500M** — mean 49.4; **fastest** (0.77); competitive on charts/OCR; weakest on layout-heavy InfoVQA.

**Accuracy–latency tradeoff.** InternVL buys its ~+11–14-pt mean-accuracy lead with ~**1.5×** the per-sample latency. Both effects share a cause — **dynamic high-resolution tiling** (more image tokens ⇒ more detail *and* more compute). For a fixed edge-latency budget, the 0.5B models are ~1.5× faster at ~10–14 pts lower accuracy. (InternVL's TableVQA is its *fastest* benchmark, 0.47 s — rendered tables tile into fewer patches than infographics.)

### 5.4 Key findings

1. **Within sub-1B, document accuracy tracks vision-encoder capacity + input resolution:** InternViT-300M + dynamic tiling > SigLIP + AnyRes > SigLIP + aggressive compression.
2. **InfoVQA is the most discriminating benchmark; TableVQA the least** (a shared failure mode).
3. **Structured-table reasoning is a universal sub-1B weakness:** even the best model drops from ~81% (DocVQA) to ~33% (TableVQA) — a ~48-pt collapse.
4. **Accuracy and latency are coupled through resolution/tiling** — the central tension for edge deployment.

### 5.5 Conclusion (Part 1)

Returning to the thesis — *can sub-1B VLMs be a reliable foundation for document-understanding domain adaptation?* — the evidence supports a **conditional yes**: for **text-centric reading and document QA**, `InternVL3-1B` reaches ~79–81% (OCRBench/DocVQA), a credible foundation; but for **structure- and layout-heavy reasoning** it is **not yet reliable** (TableVQA ~33%, InfoVQA ~54%). **`InternVL3-1B` is the strongest foundation**, and its concrete gaps — **tables** and **infographics** — together with its **latency premium** are the targets for the Part-2 improvement strategy.

### 5.6 Limitations / threats to validity

- **Subsampling:** N=1000 fixed-seed subset (not full splits) → sampling variance, especially for ANLS/relaxed-acc; reported as point estimates without confidence intervals.
- **Single run / single seed:** no across-seed variance estimate; small differences (e.g. SmolVLM vs LLaVA on ChartQA/OCRBench) should be read as ties.
- **Mixed 16-bit precision:** fp16 (SmolVLM, LLaVA) vs bf16 (InternVL) — comparable but not identical; documented per model (§3.1).
- **Heuristic scoring (no LLM judge):** chatty/verbose outputs may be under-credited; partially mitigated by VLMEvalKit's answer extraction.
- **Hardware/precision-specific latency:** single T4 (Turing); bf16 is not native on Turing, so InternVL's latency would likely improve on Ampere/Ada — latency rankings are indicative, not absolute.
- **Partial model set:** 3 of 6 shortlisted models evaluated; FastVLM / Qwen3.5 / LFM2.5-VL pending (custom wrappers).

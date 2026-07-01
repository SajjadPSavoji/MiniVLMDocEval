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

- **Models (all 6 shortlisted):** three VLMEvalKit-native built-ins — `InternVL3-1B` (InternViT-300M + Qwen2.5-0.5B), `LLaVA-OneVision-0.5b-ov-hf` (SigLIP + Qwen2-0.5B), `SmolVLM2-500M` (SigLIP + SmolLM2-360M) — plus three custom-wrapper models — `Qwen3.5-0.8B` (Qwen-native enc. + gated-delta/MoE), `FastVLM-0.5B` (FastViTHD + Qwen2-0.5B), `LFM2.5-VL-450M` (SigLIP2-86M + LFM2-350M). Total params span 0.45B–0.9B.
- **Benchmarks (5):** OCRBench, DocVQA (val), ChartQA (test), InfoVQA (val), TableVQABench — **N = 1000 fixed-seed (42) subsample** each (OCRBench has exactly 1000 → evaluated in full).
- **Metrics, normalized to 0–100:** OCRBench = accuracy (correct/N·100); DocVQA & InfoVQA = ANLS·100; ChartQA = relaxed accuracy (±5%); TableVQA = accuracy (mean over its 4 sub-domains). Each column is a single metric, so columns are internally comparable.
- **Inference:** Colab GPU, **greedy** decoding (`do_sample=False`); `max_new_tokens=128` (ablated accuracy-neutral, §3.1); **16-bit** precision (fp16 for SmolVLM/LLaVA/FastVLM, bf16 for InternVL/Qwen3.5/LFM2.5-VL). Per-pair generation wall-time recorded. **Hardware caveat:** the three built-ins ran in one T4 session; the three custom models ran in later sessions on a mix of T4/L4 (Colab-assigned), so **accuracy is fully comparable (hardware-independent) but cross-model latency is indicative only** (see §5.6).

### 5.2 Results

**Accuracy (0–100; higher better). Sorted by mean. Bold = best per column.**

| Rank | Model | Params | ChartQA | DocVQA | InfoVQA | OCRBench | TableVQA | **Mean** |
|---|---|---|---|---|---|---|---|---|
| 1 | **Qwen3.5-0.8B** | 0.8B | 70.8 | **89.3** | **62.3** | 79.2 | **50.1** | **70.3** |
| 2 | InternVL3-1B | 0.9B | 68.8 | 80.8 | 54.2 | **79.4** | 33.5 | 63.3 |
| 3 | LFM2.5-VL-450M | 0.45B | **73.1** | 77.2 | 41.3 | 67.8 | 40.2 | 59.9 |
| 4 | LLaVA-OV-0.5b | 0.9B | 60.0 | 70.4 | 39.0 | 60.2 | 34.4 | 52.8 |
| 5 | SmolVLM2-500M | 0.5B | 60.4 | 67.8 | 27.4 | 61.0 | 30.3 | 49.4 |
| 6 | FastVLM-0.5B | 0.8B | 44.3 | 63.6 | 32.5 | 26.2 | 17.5 | 36.8 |

**Latency (seconds/sample; lower better). Indicative — mixed T4/L4 across models (§5.6).**

| Model | ChartQA | DocVQA | InfoVQA | OCRBench | TableVQA | **Mean** |
|---|---|---|---|---|---|---|
| LFM2.5-VL-450M | 0.14 | 0.27 | 0.23 | 0.32 | 0.27 | **0.24** |
| SmolVLM2-500M | 0.72 | 0.96 | 0.70 | 0.83 | 0.66 | 0.77 |
| LLaVA-OV-0.5b | 0.46 | 1.35 | 0.83 | 0.95 | 0.57 | 0.83 |
| InternVL3-1B | 0.69 | 1.47 | 1.84 | 1.46 | 0.47 | 1.19 |
| Qwen3.5-0.8B | 0.57 | 1.44 | 1.49 | 2.22 | 0.67 | 1.28 |
| FastVLM-0.5B | 3.84 | 3.92 | 3.92 | 3.68 | 3.93 | 3.86 |

### 5.3 Analysis

**By benchmark (the diagnostic ladder):**
- **OCRBench (raw reading):** Qwen **79.2** ≈ InternVL **79.4** lead, then LFM 67.8 and the 0.5B SigLIP pair ~60. **FastVLM collapses to 26.2** — a striking outlier given its high-res FastViTHD encoder (see below). Excluding it, reading fidelity clusters 60–79 and scales with encoder + resolution.
- **DocVQA (read + locate):** the **tightest capable spread** — Qwen 89.3 leads clearly, then 77–81 (InternVL, LFM), 64–70 (LLaVA, SmolVLM, FastVLM). ANLS is edit-distance tolerant, so every model "reads" documents adequately; notably FastVLM does far better here (63.6) than on OCRBench's exact-match reading (26.2).
- **ChartQA (numeric reasoning):** **the smallest model wins** — LFM2.5-VL-450M **73.1**, ahead of Qwen 70.8 and InternVL 68.8. Chart QA is the least encoder-bound task, so backbone reasoning + tunable tiling pays off more than raw encoder size.
- **InfoVQA (layout + multi-hop):** the **most discriminating** task — 27 → 62, the widest capable spread. Native dynamic resolution wins decisively (Qwen 62.3, InternVL 54.2); SmolVLM's aggressive token compression is the clear liability (27.4).
- **TableVQA (structured tables):** **no longer an equalizer.** Qwen **50.1** breaks away from the ~30–34 pack that the three built-ins suggested, with LFM 40.2 also above it. A stronger backbone + native resolution materially lifts structured-table reasoning — there *is* headroom here.

**By model:**
- **Qwen3.5-0.8B** — overall best (mean **70.3**), wins 3/5 (DocVQA, InfoVQA, TableVQA) and ties OCRBench. Native dynamic-resolution encoding + the largest backbone in the set generalize across every task. Mid latency (1.28 s/sample).
- **InternVL3-1B** — mean 63.3; best (by a hair) on OCRBench; strong reading/document/chart; weak tables (33.5). Among the slower capable models (1.19).
- **LFM2.5-VL-450M** — the **efficiency standout**: smallest model (0.45B), 3rd overall, *wins ChartQA*, and ~3–5× faster than the field (0.24 s/sample). Weakest on InfoVQA (41.3). Best accuracy-per-param and per-latency.
- **LLaVA-OV-0.5b** — mean 52.8; mid-tier; AnyRes multi-crop helps layout vs SmolVLM (InfoVQA 39 vs 27).
- **SmolVLM2-500M** — mean 49.4; competitive on charts/OCR; weakest on layout-heavy InfoVQA (token compression).
- **FastVLM-0.5B** — last (36.8) and **slowest** (3.86 s/sample). Its encoder-side few-token compression appears to trade away exact-reading fidelity (OCRBench 26.2), while the LLaVA-style image-token expansion inflates latency. Wrapper output-extraction was verified correct (§3 fix), so this reflects the model under our greedy/128-token doc-QA setup, not a plumbing bug — flagged for follow-up.

**Accuracy–latency tradeoff (latency indicative, mixed hardware).** **LFM2.5-VL-450M dominates the Pareto front** — top-3 accuracy at the lowest latency by a wide margin. **Qwen3.5-0.8B** buys the best accuracy at mid latency. **FastVLM is Pareto-dominated** — worst accuracy *and* slowest. The dominant cost driver is the **image-token budget** (resolution/tiling strategy), not parameter count: the 0.45B LFM is ~5× faster than the 0.8B FastVLM. *Caveat:* the custom models ran in later T4/L4 sessions, so absolute latencies are not strictly comparable to the T4-session built-ins (§5.6); the qualitative ordering (LFM fastest, FastVLM slowest) is robust to this.

### 5.4 Key findings

1. **Qwen3.5-0.8B is the strongest sub-1B document model overall** (mean 70.3, wins 3/5) — native dynamic-resolution encoding plus the largest backbone in the set generalize across reading, layout, and tables alike.
2. **Architecture beats parameter count.** The smallest model, **LFM2.5-VL-450M (0.45B), outranks both 0.9B models** and wins ChartQA; resolution strategy and backbone quality matter more than raw size within this band.
3. **Document accuracy tracks resolution strategy + backbone:** native dynamic resolution (Qwen, InternVL) and tunable tiling (LFM) > AnyRes (LLaVA) > aggressive token compression (SmolVLM); FastVLM's few-token compression is an outlier liability for *exact* reading (OCRBench 26.2 despite a high-res encoder).
4. **Structured tables are the hardest task but not a hard ceiling.** Qwen reaches **50.1** on TableVQA — well above the ~30–34 the three built-ins alone suggested — showing real headroom from a stronger backbone/resolution.
5. **Latency is architecture-driven, not size-driven:** the image-token budget dominates per-sample cost — the 0.45B LFM is ~5× faster than the 0.8B FastVLM. The accuracy/latency Pareto front is owned by **LFM (cheapest)** and **Qwen (most accurate)**.

### 5.5 Conclusion (Part 1)

Returning to the thesis — *can sub-1B VLMs be a reliable foundation for document-understanding domain adaptation?* — the full six-model evidence supports a **conditional yes**, with **`Qwen3.5-0.8B` now the strongest foundation** (mean 70.3): it reads (OCRBench 79, DocVQA 89), handles layout (InfoVQA 62), and leads tables (50.1). Its remaining gaps — **InfoVQA (62)** and **TableVQA (50)** — are the natural Part-2 improvement targets, now from a markedly higher baseline than the 3-model view implied. Two deployment-relevant refinements to the earlier (built-ins-only) read: (a) the foundation pick shifts from InternVL3-1B to **Qwen3.5-0.8B**, which dominates it on 4/5 tasks; and (b) **`LFM2.5-VL-450M` is the efficiency pick** — near-top-3 accuracy at ~5× lower latency and the smallest footprint, the better starting point if the deployment budget is latency- or memory-bound. **We choose TableVQA on Qwen3.5-0.8B as the Part-2 target** (see §6): it is the best model's own weakest task, and unlike InfoVQA it is a single, well-scoped capability with clear sub-domains to target and a same-field improvement story.

### 5.6 Limitations / threats to validity

- **Subsampling:** N=1000 fixed-seed subset (not full splits) → sampling variance, especially for ANLS/relaxed-acc; reported as point estimates without confidence intervals.
- **Single run / single seed:** no across-seed variance estimate; small differences (e.g. Qwen vs InternVL on OCRBench, 79.2 vs 79.4) should be read as ties.
- **Mixed 16-bit precision:** fp16 (SmolVLM, LLaVA, FastVLM) vs bf16 (InternVL, Qwen3.5, LFM2.5-VL) — comparable but not identical; documented per model.
- **Heuristic scoring (no LLM judge):** chatty/verbose outputs may be under-credited; partially mitigated by VLMEvalKit's answer extraction.
- **Heterogeneous latency hardware:** the three built-ins ran on a single T4; the three custom models ran in later Colab sessions on a mix of **T4/L4** (and bf16 is not native on T4-Turing). **Accuracy is unaffected** (deterministic greedy decoding), but **cross-model latency is indicative, not absolute** — the qualitative ordering (LFM fastest, FastVLM slowest) holds, but exact s/sample values are not strictly comparable across models.

## 6. Part 2 — Improvement Plan: closing the table-reading gap on Qwen3.5-0.8B

This section records the direction we chose for Part 2 and *why*, so we (and a reviewer) can follow the reasoning later. §6.1–§6.5 are the plan; **§6.6 reports what actually happened** — the first attempt failed the pre-registered gate, with a diagnosis and corrected next steps.

### 6.1 What we are trying to fix, and on which model

Part 1 left two soft spots for the best model, Qwen3.5-0.8B: **visual table questions (50.1)** and **infographics (62.3)**. We chose to work on **table questions**. Reasons: (a) it is the model's single weakest task; (b) "tables" is one clearly-defined skill with four named sub-types we can measure separately (Wikipedia-style tables, synthetic tables, true/false table facts, and dense financial tables), whereas "infographics" is a fuzzier mix; and (c) there is an honest improvement story here — even our best model still gets about **half** of table questions wrong, so there is real room to move.

We keep the target on **Qwen3.5-0.8B** even though it is a brand-new, slightly awkward-to-fine-tune model, because the whole point of Part 2 is to improve *the best foundation*, and a reviewer expects us to back our own Part-1 conclusion rather than switch to an easier-to-tune second choice.

### 6.2 Our hypothesis (in plain terms)

The model can **read** table text fine (its OCR and document scores are high). What it struggles with is **using the table's structure to find and return the right answer**: locating the correct cell by row and column in a busy table.

**We have now measured exactly where the 50.1 comes from** (the four sub-types of the table benchmark, from our existing run — no new compute):

| sub-type | what it tests | Qwen score |
|---|---|---|
| **vwtq** | lookup over real Wikipedia-style tables | **27.8** ← weakest, and the *largest* slice (~half the benchmark) |
| **vwtq_syn** | lookup over synthetic Wikipedia-style tables | **33.1** |
| vtabfact | true/false fact checks | 55.5 |
| fintabnetqa | financial tables (numbers/units) | **84.0** ← already strong |

This **corrects our first guess and the outside research reports**, which all assumed financial/numeric tables would be the weak spot. The data says the opposite: the model is *already good* at financial tables, and its real weakness is **plain lookup on Wikipedia-style tables (vwtq / vwtq_syn)**. So the revised hypothesis is: **the gap is visual row/column lookup on dense general-purpose tables, not numeric reasoning.** Practising that specific skill should raise the score, and because vwtq is the biggest slice, gains there move the average the most.

### 6.3 The approach we chose, and the decisions behind it

We will do a **light, parameter-efficient fine-tune** (LoRA — we train a small set of add-on weights and leave the original model untouched, which is cheap and easy to undo) on a **small mixture of table-reasoning examples**, while mixing in some of the model's already-strong document data so it does not forget its other skills. This fits one day on a single free GPU.

Key decisions we had to make (option chosen → why):

- **Which model to improve — Qwen3.5-0.8B vs the easier-to-tune InternVL3-1B.** Chose Qwen: it is the actual best model; improving it is the honest task. We accept the extra setup risk.
- **How to train — full fine-tune vs light add-on (LoRA).** Chose LoRA: far cheaper, fits the GPU, and (per the literature) it "forgets less," which protects the four scores we must not break.
- **Precision — memory-saving 4-bit vs plain 16-bit.** Chose 16-bit. Two of the three research write-ups warn that this specific model's internals react badly to 4-bit; the model is small enough that 16-bit fits anyway, so there is no reason to risk it.
- **What to fine-tune — the vision part vs the language part.** Chose to **freeze the vision part** and adapt the language part only. This saves memory and protects the visual features the other four benchmarks rely on.
- **How much data — a big multi-stage pipeline vs a small single pass.** Chose small (~7,000 examples, one pass). A day-scale proof of concept should be minimal and clear; bigger pipelines are noted as future work.
- **Forgetting insurance — train on tables only vs mix in old data.** Chose to **mix in ~20–25% of general document data** ("replay"). The literature shows table-only fine-tuning can actually *lower* a strong model's table scores; the replay mix is our main guard against breaking what already works.

### 6.4 Assumptions and known risks

- **Assumption (now tested):** we predicted the weak spot, then checked. The data refined it — the gap is **Wikipedia-style table lookup (vwtq/vwtq_syn)**, not financial/numeric tables (which are already strong). The training data is aimed at that measured weakness, not at our original guess. This is exactly the "use our own numbers" discipline the task rewards.
- **Assumption:** publicly available table-reasoning datasets are good enough to teach this skill at this size. The research write-ups point to several; we verify the actual datasets exist and load before relying on them.
- **Biggest risk:** the model is new, and the LoRA tooling may not attach cleanly to its unusual internals. We **de-risk this in the first hour** with a tiny smoke test (attach the add-on weights, confirm the output actually changes) before spending time on data or training. If it fails, we fall back to a simpler attach configuration, and document that honestly.
- **Honest ceiling:** at 0.8B the vision encoder can only resolve so much of a dense table, so some lookups will stay wrong no matter the data. We expect a **modest, targeted gain on the lookup sub-types, not a fix**.

### 6.5 How we will test the hypothesis scientifically

A simple before/after experiment with a guardrail:

1. **Measure first (the control) — done.** We scored the baseline on the four table sub-types separately (`scripts/tablevqa_subdomain_report.py`, reading numbers our Part-1 run already produced). Result above: **vwtq 27.8 and vwtq_syn 33.1 are the floor; fintabnetqa 84.0 is already strong.** This redirected the plan away from the financial/numeric focus the outside reports assumed.
2. **Make one change (the intervention).** Run a single LoRA fine-tune on a mixture deliberately weighted toward **Wikipedia-style table lookup (vwtq)**, the measured weak spot and the highest-leverage slice.
3. **Measure again, the same way.** Re-score table questions (per sub-type) **and** re-score the other four benchmarks on the *exact same* fixed 1000-example subsets used in Part 1, so the comparison is fair.
4. **Decide with a pre-set rule (the guardrail).** We declare success only if the **table score rises by at least ~3 points, with the gains landing in the sub-types we targeted, and none of the other four benchmarks drops by more than ~1.5 points.** Setting this rule *before* training keeps us honest. Because the table benchmark is small (~1,500 questions), we always report the sub-type breakdown, not just the average, so we do not over-read noise.
5. **If it does not work,** we have a planned next move (more replay or a gentler learning rate; failing that, a larger same-family model generating extra practice data) rather than endless blind tweaking.

Everything is scripted with fixed seeds and pinned software versions so another engineer can reproduce the before/after numbers — the same reproducibility standard we held in Part 1.

### 6.6 Results: the first attempt failed the gate (reported faithfully)

We ran the single LoRA fine-tune as specified and re-evaluated on the same fixed seed-42 subsets. **It failed the pre-registered gate badly** — the table score dropped instead of rising, and the model's previously-strong document skills collapsed:

| Benchmark | Baseline | After LoRA | Δ |
|---|---:|---:|---:|
| OCRBench | 79.2 | 77.4 | −1.8 |
| DocVQA | 89.3 | 50.8 | **−38.4** |
| ChartQA | 70.8 | 75.3 | +4.5 |
| InfoVQA | 62.3 | 27.8 | **−34.5** |
| **TableVQA (target)** | **50.1** | **19.6** | **−30.5** |

TableVQA sub-domains: vwtq 27.8→**0.6**, vwtq_syn 33.1→**0.0**, vtabfact 55.5→62.4, fintabnetqa 84.0→**15.4**.

**Diagnosis (what the numbers say).** Only VTabFact — a fixed True/False output space — survived (it even rose to 62.4), while every sub-task that must emit a *specific value* collapsed toward zero, and the model's best general skills (DocVQA, InfoVQA) collapsed too. That signature is an **output-distribution collapse**, not lost perception. The proximate cause is an **answer-style mismatch**: our training source (Visual-TableQA, synthetic) supplies *verbose, full-sentence* answers (e.g. "The Anglo-Saxon rune set has the highest hardness…"), whereas every benchmark is scored on *short cell values or spans*. We effectively trained the model to be verbose, which exact-match and ANLS scoring punish across the board; the 25% general-document replay was too weak to counteract the shift.

**What we would change (next steps — not run here).**
1. **Match training answer style to the evaluation:** short-answer table lookup (filter Visual-TableQA to short answers, or use WikiTableQuestions-derived short answers under the same question-overlap guard). This is the single highest-leverage fix.
2. **Raise replay to ~40–50%** to protect DocVQA/InfoVQA.
3. **Train more gently:** LR 5e-5, fewer steps, with early-stopping on a held-out dev gate.
4. **Add a pre-training data-quality check** comparing the answer-length distribution of the training mix against the benchmarks — this one check would have caught the failure before any GPU time was spent.

**What still holds.** The pre-registered gate did exactly its job: it **rejected a degraded model** instead of letting a misleading "improvement" through — the methodological point of Part 2 survives the negative result. The VWTQ lookup gap from §6.2 is unchanged and remains the right target; closing it requires **answer-style-matched supervision**, not merely more table data.

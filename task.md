# Task: Adapting Small Vision-Language Models (VLMs) for Document Understanding

## Problem Statement

Small VLMs in the sub-1B parameter range have shown promising general visual reasoning capabilities, but their performance on document understanding tasks is largely uncharacterized. Most existing evaluations focus on large models (3B+), leaving a critical gap in understanding whether small VLMs can serve as a reliable foundation for document understanding domain adaptation. This task addresses that gap through systematic evaluation and analysis.

## Objectives

The task has two parts:

- **Part 1 — Evaluation:** Identify the top small VLMs (under 1B parameters), select or design document understanding benchmarks from [VLMEvalKit](https://github.com/open-compass/VLMEvalKit), and produce a systematic comparison of model performance across relevant metrics.
- **Part 2 — Improvement Strategy:** Based on your evaluation results, identify the knowledge gap of the best-performing model and propose a concrete, justified strategy to improve its performance on the benchmark.

### Ground rules

- You may select any models, benchmarks, datasets, evaluation metrics, and improvement methodology, but must provide a clear rationale for every choice.
- The evaluation must go beyond generic VQA accuracy and probe capabilities that matter specifically for the desired use case, like document understanding.
- Your PoC should be reproducible — another engineer should be able to run your scripts and obtain the same results.
- You are expected to use free GPU resources (e.g. Google Colab, Kaggle) or your own hardware.

## Your Proposal Must Address

### Part 1 — Evaluation

1. **Model Selection Criteria:** Identify the small VLMs (under 1B parameters) included in your comparison. Justify your selection based on architecture, pretraining data, and known capabilities.
2. **Benchmark Selection or Design:** Describe the benchmark used for evaluation. You may use existing benchmarks from VLMEvalKit, construct your own, or combine both — but justify your choice in terms of relevance, diversity, and quality.
3. **Evaluation Metrics:** Define the metrics used to compare models. Go beyond accuracy — consider calibration, robustness to domain-specific terminology. Justify each metric's relevance to the specific domain.
4. **Experimental Setup & Reproducibility:** Specify hardware, inference configuration, and preprocessing applied consistently across all models to ensure a fair and reproducible comparison.
5. **Software Stack:** List the frameworks and tools used. Justify selections based on compatibility with small VLMs and suitability for edge deployment.

### Part 2 — Knowledge Gap & Improvement Strategy

1. **Knowledge Gap Analysis:** Based on your evaluation results, identify where and why the best-performing model falls short — which task types, modalities, or reasoning skills expose its limitations. Use your benchmark results as evidence.
2. **Improvement Strategy:** Propose a concrete methodology to improve the selected model's performance on your benchmark. Justify your approach — whether fine-tuning, knowledge distillation, parameter-efficient adaptation, or a combination — with reference to relevant literature. Address challenges specific to the domain.
3. **Expected Outcomes:** Describe the improvement you would expect from your proposed strategy and how you would measure it using your benchmark.

## Proof of Concept (PoC) Development

Develop a PoC demonstrating fine-tuning methodology, including:

- **Model Survey & Selection:** A documented rationale for the models included, with a brief profile of each covering architecture, parameter count, pretraining data, and known capabilities.
- **Benchmark Construction or Adaptation:** Scripts or data files defining your evaluation set with clear documentation of design decisions.
- **Evaluation Pipeline:** A single script that loads any candidate model, runs it on the benchmark, and outputs per-model scores. The pipeline should require minimal modification to evaluate a new model.
- **Results & Comparison Table:** A structured table comparing all models across your defined metrics, with a written interpretation identifying the strongest candidate and its key failure modes.
- **Improvement Plan:** A written section translating your gap analysis into a step-by-step improvement strategy, grounded in your benchmark findings and supported by relevant literature.

## Deliverables

1. Source code for the evaluation pipeline and any benchmark construction scripts.
2. Comparison table with results and written analysis.
3. Technical report (PDF) covering benchmark design, evaluation methodology, knowledge gap analysis, and improvement strategy.

## Due Date

Submission must be completed within **5 days**. Be prepared to **present and justify your approach** in an upcoming interview session.

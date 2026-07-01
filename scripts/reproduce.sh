#!/usr/bin/env bash
# Reproduce every reported result, from a GPU machine (Colab / any CUDA box).
#
# Why phases: the six models split across TWO incompatible transformers versions
# (Env A: transformers<4.57 for the built-ins + FastVLM; Env B: transformers@main
# for Qwen3.5-0.8B, LFM2.5-VL-450M, and the Part-2 fine-tune). They cannot share a
# Python environment, so reproduction is two phases with a fresh env between them,
# then a light figures phase that runs in either.
#
#   OUT_DIR=outputs bash scripts/reproduce.sh env-a      # phase A
#   #  <-- start a FRESH environment / restart the runtime -->
#   OUT_DIR=outputs bash scripts/reproduce.sh env-b      # phase B (incl. Part 2)
#   OUT_DIR=outputs bash scripts/reproduce.sh figures    # plots for the report
#
# Env vars (all optional): OUT_DIR (results tree, default ./outputs),
#   DATA_DIR (LoRA training data, default ./table_sft), N (samples/dataset, 1000),
#   PY (python interpreter, default python).
set -euo pipefail

PHASE="${1:-}"
OUT_DIR="${OUT_DIR:-outputs}"
DATA_DIR="${DATA_DIR:-table_sft}"
N="${N:-1000}"
PY="${PY:-python}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
mkdir -p "$OUT_DIR"

hr() { printf '\n=== %s ===\n' "$1"; }

# ---- Phase A: Env A models (SmolVLM2, InternVL3, LLaVA-OV, FastVLM) ----
phase_a() {
  hr "Env A setup (transformers<4.57)"
  bash setup.sh
  hr "Smoke test (sanity)"
  $PY scripts/smoke_test.py --score || echo "[warn] smoke test issues (non-fatal)"
  hr "Efficiency ablations (fp16 + generation cap)"
  $PY scripts/efficiency_ablation.py --out "$OUT_DIR/ablation" --n 100
  hr "Full eval: 3 built-in models (N=$N)"
  $PY scripts/run_eval.py --out "$OUT_DIR" --n "$N" --fp16
  hr "Full eval: FastVLM-0.5B (Env A custom wrapper)"
  $PY scripts/run_eval.py --out "$OUT_DIR" --n "$N" --fp16 --models FastVLM-0.5B
  echo; echo ">> Phase A done. START A FRESH ENVIRONMENT, then: reproduce.sh env-b"
}

# ---- Phase B: Env B models + Part-2 improvement study ----
phase_b() {
  hr "Env B setup (transformers@main + peft/trl/accelerate)"
  bash setup.sh --bleeding-edge --train
  hr "Full eval: Qwen3.5-0.8B + LFM2.5-VL-450M (N=$N)"
  $PY scripts/run_eval.py --out "$OUT_DIR" --n "$N" --fp16 --models Qwen3.5-0.8B LFM2.5-VL-450M

  hr "Part 2 / E0: baseline TableVQA sub-domain diagnosis"
  $PY scripts/tablevqa_subdomain_report.py --out "$OUT_DIR" --models Qwen3.5-0.8B
  hr "Part 2 / de-risk: LoRA adapter smoke test"
  $PY scripts/train_lora_qwen.py --smoke
  hr "Part 2 / build VWTQ-focused training mixture -> $DATA_DIR"
  $PY scripts/build_table_sft.py --data-dir "$DATA_DIR" --n-total 7000
  hr "Part 2 / LoRA fine-tune (1 epoch, vision frozen)"
  $PY scripts/train_lora_qwen.py --out "$OUT_DIR" --data "$DATA_DIR/train.jsonl" \
      --run-name table_lora_v1 --epochs 1
  hr "Part 2 / re-evaluate the tuned model on all 5 benchmarks"
  MVDE_LORA_ADAPTER="$OUT_DIR/lora_adapters/table_lora_v1" \
    $PY scripts/run_eval.py --out "$OUT_DIR" --n "$N" --fp16 --models Qwen3.5-0.8B-TableLoRA
  hr "Part 2 / no-regression gate  (exit 1 on FAIL is EXPECTED for our run)"
  $PY scripts/regression_gate.py --out "$OUT_DIR" || echo "[gate] FAIL — as reported (Sec 8)"
  hr "Part 2 / sub-domain before/after"
  $PY scripts/tablevqa_subdomain_report.py --out "$OUT_DIR" \
      --models Qwen3.5-0.8B Qwen3.5-0.8B-TableLoRA
  hr "Report assets: export qualitative examples"
  $PY scripts/export_examples.py --out "$OUT_DIR" --model Qwen3.5-0.8B
  echo; echo ">> Phase B done. Aggregate table is in $OUT_DIR/summary/."
}

# ---- Figures for the report (either env; needs matplotlib) ----
phase_figures() {
  hr "Report figures (Pareto, per-benchmark, sub-domains)"
  $PY scripts/make_report_figures.py --data-dir "$OUT_DIR"
  if [ -f "$OUT_DIR/examples/manifest.json" ]; then
    hr "Report figure: qualitative-examples gallery"
    $PY scripts/make_examples_figure.py --src "$OUT_DIR/examples"
  else
    echo "[skip] no $OUT_DIR/examples — run env-b first to export examples"
  fi
  echo; echo ">> Figures in technical_report/figures/. Compile the PDF with:"
  echo "   cd technical_report && latexmk -pdf report.tex"
}

case "$PHASE" in
  env-a)   phase_a ;;
  env-b)   phase_b ;;
  figures) phase_figures ;;
  *) echo "usage: OUT_DIR=outputs bash scripts/reproduce.sh {env-a|env-b|figures}"; exit 2 ;;
esac

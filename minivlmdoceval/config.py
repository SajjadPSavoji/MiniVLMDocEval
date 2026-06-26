"""Central configuration for MiniVLMDocEval evaluation runs."""

# Models already supported by VLMEvalKit out-of-the-box (no custom wrapper).
# Values are VLMEvalKit registry keys. The remaining shortlist models
# (FastVLM-0.5B, Qwen3.5-0.8B -> custom wrappers; LFM2.5-VL-450M -> config key)
# are added later.
BUILTIN_MODELS = [
    "SmolVLM2-500M",                     # HuggingFaceTB/SmolVLM2-500M-Video-Instruct
    "InternVL3-1B",                      # OpenGVLab/InternVL3-1B
    "llava-onevision-qwen2-0.5b-ov-hf",  # llava-hf/... (HF-native; our shortlist pick)
]

# Document-understanding benchmark suite — FULL splits (VLMEvalKit dataset keys).
DATASETS = [
    "OCRBench",       # OCR across 5 sub-tasks          -> accuracy (/1000 norm)
    "DocVQA_VAL",     # dense document text VQA         -> ANLS
    "ChartQA_TEST",   # chart data + numeric reasoning  -> relaxed accuracy
    "InfoVQA_VAL",    # infographic layout reasoning    -> ANLS
    "TableVQABench",  # table structure reasoning       -> accuracy
]

# Output tree. EVERYTHING is written under a single base dir (`--out`), so the
# code runs on any system; on Colab point --out at Google Drive so it persists
# across sessions (and enables VLMEvalKit --reuse).
#   <out>/predictions/<model>/<eval_id>/...   VLMEvalKit work-dir (status.json, preds)
#   <out>/summary/comparison.{csv,md}, scores_long.csv
#   <out>/logs/<model>_<timestamp>.log        tee of each run (resume/debug)
DEFAULT_OUT = "outputs"
PREDICTIONS_SUBDIR = "predictions"
SUMMARY_SUBDIR = "summary"
LOGS_SUBDIR = "logs"

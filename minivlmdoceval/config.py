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

# Output locations.
# Heavy predictions/resume-state: point at Google Drive on Colab to survive
# session disconnects (and enable VLMEvalKit --reuse). The light score tables go
# to RESULTS_DIR (git-tracked) for sharing.
DEFAULT_WORK_DIR = "outputs"
RESULTS_DIR = "results"

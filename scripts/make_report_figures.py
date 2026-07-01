#!/usr/bin/env python
"""Generate the publication figures for the technical report.

Reads the evaluation artifacts (no recompute) and writes vector PDFs:
  fig_pareto.pdf       mean accuracy vs mean latency (the accuracy/efficiency frontier)
  fig_benchmarks.pdf   per-benchmark accuracy, all 6 models (the diagnostic ladder)
  fig_subdomains.pdf   TableVQA sub-domain accuracy, all 6 models (the gap centerpiece)

Inputs (default under drive_sync/, the synced eval mirror):
  summary/comparison.csv                      6x5 accuracy matrix
  summary/scores_long.csv                     per-pair timing (s_per_sample)
  predictions/<model>/TableVQABench_n1000_acc.csv   per-sub-domain accuracy

Stdlib csv only (no pandas). Usage:
  python scripts/make_report_figures.py
  python scripts/make_report_figures.py --data-dir drive_sync --out <figdir>
"""
import argparse
import ast
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
FIGDIR_DEFAULT = REPO / "technical_report" / "figures"

# Display names + a fixed colour per model (Okabe–Ito, colour-blind safe), ordered by mean acc.
MODELS = [
    ("Qwen3.5-0.8B", "Qwen3.5-0.8B", "#0072B2"),
    ("InternVL3-1B", "InternVL3-1B", "#E69F00"),
    ("LFM2.5-VL-450M", "LFM2.5-VL-450M", "#009E73"),
    ("llava-onevision-qwen2-0.5b-ov-hf", "LLaVA-OV-0.5B", "#CC79A7"),
    ("SmolVLM2-500M", "SmolVLM2-500M", "#56B4E9"),
    ("FastVLM-0.5B", "FastVLM-0.5B", "#D55E00"),
]
BENCH = [  # csv key -> display
    ("OCRBench", "OCRBench"), ("DocVQA_VAL", "DocVQA"), ("ChartQA_TEST", "ChartQA"),
    ("InfoVQA_VAL", "InfoVQA"), ("TableVQABench", "TableVQA"),
]
SUBDOMAINS = [("vwtq", "VWTQ"), ("vwtq_syn", "VWTQ-Syn"), ("vtabfact", "VTabFact"),
              ("fintabnetqa", "FinTabNetQA")]

plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def read_accuracy(data_dir):
    """{model_key: {bench_key: acc}} from summary/comparison.csv."""
    out = {}
    with open(Path(data_dir) / "summary" / "comparison.csv", newline="") as f:
        for row in csv.DictReader(f):
            out[row["model"]] = {b: float(row[b]) for b, _ in BENCH if row.get(b)}
    return out


def read_latency(data_dir):
    """{model_key: mean s_per_sample} from summary/scores_long.csv."""
    acc, cnt = {}, {}
    with open(Path(data_dir) / "summary" / "scores_long.csv", newline="") as f:
        for row in csv.DictReader(f):
            m = row["model"]
            acc[m] = acc.get(m, 0.0) + float(row["s_per_sample"])
            cnt[m] = cnt.get(m, 0) + 1
    return {m: acc[m] / cnt[m] for m in acc}


def read_subdomains(data_dir, model):
    """{subdomain: list-mean acc} from predictions/<model>/TableVQABench_n1000_acc.csv."""
    p = Path(data_dir) / "predictions" / model / "TableVQABench_n1000_acc.csv"
    out = {}
    if not p.exists():
        return out
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            vals = ast.literal_eval(row["average_scores"])
            vals = vals if isinstance(vals, (list, tuple)) else [vals]
            out[row["split"]] = sum(vals) / len(vals)
    return out


def _pareto_front(pts):
    """Display names that are not dominated (no other point has <= latency AND
    >= accuracy with at least one strict)."""
    front = []
    for d, (x, y) in pts.items():
        if not any(ox <= x and oy >= y and (ox < x or oy > y)
                   for od, (ox, oy) in pts.items() if od != d):
            front.append(d)
    return front


def fig_pareto(acc, lat, out):
    # Short tags + per-model label placement (offset in points) to prevent overlap.
    SHORT = {"Qwen3.5-0.8B": "Qwen3.5", "InternVL3-1B": "InternVL3",
             "LFM2.5-VL-450M": "LFM2.5-VL", "LLaVA-OV-0.5B": "LLaVA-OV",
             "SmolVLM2-500M": "SmolVLM2", "FastVLM-0.5B": "FastVLM"}
    OFFS = {
        "Qwen3.5-0.8B":   (8, 5, "left", "bottom"),
        "InternVL3-1B":   (9, -1, "left", "center"),   # push right into open space
        "LFM2.5-VL-450M": (9, 0, "left", "center"),
        "LLaVA-OV-0.5B":  (0, 10, "center", "bottom"),
        "SmolVLM2-500M":  (0, -11, "center", "top"),
        "FastVLM-0.5B":   (-9, 2, "right", "bottom"),
    }
    pts = {}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, disp, color in MODELS:
        if key not in acc or key not in lat:
            continue
        mean_acc = sum(acc[key].values()) / len(acc[key])
        x = lat[key]
        pts[disp] = (x, mean_acc)
        ax.scatter(x, mean_acc, s=95, color=color, zorder=3, edgecolor="k", linewidth=0.6)
        dx, dy, ha, va = OFFS.get(disp, (8, 0, "left", "center"))
        ax.annotate(SHORT.get(disp, disp), (x, mean_acc), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va=va, fontsize=11)
    # Pareto frontier line (non-dominated points, sorted by latency).
    front = sorted(_pareto_front(pts), key=lambda d: pts[d][0])
    ax.plot([pts[d][0] for d in front], [pts[d][1] for d in front],
            "--", color="0.5", lw=1.2, zorder=1, label="Pareto frontier")
    from matplotlib.ticker import FuncFormatter, NullFormatter
    ax.set_xscale("log")
    ax.set_xlabel("Mean latency (s / sample, log scale; indicative — mixed T4/L4)", fontsize=11)
    ax.set_ylabel("Mean accuracy (0–100)", fontsize=11)
    ax.set_title("Accuracy–efficiency frontier", fontsize=12)
    ax.set_xlim(0.17, 6.0)
    ax.set_ylim(32, 77)
    ax.set_xticks([0.2, 0.3, 0.5, 1, 2, 4])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=9, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_pareto.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_benchmarks(acc, out):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    n = len([m for m in MODELS if m[0] in acc])
    width = 0.8 / n
    xs = range(len(BENCH))
    for i, (key, disp, color) in enumerate([m for m in MODELS if m[0] in acc]):
        vals = [acc[key].get(b, 0.0) for b, _ in BENCH]
        offs = [x + (i - (n - 1) / 2) * width for x in xs]
        ax.bar(offs, vals, width=width, label=disp, color=color, edgecolor="k", linewidth=0.3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d for _, d in BENCH])
    ax.set_ylabel("Accuracy (0–100)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-benchmark accuracy (diagnostic ladder)")
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_benchmarks.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_subdomains(data_dir, out):
    sub = {key: read_subdomains(data_dir, key) for key, _, _ in MODELS}
    present = [m for m in MODELS if sub[m[0]]]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    n = len(present)
    width = 0.8 / n
    xs = range(len(SUBDOMAINS))
    for i, (key, disp, color) in enumerate(present):
        vals = [sub[key].get(s, 0.0) for s, _ in SUBDOMAINS]
        offs = [x + (i - (n - 1) / 2) * width for x in xs]
        ax.bar(offs, vals, width=width, label=disp, color=color, edgecolor="k", linewidth=0.3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([d for _, d in SUBDOMAINS])
    ax.set_ylabel("Accuracy (0–100)")
    ax.set_ylim(0, 100)
    ax.set_title("TableVQA sub-domains: lookup (VWTQ/-Syn) is the frontier, not financial tables")
    ax.legend(fontsize=7.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    fig.tight_layout()
    fig.savefig(out / "fig_subdomains.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(REPO / "drive_sync"))
    ap.add_argument("--out", default=str(FIGDIR_DEFAULT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    acc = read_accuracy(args.data_dir)
    lat = read_latency(args.data_dir)
    fig_pareto(acc, lat, out)
    fig_benchmarks(acc, out)
    fig_subdomains(args.data_dir, out)
    print(f"[figures] wrote 3 PDFs -> {out}")
    for p in sorted(out.glob("fig_*.pdf")):
        print(f"  {p.name}  ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()

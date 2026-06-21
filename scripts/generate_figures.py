import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "source_artifacts"
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


COLORS = {
    "fp16": "#4C78A8",
    "gptq": "#F58518",
    "sg": "#54A24B",
    "negative": "#E45756",
    "neutral": "#B279A2",
    "gray": "#6B7280",
}

MODEL_LABELS = {
    "Qwen2.5-0.5B": "Qwen-0.5B",
    "Qwen2.5-1.5B": "Qwen-1.5B",
    # Historical result files retain the old key; the actual source checkpoint
    # is HuggingFaceTB/SmolLM2-1.7B.
    "SmolLM-1.7B": "SmolLM2-1.7B",
    "gemma-2-2b-it": "Gemma-2-2B-it",
}

SG_METHOD = {
    "Qwen2.5-0.5B": "config_b",
    "Qwen2.5-1.5B": "config_b_2a",
    "SmolLM-1.7B": "config_b",
}

def score(records, model, method, task):
    return float(records[(model, method)][task])


def wilson(score, n=300, z=1.96):
    p = score / 100.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (center - half) * 100, (center + half) * 100


def load_main_results():
    rows = []
    with open(DATA / "results" / "main_results.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_task_results():
    records = {}
    with open(DATA / "results" / "task_results_full.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            records[(r["model"], r["method"])] = r["scores"]
    return records


def load_gsm8k500_summary():
    rows = []
    path = DATA / "experiments" / "fix_gsm8k_500" / "results_direct" / "summary_gsm8k500.jsonl"
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_gsm8k500_paired():
    path = DATA / "experiments" / "fix_gsm8k_500" / "results_direct" / "paired_stats_gsm8k500.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def fig1_degradation():
    rows = load_main_results()
    fp16 = {(r["Model"],): r for r in rows if r["Method"] == "fp16"}
    tasks = ["ARC_Challenge", "HellaSwag", "MMLU", "GSM8K"]
    labels = ["ARC-C", "HellaSwag", "MMLU", "GSM8K"]
    methods = ["rtn", "gptq", "awq"]

    values = {t: [] for t in tasks}
    for r in rows:
        if r["Method"] not in methods:
            continue
        base = fp16[(r["Model"],)]
        for t in tasks:
            values[t].append(float(r[t]) - float(base[t]))

    means = [np.mean(values[t]) for t in tasks]
    xs = np.arange(len(tasks))

    fig, ax = plt.subplots(figsize=(5.6, 3.2), constrained_layout=True)
    bars = ax.bar(xs, means, color=["#8DA0CB", "#66C2A5", "#FC8D62", "#E45756"], width=0.62)
    for i, t in enumerate(tasks):
        jitter = np.linspace(-0.16, 0.16, len(values[t]))
        ax.scatter(np.full(len(values[t]), i) + jitter, values[t], s=16, color="#263238", alpha=0.7, zorder=3)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks(xs, labels)
    ax.set_ylabel("Accuracy change vs FP16 (points)")
    ax.set_title("4-bit PTQ degradation is concentrated on GSM8K")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m - 0.8, f"{m:.2f}", ha="center", va="top", color="white", fontsize=8)
    fig.savefig(OUT / "fig1_benchmark_degradation.pdf")
    fig.savefig(OUT / "fig1_benchmark_degradation.png")
    plt.close(fig)


def fig2_repair_budget():
    summary = load_gsm8k500_summary()
    paired = {r["model"]: r for r in load_gsm8k500_paired()}
    models = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "SmolLM-1.7B", "gemma-2-2b-it"]
    acc = {(r["model"], r["method"]): r["accuracy"] for r in summary}
    labels = [MODEL_LABELS[m] for m in models]
    fp16 = [acc[(m, "fp16")] for m in models]
    gptq = [acc[(m, "gptq")] for m in models]
    sg = [acc[(m, "sg")] for m in models]
    x = np.arange(len(models))
    w = 0.22

    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), gridspec_kw={"width_ratios": [1.45, 1.0]}, constrained_layout=True)
    ax = axes[0]
    ax.bar(x - w, fp16, w, label="FP16", color=COLORS["fp16"])
    ax.bar(x, gptq, w, label="GPTQ-W4", color=COLORS["gptq"])
    ax.bar(x + w, sg, w, label="SG-MMP", color=COLORS["sg"])
    for xi, yi, gy, m in zip(x + w, sg, gptq, models):
        lo, hi = wilson(yi, n=500)
        ax.errorbar(xi, yi, yerr=[[yi - lo], [hi - yi]], fmt="none", ecolor="#222222", elinewidth=0.8, capsize=2)
        dy = 1.3 if m != "gemma-2-2b-it" else 2.4
        ax.text(xi, hi + dy, f"+{yi - gy:.1f}", ha="center", va="bottom", fontsize=7, color="#1F2937")
    ax.set_xticks(x, labels, rotation=10)
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("(a) Direct GSM8K-500 accuracy")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_ylim(0, max(fp16 + sg) + 10)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    ax = axes[1]
    y = np.arange(len(models))
    deltas = [paired[m]["delta"] for m in models]
    lows = [paired[m]["paired_bootstrap_ci95"][0] for m in models]
    highs = [paired[m]["paired_bootstrap_ci95"][1] for m in models]
    colors = [COLORS["sg"], COLORS["sg"], COLORS["sg"], COLORS["gray"]]
    ax.axvline(0, color="#333333", linewidth=0.8)
    for yi, d, lo, hi, c, m in zip(y, deltas, lows, highs, colors, models):
        ax.errorbar(d, yi, xerr=[[d - lo], [hi - d]], fmt="o", color=c, ecolor=c, capsize=3, markersize=4)
        p = paired[m]["mcnemar_p_exact"]
        ptxt = "p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.text(hi + 0.45, yi, ptxt, va="center", ha="left", fontsize=7)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("SG-MMP - GPTQ-W4 (points)")
    ax.set_title("(b) Paired bootstrap 95% CI")
    ax.set_xlim(min(lows) - 1.0, max(highs) + 4.0)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.savefig(OUT / "fig2_repair_budget.pdf")
    fig.savefig(OUT / "fig2_repair_budget.png")
    plt.close(fig)


def fig3_same_budget():
    records = load_task_results()
    data = [
        ("SG-MMP", "config_b_samebudget_300", COLORS["sg"]),
        ("Random 42", "random_42_samebudget_300", COLORS["gray"]),
        ("Random 123", "random_123_samebudget_300", COLORS["gray"]),
        ("Random 456", "random_456_samebudget_300", COLORS["gray"]),
        ("First 4", "first_4_samebudget_300", COLORS["neutral"]),
        ("Last 4", "last_4_samebudget_300", COLORS["neutral"]),
    ]
    labels = [d[0] for d in data]
    y = np.arange(len(data))
    scores = [score(records, "Qwen2.5-0.5B", d[1], "gsm8k") for d in data]
    colors = [d[2] for d in data]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), constrained_layout=True)
    ax.barh(y, scores, color=colors, height=0.64)
    for yi, si in zip(y, scores):
        lo, hi = wilson(si)
        ax.errorbar(si, yi, xerr=[[si - lo], [hi - si]], fmt="none", ecolor="#222222", elinewidth=0.8, capsize=2)
        ax.text(si + 0.7, yi, f"{si:.2f}", va="center", ha="left", fontsize=7)
    random_scores = [score(records, "Qwen2.5-0.5B", f"random_{seed}_samebudget_300", "gsm8k") for seed in ["42", "123", "456"]]
    random_mean = np.mean(random_scores)
    ax.axvline(random_mean, color="#333333", linestyle="--", linewidth=1.0, label=f"Random mean = {random_mean:.2f}")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("GSM8K accuracy (%)")
    ax.set_title("Same-budget comparison at 4.90 average weight bits")
    ax.set_xlim(0, max(scores) + 9)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.savefig(OUT / "fig3_same_budget.pdf")
    fig.savefig(OUT / "fig3_same_budget.png")
    plt.close(fig)

def fig4_mechanism():
    # Panel A: single-layer sensitivity
    baseline = None
    layer_scores = {}
    with open(DATA / "experiments" / "exp02_per_layer" / "results" / "per_layer.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "baseline":
                baseline = r["gsm8k_score"]
            elif r["type"] == "run" and r["bits"] == 4:
                layer_scores[int(r["layer"])] = r["gsm8k_score"]
    layers = np.arange(24)
    drops = np.array([baseline - layer_scores[i] for i in layers])

    # Panel B: first divergent layer counts
    div_layers, t099, t095, t09 = [], [], [], []
    with open(DATA / "experiments" / "exp14_first_divergent_step" / "results" / "per_layer_summary.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            div_layers.append(int(row["layer"]))
            t099.append(int(row["count_t099"]))
            t095.append(int(row["count_t095"]))
            t09.append(int(row["count_t09"]))

    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.3), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.bar(layers, drops, color=[COLORS["sg"] if i in {2, 6, 7, 11} else "#9CA3AF" for i in layers], width=0.75)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_ylabel("FP16 - single-layer W4")
    ax.set_title("(a) Layer sensitivity from single-layer W4 quantization", pad=8)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.text(0.01, 0.92, f"FP16 sweep baseline = {baseline:.2f}", transform=ax.transAxes, fontsize=8)

    ax = axes[1]
    ax.bar(div_layers, t099, color="#E45756", label="threshold 0.99", width=0.75)
    ax.plot(div_layers, t095, color="#4C78A8", marker="o", markersize=3, linewidth=1.0, label="threshold 0.95")
    ax.plot(div_layers, t09, color="#54A24B", marker="s", markersize=3, linewidth=1.0, label="threshold 0.90")
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Divergence count")
    ax.set_title("(b) FP16/GPTQ hidden-state divergence concentrates in deep layers", pad=8)
    ax.set_xticks(np.arange(0, 24, 2))
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("threshold 0.99"), labels.index("threshold 0.95"), labels.index("threshold 0.90")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.savefig(OUT / "fig4_mechanism_layers.pdf")
    fig.savefig(OUT / "fig4_mechanism_layers.png")
    plt.close(fig)


def fig5_failure_modes():
    records = load_task_results()

    # Panel A: global transforms
    global_methods = [
        ("GPTQ-W4", "gptq"),
        ("Hadamard", "hadamard_gptq"),
        ("QEP 0.0", "onecomp_qep_g0.0"),
        ("QEP 1.0", "onecomp_qep_g1.0"),
        ("QEP 1.0\n64x1024", "onecomp_qep_g1.0_64x1024"),
        ("SG-MMP", "config_b"),
    ]
    arc = [records[("Qwen2.5-0.5B", m)]["arc_challenge"] for _, m in global_methods]
    gsm = [records[("Qwen2.5-0.5B", m)]["gsm8k"] for _, m in global_methods]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.55), constrained_layout=True)
    ax = axes[0]
    ax.scatter(arc, gsm, s=45, color=[COLORS["sg"] if n == "SG-MMP" else COLORS["negative"] if n != "GPTQ-W4" else COLORS["gptq"] for n, _ in global_methods])
    offsets = {
        "GPTQ-W4": (6, 5),
        "Hadamard": (10, 16),
        "QEP 0.0": (6, 10),
        "QEP 1.0": (8, 10),
        "QEP 1.0\n64x1024": (-42, 8),
        "SG-MMP": (6, 5),
    }
    for (name, _), xval, yval in zip(global_methods, arc, gsm):
        dx, dy = offsets[name]
        ax.annotate(
            name,
            (xval, yval),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.78),
        )
    ax.set_xlabel("ARC-Challenge accuracy (%)")
    ax.set_ylabel("GSM8K accuracy (%)")
    ax.set_title("(a) Choice accuracy can hide math damage", fontsize=9)
    ax.set_xlim(min(arc) - 0.55, max(arc) + 0.85)
    ax.set_ylim(min(gsm) - 0.6, max(gsm) + 3.6)
    ax.grid(True, linestyle=":", alpha=0.35)

    ax = axes[1]
    labels = ["SG-MMP", "+ deep LoRA"]
    gsm8k = [
        score(records, "Qwen2.5-0.5B", "config_b", "gsm8k"),
        score(records, "Qwen2.5-0.5B", "config_b_failure_lora_6c", "gsm8k"),
    ]
    svamp = [
        score(records, "Qwen2.5-0.5B", "config_b", "svamp"),
        score(records, "Qwen2.5-0.5B", "config_b_failure_lora_6c", "svamp"),
    ]
    x = np.arange(len(labels))
    w = 0.32
    ax.bar(x - w / 2, gsm8k, w, color=COLORS["sg"], label="GSM8K")
    ax.bar(x + w / 2, svamp, w, color=COLORS["negative"], label="SVAMP")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) LoRA improves GSM8K but drops on SVAMP", fontsize=9)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.03))
    ax.set_ylim(0, max(gsm8k) + 4)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    fig.savefig(OUT / "fig5_failure_modes.pdf")
    fig.savefig(OUT / "fig5_failure_modes.png")
    plt.close(fig)


def main():
    fig1_degradation()
    fig2_repair_budget()
    fig3_same_budget()
    fig4_mechanism()
    fig5_failure_modes()
    print(f"Figures written to {OUT}")


if __name__ == "__main__":
    main()

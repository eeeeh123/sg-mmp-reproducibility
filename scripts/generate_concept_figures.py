from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data" / "processed"

GSM500 = (
    DATA
    / "source_artifacts"
    / "experiments"
    / "fix_gsm8k_500"
    / "results_direct"
    / "summary_gsm8k500.jsonl"
)
PAIRED_STATS = DATA / "gsm8k500" / "recomputed_paired_stats.json"
BIT_BUDGET = DATA / "bit_budget_summary.json"
PER_LAYER = DATA / "source_artifacts" / "experiments" / "exp02_per_layer" / "results" / "per_layer.jsonl"
FIRST_DIV = DATA / "source_artifacts" / "experiments" / "exp14_first_divergent_step" / "results" / "per_layer_summary.csv"
LAYER_REPLACE = DATA / "source_artifacts" / "experiments" / "exp07_layer_replacement" / "layer_replacement.csv"
ABLATION = (
    DATA
    / "source_artifacts"
    / "experiments"
    / "fix_gsm8k_500"
    / "results_direct"
    / "qwen05_ablation_analysis_gsm8k500.json"
)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.0,
        "axes.titlesize": 9.2,
        "axes.labelsize": 8.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
    }
)

DARK = "#263238"
MUTED = "#687780"
GRID = "#D8E0E4"
PANEL = "#F7F9FA"
BLUE = "#0077BB"
TEAL = "#009988"
ORANGE = "#EE7733"
RED = "#CC3311"
MAGENTA = "#EE3377"
GREY = "#B8C1C6"
LIGHT_GREY = "#E8ECEE"
LIGHT_TEAL = "#E4F3EF"
LIGHT_ORANGE = "#FDEDE5"
LIGHT_MAGENTA = "#FCE8F1"

PRIMARY_KEYS = ["qwen05", "qwen15", "smollm"]
DISPLAY_NAMES = {
    "qwen05": "Qwen2.5-0.5B",
    "qwen15": "Qwen2.5-1.5B",
    "smollm": "SmolLM2-1.7B",
}
SELECTED_LAYERS = {2, 6, 7, 11}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_figure(fig, stem: str):
    fig.savefig(OUT / f"{stem}.pdf")
    fig.savefig(OUT / f"{stem}.png")
    plt.close(fig)


def style_panel(ax, title: str):
    ax.set_title(title, loc="left", fontweight="bold", color=DARK, pad=7)
    ax.set_facecolor(PANEL)
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    ax.tick_params(colors=DARK, width=0.6, length=2.5)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)


def add_card(ax, x, y, w, h, face="white", edge=GRID, radius=0.035, lw=0.9):
    card = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
        clip_on=False,
    )
    ax.add_patch(card)
    return card


def axes_arrow(ax, start, end, color=MUTED, style="-|>", lw=1.1, dashed=False, rad=0.0):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={
            "arrowstyle": style,
            "color": color,
            "linewidth": lw,
            "linestyle": "--" if dashed else "-",
            "mutation_scale": 9,
            "connectionstyle": f"arc3,rad={rad}",
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def get_gsm500_rows():
    rows = load_jsonl(GSM500)
    return {(row["model_key"], row["method"]): row for row in rows}


def get_primary_bits():
    rows = load_json(BIT_BUDGET)["rows"]
    by_model = {row["model"]: row for row in rows if row["role"] == "primary"}
    return {
        "qwen05": by_model["Qwen2.5-0.5B"]["average_weight_bits"],
        "qwen15": by_model["Qwen2.5-1.5B"]["average_weight_bits"],
        "smollm": by_model["SmolLM2-1.7B"]["average_weight_bits"],
    }


def get_layer_sensitivity():
    rows = load_jsonl(PER_LAYER)
    baseline = next(float(row["gsm8k_score"]) for row in rows if row["type"] == "baseline")
    w4_rows = [row for row in rows if row["type"] == "run" and int(row["bits"]) == 4]
    w4_rows.sort(key=lambda row: int(row["layer"]))
    layers = np.array([int(row["layer"]) for row in w4_rows])
    sensitivity = np.array([baseline - float(row["gsm8k_score"]) for row in w4_rows])
    return baseline, layers, sensitivity


def get_divergence_counts():
    rows = load_csv(FIRST_DIV)
    counts = np.zeros(24, dtype=int)
    for row in rows:
        counts[int(row["layer"])] = int(row["count_t099"])
    return counts


def get_replacement_deltas():
    rows = load_csv(LAYER_REPLACE)
    deltas = np.zeros(24, dtype=float)
    for row in rows:
        if row["layer"].isdigit():
            deltas[int(row["layer"])] = float(row["delta"])
    return deltas


def figure1_overview():
    gsm500 = get_gsm500_rows()
    paired = {row["model_key"]: row for row in load_json(PAIRED_STATS)}
    bits = get_primary_bits()
    _, _, sensitivity = get_layer_sensitivity()
    div_counts = get_divergence_counts()

    losses = np.array(
        [gsm500[(key, "fp16")]["accuracy"] - gsm500[(key, "gptq")]["accuracy"] for key in PRIMARY_KEYS]
    )
    mean_loss = float(losses.mean())
    if not np.isclose(mean_loss, 15.6):
        raise ValueError(f"Unexpected primary GSM8K-500 mean loss: {mean_loss:.3f}")

    fig = plt.figure(figsize=(7.1, 3.45), constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.25, 1.65], wspace=0.30)

    # (a) Direct, matched GSM8K-500 losses.
    ax = fig.add_subplot(grid[0, 0])
    style_panel(ax, "(a) Quantization damage")
    xpos = np.arange(3)
    bars = ax.bar(xpos, losses, width=0.62, color=[RED, ORANGE, MAGENTA], edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(mean_loss, color=DARK, linestyle=(0, (4, 2)), linewidth=1.0, zorder=2)
    ax.text(2.48, mean_loss + 0.7, f"mean = {mean_loss:.1f}", ha="right", va="bottom", fontsize=7.4, color=DARK)
    for bar, value in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.7, f"{value:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color=DARK)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["Qwen2.5\n0.5B", "Qwen2.5\n1.5B", "SmolLM2\n1.7B"])
    ax.set_ylabel("GSM8K accuracy loss (pp)")
    ax.set_ylim(0, 25)
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.text(0.02, 0.98, "FP16 -> GPTQ-W4; n=500/model", transform=ax.transAxes, ha="left", va="top", fontsize=6.9, color=MUTED)

    # (b) Diagnostic evidence chain. The first two values share the broad-300 protocol.
    ax = fig.add_subplot(grid[0, 1])
    ax.set_title("(b) From perturbation to failure", loc="left", fontweight="bold", color=DARK, pad=7)
    ax.set_axis_off()
    card_y = [0.73, 0.51, 0.29, 0.07]
    labels = [
        ("Single-layer perturbation", f"worst loss {max(sensitivity):.1f} pp", BLUE),
        ("Full-model GPTQ-W4", "loss 14.6 pp", RED),
        ("Deep hidden-state divergence", f"peak at L{int(np.argmax(div_counts))}", MAGENTA),
        ("Source-directed allocation", "protect {2, 6, 7, 11}", TEAL),
    ]
    for idx, (title, value, color) in enumerate(labels):
        y = card_y[idx]
        add_card(ax, 0.05, y, 0.90, 0.15, face="white", edge=color, radius=0.035, lw=1.1)
        ax.text(0.27, y + 0.100, title, transform=ax.transAxes, ha="left", va="center", fontsize=7.5, fontweight="bold", color=DARK)
        ax.text(0.27, y + 0.048, value, transform=ax.transAxes, ha="left", va="center", fontsize=7.3, color=color, fontweight="bold")

        # Compact visual glyphs that encode the measured object.
        if idx == 0:
            heights = [0.03, 0.07, 0.12, 0.05, 0.09]
            for j, height in enumerate(heights):
                ax.add_patch(patches.Rectangle((0.09 + j * 0.025, y + 0.035), 0.016, height, transform=ax.transAxes, facecolor=BLUE if j == 2 else GREY, edgecolor="none"))
        elif idx == 1:
            for j in range(5):
                ax.add_patch(patches.Rectangle((0.088 + j * 0.026, y + 0.052), 0.021, 0.075, transform=ax.transAxes, facecolor=RED, edgecolor="white", linewidth=0.3))
        elif idx == 2:
            mini = div_counts / max(div_counts.max(), 1)
            for j, height in enumerate(mini[12:]):
                ax.add_patch(patches.Rectangle((0.078 + j * 0.012, y + 0.035), 0.009, 0.105 * height, transform=ax.transAxes, facecolor=MAGENTA, edgecolor="none"))
        else:
            for j in range(8):
                face = TEAL if j in {1, 3, 4} else LIGHT_GREY
                ax.add_patch(patches.Rectangle((0.08 + j * 0.019, y + 0.058), 0.015, 0.065, transform=ax.transAxes, facecolor=face, edgecolor="white", linewidth=0.3))

        if idx < 3:
            axes_arrow(ax, (0.50, y - 0.005), (0.50, card_y[idx + 1] + 0.158), color=MUTED, lw=0.9)

    ax.text(0.50, 0.005, "Loss values: Qwen2.5-0.5B broad-300; divergence: 50 traces", transform=ax.transAxes, ha="center", va="bottom", fontsize=6.8, color=MUTED)

    # (c) Direct paired SG-MMP repair with full model names and average bits.
    ax = fig.add_subplot(grid[0, 2])
    style_panel(ax, "(c) Paired SG-MMP repair")
    y = np.arange(3)[::-1]
    for yi, key in zip(y, PRIMARY_KEYS):
        row = paired[key]
        low, high = row["paired_bootstrap_ci95"]
        delta = row["delta"]
        ax.plot([low, high], [yi, yi], color=TEAL, linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([delta], [yi], s=38, marker="D", color=TEAL, edgecolor="white", linewidth=0.6, zorder=3)
        ax.text(delta, yi + 0.18, f"+{delta:.1f}", ha="center", va="bottom", fontsize=7.1, fontweight="bold", color=TEAL)
        ax.text(18.0, yi, f"{bits[key]:.2f} b", ha="center", va="center", fontsize=7.1, color=DARK, bbox={"boxstyle": "round,pad=0.22", "facecolor": LIGHT_TEAL, "edgecolor": TEAL, "linewidth": 0.7})
    ax.axvline(0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    for yi, key in zip(y, PRIMARY_KEYS):
        ax.text(-7.1, yi, DISPLAY_NAMES[key], ha="left", va="center", fontsize=7.3, color=DARK)
    ax.set_xlim(-7.5, 20.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Accuracy gain over GPTQ-W4 (pp)")
    ax.set_ylim(-0.65, 2.65)
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.text(0.01, 0.98, "GSM8K-500; paired bootstrap 95% CI", transform=ax.transAxes, ha="left", va="top", fontsize=6.9, color=MUTED)
    ax.text(18.0, 2.40, "avg bits", ha="center", va="bottom", fontsize=6.8, color=MUTED)

    fig.subplots_adjust(left=0.065, right=0.985, top=0.91, bottom=0.20)
    save_figure(fig, "concept_fig1_overview_v3")


def draw_process_strip(ax):
    ax.set_axis_off()
    steps = [
        ("1", "Screen each layer", BLUE),
        ("2", "Select sensitive layers", BLUE),
        ("3", "Assign module precision", TEAL),
        ("4", "Quantize and evaluate", ORANGE),
    ]
    xs = [0.09, 0.36, 0.64, 0.91]
    for idx, ((number, label, color), x) in enumerate(zip(steps, xs)):
        ax.add_patch(patches.FancyBboxPatch((x - 0.025, 0.49), 0.05, 0.24, boxstyle="round,pad=0.010,rounding_size=0.025", transform=ax.transAxes, facecolor="white", edgecolor=color, linewidth=1.5))
        ax.text(x, 0.61, number, transform=ax.transAxes, ha="center", va="center", fontsize=8.3, fontweight="bold", color=color)
        ax.text(x, 0.20, label, transform=ax.transAxes, ha="center", va="center", fontsize=7.5, fontweight="bold", color=DARK)
        if idx < len(xs) - 1:
            axes_arrow(ax, (x + 0.055, 0.61), (xs[idx + 1] - 0.055, 0.61), color=MUTED, lw=1.0)


def figure2_method():
    baseline, layers, sensitivity = get_layer_sensitivity()
    ablation = load_json(ABLATION)
    ablation_rows = {row["method"]: row for row in ablation["rows"]}
    paired = {(row["base"], row["other"]): row for row in ablation["paired"]}

    fig = plt.figure(figsize=(7.1, 4.15), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, height_ratios=[0.23, 1.0], width_ratios=[1.05, 1.35, 1.25], hspace=0.20, wspace=0.32)
    draw_process_strip(fig.add_subplot(grid[0, :]))

    # (a) Real sensitivity screen, including negative estimates.
    ax = fig.add_subplot(grid[1, 0])
    style_panel(ax, "(a) Layer sensitivity screen")
    colors = [TEAL if int(layer) in SELECTED_LAYERS else GREY for layer in layers]
    ax.bar(layers, sensitivity, width=0.78, color=colors, edgecolor="white", linewidth=0.35, zorder=3)
    ax.axhline(0, color=DARK, linewidth=0.8)
    for layer in sorted(SELECTED_LAYERS):
        value = sensitivity[layer]
        ax.text(layer, value - 0.28, str(layer), ha="center", va="top", fontsize=6.8, fontweight="bold", color="white")
    ax.set_xlim(-0.8, 23.8)
    ax.set_ylim(-7.0, 5.2)
    ax.set_xticks([0, 2, 6, 7, 11, 16, 23])
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel(r"$S_l$ (pp; positive = loss)")
    ax.text(0.03, 0.97, rf"$S_l=Acc_{{FP16}}-Acc_{{l=W4}}$" + f"\nFP16={baseline:.2f}, GSM8K-300", transform=ax.transAxes, ha="left", va="top", fontsize=7.0, color=MUTED)
    ax.text(0.50, 0.04, "selected: {2, 6, 7, 11}", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.1, fontweight="bold", color=TEAL)

    # (b) Layer strip plus exact module-level assignment matrix.
    ax = fig.add_subplot(grid[1, 1])
    ax.set_title("(b) Precision allocation policy", loc="left", fontweight="bold", color=DARK, pad=7)
    ax.set_axis_off()
    ax.text(0.03, 0.94, "Layer map", transform=ax.transAxes, ha="left", va="center", fontsize=7.3, fontweight="bold", color=DARK)
    x0, y0, width = 0.22, 0.90, 0.73
    gap = 0.006
    bw = (width - gap * 23) / 24
    for layer in range(24):
        face = TEAL if layer in SELECTED_LAYERS else LIGHT_GREY
        edge = TEAL if layer in SELECTED_LAYERS else "white"
        height = 0.080 if layer in SELECTED_LAYERS else 0.055
        ax.add_patch(patches.Rectangle((x0 + layer * (bw + gap), y0 - height / 2), bw, height, transform=ax.transAxes, facecolor=face, edgecolor=edge, linewidth=0.4))
    ax.text(0.585, 0.825, "W8-selected layers: 2, 6, 7, 11", transform=ax.transAxes, ha="center", va="center", fontsize=7.0, color=TEAL, fontweight="bold")

    module_labels = ["q", "k", "v", "o", "g", "u", "d"]
    row_labels = ["Sensitive\nlayers", "Other\nlayers"]
    row_y = [0.59, 0.35]
    cell_x0 = 0.31
    cell_w = 0.077
    cell_gap = 0.012
    for col, label in enumerate(module_labels):
        ax.text(cell_x0 + col * (cell_w + cell_gap) + cell_w / 2, 0.72, label, transform=ax.transAxes, ha="center", va="center", fontsize=6.8, fontweight="bold", color=DARK)
    for ridx, y in enumerate(row_y):
        ax.text(0.04, y + 0.055, row_labels[ridx], transform=ax.transAxes, ha="left", va="center", fontsize=7.1, fontweight="bold", color=DARK)
        for col in range(7):
            is_w8 = ridx == 0 or col < 3
            face = TEAL if is_w8 else ORANGE
            ax.add_patch(patches.FancyBboxPatch((cell_x0 + col * (cell_w + cell_gap), y), cell_w, 0.11, boxstyle="round,pad=0.004,rounding_size=0.010", transform=ax.transAxes, facecolor=face, edgecolor="white", linewidth=0.6))
            ax.text(cell_x0 + col * (cell_w + cell_gap) + cell_w / 2, y + 0.055, "W8" if is_w8 else "W4", transform=ax.transAxes, ha="center", va="center", fontsize=6.8, fontweight="bold", color="white")

    ax.text(0.66, 0.25, "g/u/d = gate/up/down projections", transform=ax.transAxes, ha="center", va="center", fontsize=6.7, color=MUTED)
    add_card(ax, 0.11, 0.07, 0.78, 0.12, face=LIGHT_TEAL, edge=TEAL, radius=0.025, lw=0.9)
    ax.text(0.50, 0.13, "88 W8 / 80 W4 modules   |   avg. 4.90 bits", transform=ax.transAxes, ha="center", va="center", fontsize=7.0, fontweight="bold", color=TEAL)

    # (c) Component effects with paired confidence intervals and cost.
    ax = fig.add_subplot(grid[1, 2])
    style_panel(ax, "(c) Component effects and cost")
    methods = [
        ("Sensitive\nlayers only", "abl_only_sensitive_w8", TEAL),
        ("q/k/v only", "abl_only_qkv_w8", BLUE),
        ("SG-MMP", "sg", TEAL),
        ("Sensitive+FFN\n(high budget)", "abl_sensitive_plus_ffn_w8", ORANGE),
    ]
    y = np.arange(len(methods))[::-1]
    for yi, (label, method, color) in zip(y, methods):
        stats = paired[("gptq", method)]
        low, high = stats["bootstrap_ci95"]
        delta = stats["delta"]
        avg_bit = ablation_rows[method]["avg_bit"]
        ax.plot([low, high], [yi, yi], color=color, linewidth=2.0, solid_capstyle="round", zorder=2)
        ax.scatter([delta], [yi], s=32, marker="D" if method == "sg" else "o", color=color, edgecolor="white", linewidth=0.5, zorder=3)
        ax.text(delta, yi + 0.17, f"+{delta:.1f}", ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=color)
        ax.text(18.2, yi, f"{avg_bit:.2f} b", ha="center", va="center", fontsize=6.9, color=color, bbox={"boxstyle": "round,pad=0.20", "facecolor": LIGHT_ORANGE if color == ORANGE else "white", "edgecolor": color, "linewidth": 0.6})
    ax.axvline(0, color=MUTED, linestyle=(0, (3, 2)), linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([])
    for yi, (label, _, _) in zip(y, methods):
        ax.text(-7.6, yi, label, ha="left", va="center", fontsize=7.0, color=DARK)
    ax.set_xlim(-8.0, 20.5)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_xlabel("Gain over GPTQ-W4 (pp)")
    ax.set_ylim(-0.65, 3.65)
    ax.grid(axis="x", color=GRID, linewidth=0.55, alpha=0.85, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.text(0.02, 0.98, "95% paired CI; n=500", transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=MUTED)
    ax.text(18.2, 3.42, "avg bits", ha="center", va="bottom", fontsize=6.7, color=MUTED)

    fig.subplots_adjust(left=0.070, right=0.985, top=0.96, bottom=0.14)
    save_figure(fig, "concept_fig2_policy_v3")


def draw_mechanism_band(ax, div_counts):
    ax.set_axis_off()
    ax.text(0.01, 0.95, "Mechanistic interpretation", transform=ax.transAxes, ha="left", va="top", fontsize=9.2, fontweight="bold", color=DARK)

    x0, x1 = 0.07, 0.94
    y = 0.37
    gap = 0.004
    bw = (x1 - x0 - gap * 23) / 24
    for layer in range(24):
        face = TEAL if layer in SELECTED_LAYERS else GREY
        height = 0.18 if layer in SELECTED_LAYERS else 0.13
        ax.add_patch(patches.Rectangle((x0 + layer * (bw + gap), y - height / 2), bw, height, transform=ax.transAxes, facecolor=face, edgecolor="white", linewidth=0.35))

    # Measured first-divergence counts are overlaid above the layer strip.
    max_count = max(div_counts.max(), 1)
    for layer, count in enumerate(div_counts):
        if count <= 0:
            continue
        height = 0.23 * count / max_count
        ax.add_patch(patches.Rectangle((x0 + layer * (bw + gap), y + 0.11), bw, height, transform=ax.transAxes, facecolor=MAGENTA, edgecolor="none", alpha=0.95))

    ax.plot([x0 - 0.02, x1 + 0.025], [y, y], transform=ax.transAxes, color=DARK, linewidth=0.8, zorder=0)
    axes_arrow(ax, (x1 + 0.018, y), (0.985, y), color=DARK, lw=0.8)
    ax.text(x0, 0.12, "early", transform=ax.transAxes, ha="center", va="center", fontsize=7.1, color=MUTED)
    ax.text((x0 + x1) / 2, 0.12, "middle", transform=ax.transAxes, ha="center", va="center", fontsize=7.1, color=MUTED)
    ax.text(x1, 0.12, "deep", transform=ax.transAxes, ha="center", va="center", fontsize=7.1, color=MUTED)

    # Dashed paths are explicitly interpretations rather than direct measurements.
    axes_arrow(ax, (0.20, 0.67), (0.72, 0.67), color=RED, lw=1.2, dashed=True, rad=0.0)
    axes_arrow(ax, (0.42, 0.73), (0.90, 0.73), color=RED, lw=1.2, dashed=True, rad=0.0)
    ax.text(0.58, 0.80, "inferred accumulation across quantized layers", transform=ax.transAxes, ha="center", va="center", fontsize=7.2, color=RED, fontweight="bold", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0})

    ax.text(0.22, 0.78, "SG-MMP source intervention", transform=ax.transAxes, ha="center", va="center", fontsize=7.2, color=TEAL, fontweight="bold")
    axes_arrow(ax, (0.22, 0.74), (0.25, 0.49), color=TEAL, lw=1.1)
    peak = int(np.argmax(div_counts))
    ax.text(0.88, 0.91, f"observed peak: L{peak}", transform=ax.transAxes, ha="center", va="center", fontsize=7.2, color=MAGENTA, fontweight="bold")
    axes_arrow(ax, (0.88, 0.86), (0.94, 0.70), color=MAGENTA, lw=1.0)

    ax.add_patch(patches.Rectangle((0.08, 0.02), 0.018, 0.025, transform=ax.transAxes, facecolor=TEAL, edgecolor="none"))
    ax.text(0.105, 0.032, "measured selected layers", transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color=MUTED)
    ax.add_patch(patches.Rectangle((0.31, 0.02), 0.018, 0.025, transform=ax.transAxes, facecolor=MAGENTA, edgecolor="none"))
    ax.text(0.335, 0.032, "measured divergence counts", transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color=MUTED)
    ax.plot([0.61, 0.66], [0.032, 0.032], transform=ax.transAxes, color=RED, linestyle="--", linewidth=1.2)
    ax.text(0.675, 0.032, "interpretation", transform=ax.transAxes, ha="left", va="center", fontsize=6.8, color=MUTED)


def figure3_mechanism():
    _, layers, sensitivity = get_layer_sensitivity()
    div_counts = get_divergence_counts()
    replacement = get_replacement_deltas()

    worst_single = float(max(sensitivity))
    full_broad_loss = 14.6

    fig = plt.figure(figsize=(7.1, 4.15), constrained_layout=False)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.80], width_ratios=[0.90, 1.45, 1.25], hspace=0.30, wspace=0.40)
    draw_mechanism_band(fig.add_subplot(grid[0, :]), div_counts)

    # (a) Local perturbation does not reproduce full-model damage.
    ax = fig.add_subplot(grid[1, 0])
    style_panel(ax, "(a) Local vs full (n=300)")
    values = [worst_single, full_broad_loss]
    bars = ax.bar([0, 1], values, width=0.58, color=[BLUE, RED], edgecolor="white", linewidth=0.5, zorder=3)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}", ha="center", va="bottom", fontsize=7.4, fontweight="bold", color=DARK)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Worst single\nW4 layer", "Full\nGPTQ-W4"])
    ax.set_ylabel("GSM8K loss (pp)")
    ax.set_ylim(0, 18)

    # (b) Show all replacement interventions to avoid cherry-picking only positive layers.
    ax = fig.add_subplot(grid[1, 1])
    style_panel(ax, "(b) Layer replacement (n=100)")
    colors = [TEAL if value > 0 else LIGHT_GREY for value in replacement]
    ax.bar(np.arange(24), replacement, width=0.78, color=colors, edgecolor="white", linewidth=0.3, zorder=3)
    ax.axhline(0, color=DARK, linewidth=0.8)
    for layer in [1, 6, 11]:
        ax.text(layer, replacement[layer] + 0.35, f"L{layer}", ha="center", va="bottom", fontsize=6.8, fontweight="bold", color=TEAL)
    ax.set_xlim(-0.8, 23.8)
    ax.set_ylim(-8, 4.5)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.set_xlabel("Layer restored to FP16")
    ax.set_ylabel("Delta vs GPTQ-W4 (pp)")
    ax.text(0.98, 0.04, "positive = recovery", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.7, color=MUTED)

    # (c) First-divergent-layer distribution from all 50 traces.
    ax = fig.add_subplot(grid[1, 2])
    style_panel(ax, "(c) First divergent layer (n=50)")
    colors = [MAGENTA if layer >= 12 and count > 0 else GREY for layer, count in enumerate(div_counts)]
    ax.bar(np.arange(24), div_counts, width=0.78, color=colors, edgecolor="white", linewidth=0.3, zorder=3)
    peak = int(np.argmax(div_counts))
    ax.annotate(f"L{peak}: {div_counts[peak]}/50", xy=(peak, div_counts[peak]), xytext=(18.2, div_counts[peak] + 0.5), ha="center", va="bottom", fontsize=7.1, fontweight="bold", color=MAGENTA, arrowprops={"arrowstyle": "-|>", "color": MAGENTA, "linewidth": 0.8})
    ax.set_xlim(-0.8, 23.8)
    ax.set_ylim(0, 20.5)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 23])
    ax.set_xlabel("Transformer layer index")
    ax.set_ylabel("First-divergence count")
    ax.text(0.03, 0.97, "cosine threshold = 0.99", transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=MUTED)

    fig.subplots_adjust(left=0.070, right=0.985, top=0.97, bottom=0.14)
    save_figure(fig, "concept_fig3_error_propagation_v3")


def main():
    figure1_overview()
    figure2_method()
    figure3_mechanism()
    print("Generated v3 concept figures from released JSON/CSV sources")


if __name__ == "__main__":
    main()

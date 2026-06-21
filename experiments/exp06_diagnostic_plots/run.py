"""实验1：通用 benchmark vs GSM8K 背离图。"""
import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

RESULTS_FILE = "results/task_results_full.jsonl"
OUT_DIR = "experiments/exp06_diagnostic_plots"
os.makedirs(OUT_DIR, exist_ok=True)

FP16_TASKS = ["arc_challenge", "hellaswag", "mmlu", "gsm8k"]
# short key mapping
TASK_KEYS = {"arc_challenge": "ret_arc", "hellaswag": "ret_hella", "mmlu": "ret_mmlu", "gsm8k": "ret_gsm8k"}

METHODS_TO_INCLUDE = [
    "rtn", "gptq", "awq", "smoothquant",
    "onecomp_qep_g0.0", "onecomp_qep_g1.0",
    "onecomp_qep_g0.0_64x1024", "onecomp_qep_g1.0_64x1024",
    "hadamard_gptq",
    "mixed_precision", "config_b",
    "config_b_lora", "config_b_lora_v2",
]

# ---- load data ----
records = []
with open(RESULTS_FILE) as f:
    for line in f:
        r = json.loads(line)
        if r["model"] == "Qwen2.5-0.5B":
            records.append(r)

fp16 = None
for r in records:
    if r["method"] == "fp16":
        fp16 = r["scores"]
        break
assert fp16 is not None, "fp16 baseline missing"
base = {t: fp16[t] for t in FP16_TASKS}
print(f"FP16 baseline: {base}")

# ---- compute retention ----
data = {}
for method in METHODS_TO_INCLUDE:
    found = [r for r in records if r["method"] == method]
    if not found:
        print(f"  SKIP {method}: no data")
        continue
    scores = found[0]["scores"]
    entry = {"method": method, "scores": scores}
    for task in FP16_TASKS:
        key = TASK_KEYS[task]
        if task in scores and scores[task] is not None:
            entry[key] = scores[task] / base[task] * 100
    data[method] = entry

all_keys = ["ret_arc", "ret_hella", "ret_mmlu", "ret_gsm8k"]
print(f"\nLoaded {len(data)} methods")
for m, e in data.items():
    present = [k for k in all_keys if e.get(k) is not None]
    print(f"  {m}: {len(present)}/4 tasks")

# ---- plot helper ----
def plot_scatter(x_key, x_label):
    xs, ys, labels = [], [], []
    for method, e in data.items():
        x = e.get(x_key)
        y = e.get("ret_gsm8k")
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(y)
        labels.append(method)

    if len(xs) < 3:
        print(f"  {x_label}: only {len(xs)} points, skip R^2")
        return None

    arr_x = np.array(xs).reshape(-1, 1)
    arr_y = np.array(ys)
    reg = LinearRegression().fit(arr_x, arr_y)
    r2 = reg.score(arr_x, arr_y)
    slope = reg.coef_[0]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(arr_x, arr_y, s=80, alpha=0.8, edgecolors="black", linewidth=0.5)

    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (arr_x[i][0], arr_y[i]), fontsize=7,
                    textcoords="offset points", xytext=(6, 4),
                    bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.7))

    x_line = np.linspace(min(arr_x) - 2, max(arr_x) + 2, 100).reshape(-1, 1)
    y_line = reg.predict(x_line)
    ax.plot(x_line, y_line, "r--", alpha=0.6, label=f"R^2={r2:.3f}, slope={slope:.2f}")

    ax.axhline(y=100, color="green", linestyle=":", alpha=0.4, label="FP16 GSM8K")
    ax.axvline(x=100, color="green", linestyle=":", alpha=0.4, label="FP16 reference")

    ax.set_xlabel(f"{x_label} retention rate (%)", fontsize=12)
    ax.set_ylabel("GSM8K retention rate (%)", fontsize=12)
    ax.set_title(f"Qwen2.5-0.5B: {x_label} vs GSM8K Retention (R^2={r2:.3f})", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fname = os.path.join(OUT_DIR, f"scatter_{x_key}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  Saved: {fname}  R^2={r2:.4f}  n={len(xs)}")
    return r2

# ---- generate ----
print("\nGenerating plots...")
r2_arc = plot_scatter("ret_arc", "ARC-Challenge")
r2_hella = plot_scatter("ret_hella", "HellaSwag")
r2_mmlu = plot_scatter("ret_mmlu", "MMLU")

print(f"\n{'='*60}")
print("R^2 Summary:")
if r2_arc:
    print(f"  ARC-Challenge vs GSM8K: R^2={r2_arc:.4f}")
if r2_hella:
    print(f"  HellaSwag vs GSM8K:    R^2={r2_hella:.4f}")
if r2_mmlu:
    print(f"  MMLU vs GSM8K:         R^2={r2_mmlu:.4f}")

all_r2 = [v for v in [r2_arc, r2_hella, r2_mmlu] if v is not None]
if all_r2 and max(all_r2) < 0.3:
    print("\n*** PAPER-LEVEL FINDING: All R^2 < 0.3 ***")
    print("GSM8K error pattern is independent of generic benchmarks!")
elif all_r2:
    print(f"\n  Max R^2 = {max(all_r2):.3f}")

print(f"\nPlots saved to: {OUT_DIR}/")

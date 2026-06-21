"""GSM8K Wilson 95% confidence interval analysis.

Reads results/task_results_full.jsonl, computes Wilson CI for all methods
with gsm8k scores. Outputs Markdown to paper/analysis_gsm8k_ci.md.
"""
import json, math, os

RESULTS_FILE = "results/task_results_full.jsonl"
OUTPUT_FILE = "paper/analysis_gsm8k_ci.md"

# Methods to highlight in the main table (order preserved)
MAIN_METHODS = [
    "fp16", "gptq", "awq", "smoothquant", "config_b",
    "config_b_samebudget_300",
    "random_42_samebudget_300", "random_123_samebudget_300",
    "random_456_samebudget_300",
    "first_4_samebudget_300", "last_4_samebudget_300",
    "config_b_2a", "config_b_2b",
    "config_b_lora", "config_b_lora_v2", "config_b_failure_lora_6c",
]

LEGACY_METHODS = [
    "random_42", "random_123", "random_456",
    "first_4", "last_4",
]

MODEL_ORDER = ["Qwen2.5-0.5B", "Qwen2.5-1.5B", "SmolLM-1.7B"]


def wilson_ci(score, n, z=1.96):
    """Wilson score interval for binomial proportion."""
    if n <= 0 or score is None:
        return None, None, None
    p = score / 100.0
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center * 100, (center - margin) * 100, (center + margin) * 100


def load_results():
    records = {}
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["model"], r["method"])
            if key not in records:
                records[key] = {}
            records[key].update(r["scores"])
    return records


def get_n(method, model):
    """Return GSM8K sample size for a given method."""
    if "_samebudget_300" in method or method.endswith("_300"):
        return 300
    if method in LEGACY_METHODS:
        return 100
    return 300


def build_table(records, methods, title, n_override=None):
    lines = [f"### {title}", ""]
    header = "| Model | Method | GSM8K | CI | n |"
    sep = "|---|---:|---:|---:|--:|"
    lines.extend([header, sep])

    for model in MODEL_ORDER:
        for method in methods:
            key = (model, method)
            if key not in records or "gsm8k" not in records[key]:
                continue
            score = records[key]["gsm8k"]
            n = n_override if n_override else get_n(method, model)
            _, lo, hi = wilson_ci(score, n)
            ci_str = f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "—"
            lines.append(f"| {model} | {method} | {score:.2f} | {ci_str} | {n} |")

    if len(lines) == 3:  # no data rows
        lines.append("| — | — | — | — | — |")
    lines.append("")
    return lines


def main():
    records = load_results()

    # Separate SmolLM rows
    smol_main = [m for m in MAIN_METHODS
                 if ("SmolLM-1.7B", m) in records and "gsm8k" in records[("SmolLM-1.7B", m)]]
    nonsmol_main = [m for m in MAIN_METHODS
                    if any((mdl, m) in records and "gsm8k" in records[(mdl, m)]
                           for mdl in MODEL_ORDER if mdl != "SmolLM-1.7B")]

    # Collect all other methods with gsm8k
    seen = set(MAIN_METHODS) | set(LEGACY_METHODS)
    other = []
    for (model, method), scores in records.items():
        if "gsm8k" in scores and method not in seen:
            other.append((model, method))
    other.sort(key=lambda x: (MODEL_ORDER.index(x[0]) if x[0] in MODEL_ORDER else 99, x[1]))

    lines = ["# GSM8K Wilson 95% Confidence Interval Analysis", ""]
    lines.append(f"n=300 for most methods; legacy same-budget uses n=100.")
    lines.append("Wilson score interval, z=1.96 for 95% CI.")
    lines.append("")

    lines.extend(build_table(records, [
        m for m in MAIN_METHODS if m not in ["config_b_samebudget_300",
        "random_42_samebudget_300", "random_123_samebudget_300",
        "random_456_samebudget_300", "first_4_samebudget_300",
        "last_4_samebudget_300",
        "config_b_2a", "config_b_2b"]
    ], "Main Table (0.5B)"))

    lines.extend(build_table(records, [
        "config_b_samebudget_300", "random_42_samebudget_300",
        "random_123_samebudget_300", "random_456_samebudget_300",
        "first_4_samebudget_300", "last_4_samebudget_300",
    ], "Same-Budget Comparison (0.5B, limit=300)"))

    lines.extend(build_table(records, [
        "config_b_2a", "config_b_2b",
    ], "Qwen2.5-1.5B config_b Migration"))

    if smol_main:
        lines.extend(build_table(records, smol_main, "SmolLM-1.7B"))

    lines.extend(build_table(records, LEGACY_METHODS, "Legacy (old same-budget, n=100)", n_override=100))

    if other:
        lines.append("### Other Methods with GSM8K")
        lines.append("")
        header = "| Model | Method | GSM8K | CI | n |"
        sep = "|---|---:|---:|---:|--:|"
        lines.extend([header, sep])
        for model, method in other:
            score = records[(model, method)]["gsm8k"]
            n = get_n(method, model)
            _, lo, hi = wilson_ci(score, n)
            ci_str = f"[{lo:.1f}, {hi:.1f}]" if lo is not None else "—"
            lines.append(f"| {model} | {method} | {score:.2f} | {ci_str} | {n} |")
        lines.append("")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Written: {OUTPUT_FILE} ({len(lines)} lines)")


if __name__ == "__main__":
    main()

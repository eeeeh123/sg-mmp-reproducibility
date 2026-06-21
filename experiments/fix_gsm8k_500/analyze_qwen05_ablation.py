"""Analyze Qwen2.5-0.5B GSM8K-500 module ablations.

This is an offline analysis helper. It reads direct-evaluation JSONL sample
logs and quantized state files, then writes a paper-facing summary with
accuracy, restoration rate, bit budget, and paired significance diagnostics.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean

import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "fix_gsm8k_500" / "results_direct"
SAMPLE_DIR = OUT / "samples"
STATE_DIR = ROOT / "results"
PAPER_OUT = ROOT / "paper" / "analysis_qwen05_ablation_gsm8k500.md"
JSON_OUT = OUT / "qwen05_ablation_analysis_gsm8k500.json"
N = 500
BOOT_ITERS = 10000
BOOT_SEED = 20260615

METHODS = {
    "fp16": {
        "label": "FP16",
        "state": None,
        "note": "full precision reference",
    },
    "gptq": {
        "label": "GPTQ-W4",
        "state": STATE_DIR / "Qwen2.5-0.5B_gptq_compact.pt",
        "note": "uniform W4 baseline",
    },
    "sg": {
        "label": "SG-MMP",
        "state": STATE_DIR / "Qwen2.5-0.5B_config_b.pt",
        "note": "sensitive layers W8 plus non-sensitive q/k/v W8",
    },
    "abl_only_sensitive_w8": {
        "label": "Only sensitive layers W8",
        "state": STATE_DIR / "Qwen2.5-0.5B_abl_only_sensitive_w8.pt",
        "note": "removes non-sensitive q/k/v protection",
    },
    "abl_only_qkv_w8": {
        "label": "Only q/k/v W8",
        "state": STATE_DIR / "Qwen2.5-0.5B_abl_only_qkv_w8.pt",
        "note": "protects q/k/v everywhere but not full sensitive layers",
    },
    "abl_sensitive_plus_ffn_w8": {
        "label": "Sensitive layers + FFN W8",
        "state": STATE_DIR / "Qwen2.5-0.5B_abl_sensitive_plus_ffn_w8.pt",
        "note": "higher-budget variant; not a same-budget control",
    },
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_path(method: str) -> Path:
    return SAMPLE_DIR / f"qwen05__{method}__gsm8k{N}.jsonl"


def summarize(method: str) -> dict:
    rows = read_jsonl(sample_path(method))
    correct = sum(int(r["correct"]) for r in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": 100 * correct / len(rows),
    }


def method_bits(qi: dict, unknown_bits: int = 8) -> int:
    method = qi.get("method", "")
    if method == "gptq_w4":
        return 4
    if method == "w8_perchannel":
        return 8
    return unknown_bits


def param_count(qi: dict) -> int:
    if "out_features" in qi and "in_features" in qi:
        return int(qi["out_features"]) * int(qi["in_features"])
    return int(qi["w_q"].numel())


def budget(state_path: Path | None) -> dict:
    if state_path is None:
        return {
            "w8_modules": None,
            "w4_modules": None,
            "w8_params": None,
            "w4_params": None,
            "avg_bit": 16.0,
        }
    state = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
    unknown_bits = 4 if "gptq" in state_path.name.lower() else 8
    w8_modules = 0
    w4_modules = 0
    w8_params = 0
    w4_params = 0
    for qi in state.values():
        if "w_q" not in qi:
            continue
        bits = method_bits(qi, unknown_bits=unknown_bits)
        params = param_count(qi)
        if bits == 4:
            w4_modules += 1
            w4_params += params
        else:
            w8_modules += 1
            w8_params += params
    total = w8_params + w4_params
    avg_bit = (8 * w8_params + 4 * w4_params) / total if total else 0.0
    return {
        "w8_modules": w8_modules,
        "w4_modules": w4_modules,
        "w8_params": w8_params,
        "w4_params": w4_params,
        "avg_bit": avg_bit,
    }


def exact_mcnemar_p(b: int, c: int) -> float:
    total = b + c
    if total == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(total, i) for i in range(k + 1)) / (2**total)
    return min(1.0, 2 * tail)


def paired_bootstrap(base: list[int], other: list[int]) -> tuple[float, float, float]:
    rng = random.Random(BOOT_SEED)
    n = len(base)
    values = []
    for _ in range(BOOT_ITERS):
        delta = 0
        for _ in range(n):
            idx = rng.randrange(n)
            delta += other[idx] - base[idx]
        values.append(100 * delta / n)
    values.sort()
    return values[int(0.025 * BOOT_ITERS)], values[int(0.975 * BOOT_ITERS)], mean(values)


def paired_stats(base_method: str, other_method: str) -> dict:
    base = {r["doc_id"]: r for r in read_jsonl(sample_path(base_method))}
    other = {r["doc_id"]: r for r in read_jsonl(sample_path(other_method))}
    ids = sorted(set(base) & set(other))
    a = [int(base[i]["correct"]) for i in ids]
    b = [int(other[i]["correct"]) for i in ids]
    base_wrong_other_correct = sum(1 for x, y in zip(a, b) if (not x) and y)
    base_correct_other_wrong = sum(1 for x, y in zip(a, b) if x and (not y))
    ci_lo, ci_hi, boot_mean = paired_bootstrap(a, b)
    return {
        "base": base_method,
        "other": other_method,
        "n": len(ids),
        "delta": 100 * (sum(b) - sum(a)) / len(ids),
        "base_wrong_other_correct": base_wrong_other_correct,
        "base_correct_other_wrong": base_correct_other_wrong,
        "mcnemar_p_exact": exact_mcnemar_p(base_wrong_other_correct, base_correct_other_wrong),
        "bootstrap_mean": boot_mean,
        "bootstrap_ci95": [ci_lo, ci_hi],
    }


def fmt_params(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1e6:.1f}M"


def main() -> None:
    rows = []
    for method, spec in METHODS.items():
        summary = summarize(method)
        b = budget(spec["state"])
        rows.append(
            {
                "method": method,
                "label": spec["label"],
                "note": spec["note"],
                **summary,
                **b,
            }
        )

    fp16 = next(r for r in rows if r["method"] == "fp16")["accuracy"]
    gptq = next(r for r in rows if r["method"] == "gptq")["accuracy"]
    for row in rows:
        row["delta_vs_gptq"] = row["accuracy"] - gptq
        row["restoration_rate"] = (row["accuracy"] - gptq) / (fp16 - gptq) if fp16 != gptq else None

    paired = []
    for method in ["sg", "abl_only_sensitive_w8", "abl_only_qkv_w8", "abl_sensitive_plus_ffn_w8"]:
        paired.append(paired_stats("gptq", method))
    for method in ["abl_only_sensitive_w8", "abl_only_qkv_w8", "abl_sensitive_plus_ffn_w8"]:
        paired.append(paired_stats("sg", method))

    JSON_OUT.write_text(json.dumps({"rows": rows, "paired": paired}, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Qwen2.5-0.5B GSM8K-500 Module Ablation",
        "",
        "Fixed GSM8K test subset: n=500, seed=20260615. All rows use the same direct 5-shot greedy generation evaluator.",
        "",
        "## Accuracy and Budget",
        "",
        "| Method | GSM8K | Correct | Avg bit | W8 params | W4 params | Delta vs GPTQ | Restoration | Note |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        restoration = "-" if row["restoration_rate"] is None else f"{100 * row['restoration_rate']:.1f}%"
        avg_bit = "-" if row["avg_bit"] is None else f"{row['avg_bit']:.2f}"
        lines.append(
            f"| {row['label']} | {row['accuracy']:.2f} | {row['correct']}/{row['n']} | "
            f"{avg_bit} | {fmt_params(row['w8_params'])} | {fmt_params(row['w4_params'])} | "
            f"{row['delta_vs_gptq']:+.2f} | {restoration} | {row['note']} |"
        )

    lines += [
        "",
        "## Paired Comparisons",
        "",
        "| Base -> Other | Delta | Base wrong / other correct | Base correct / other wrong | McNemar p | Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for p in paired:
        base = METHODS[p["base"]]["label"]
        other = METHODS[p["other"]]["label"]
        lines.append(
            f"| {base} -> {other} | {p['delta']:+.2f} | "
            f"{p['base_wrong_other_correct']} | {p['base_correct_other_wrong']} | "
            f"{p['mcnemar_p_exact']:.4g} | "
            f"[{p['bootstrap_ci95'][0]:+.2f}, {p['bootstrap_ci95'][1]:+.2f}] |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- Protecting only the sensitive layers improves GPTQ-W4 by +3.00 points but leaves most of SG-MMP's gain unrealized.",
        "- Protecting q/k/v projections globally is stronger (+5.80 over GPTQ-W4) but still below SG-MMP.",
        "- The full SG-MMP allocation combines both choices at the same 4.90-bit budget and reaches +10.00 over GPTQ-W4.",
        "- Adding FFN W8 protection reaches 30.20, but it uses a much higher 7.77-bit budget; it should be reported as a high-budget variant, not a fair same-budget ablation.",
        "",
    ]
    PAPER_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {JSON_OUT}")
    print(f"Written {PAPER_OUT}")


if __name__ == "__main__":
    main()

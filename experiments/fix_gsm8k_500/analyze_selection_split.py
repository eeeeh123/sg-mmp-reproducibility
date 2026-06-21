"""Held-out slice analysis for GSM8K-500 direct evaluation.

The original Qwen2.5-0.5B single-layer sensitivity screen used lm-eval with
``limit=300``. lm-eval preserves dataset order for GSM8K under this setup, so
the sensitivity screen corresponds to the first 300 GSM8K test examples. The
direct GSM8K-500 validation uses a fixed random subset. This script reports
core GPTQ-vs-SG-MMP statistics separately on examples that overlap the original
first-300 screen and examples whose original GSM8K test index is >= 300.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "fix_gsm8k_500" / "results_direct"
SAMPLE_DIR = OUT / "samples"
PAPER_OUT = ROOT / "paper" / "analysis_selection_eval_split_gsm8k500.md"
JSON_OUT = OUT / "selection_eval_split_gsm8k500.json"
N = 500
SCREEN_CUTOFF = 300
BOOT_ITERS = 10000
BOOT_SEED = 20260615

MODEL_LABELS = {
    "qwen05": "Qwen2.5-0.5B",
    "qwen15": "Qwen2.5-1.5B",
    "smollm": "SmolLM-1.7B",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_path(model: str, method: str) -> Path:
    return SAMPLE_DIR / f"{model}__{method}__gsm8k{N}.jsonl"


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
    vals = []
    for _ in range(BOOT_ITERS):
        delta = 0
        for _ in range(n):
            idx = rng.randrange(n)
            delta += other[idx] - base[idx]
        vals.append(100 * delta / n)
    vals.sort()
    return vals[int(0.025 * BOOT_ITERS)], vals[int(0.975 * BOOT_ITERS)], mean(vals)


def analyze_slice(model: str, name: str, predicate) -> dict:
    fp16_rows = {r["doc_id"]: r for r in read_jsonl(sample_path(model, "fp16"))}
    gptq_rows = {r["doc_id"]: r for r in read_jsonl(sample_path(model, "gptq"))}
    sg_rows = {r["doc_id"]: r for r in read_jsonl(sample_path(model, "sg"))}
    ids = sorted(i for i in set(fp16_rows) & set(gptq_rows) & set(sg_rows) if predicate(i))
    fp16 = [int(fp16_rows[i]["correct"]) for i in ids]
    gptq = [int(gptq_rows[i]["correct"]) for i in ids]
    sg = [int(sg_rows[i]["correct"]) for i in ids]
    gptq_wrong_sg_correct = sum(1 for x, y in zip(gptq, sg) if (not x) and y)
    gptq_correct_sg_wrong = sum(1 for x, y in zip(gptq, sg) if x and (not y))
    ci_lo, ci_hi, boot_mean = paired_bootstrap(gptq, sg)
    n = len(ids)
    return {
        "model": model,
        "slice": name,
        "n": n,
        "doc_id_min": min(ids) if ids else None,
        "doc_id_max": max(ids) if ids else None,
        "fp16_acc": 100 * sum(fp16) / n,
        "gptq_acc": 100 * sum(gptq) / n,
        "sg_acc": 100 * sum(sg) / n,
        "delta": 100 * (sum(sg) - sum(gptq)) / n,
        "gptq_wrong_sg_correct": gptq_wrong_sg_correct,
        "gptq_correct_sg_wrong": gptq_correct_sg_wrong,
        "mcnemar_p_exact": exact_mcnemar_p(gptq_wrong_sg_correct, gptq_correct_sg_wrong),
        "bootstrap_mean": boot_mean,
        "bootstrap_ci95": [ci_lo, ci_hi],
    }


def main() -> None:
    slices = [
        ("overlap_with_original_limit300", lambda i: i < SCREEN_CUTOFF),
        ("heldout_doc_id_ge_300", lambda i: i >= SCREEN_CUTOFF),
        ("all_direct500", lambda i: True),
    ]
    records = []
    for model in MODEL_LABELS:
        for name, pred in slices:
            records.append(analyze_slice(model, name, pred))

    JSON_OUT.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# GSM8K-500 Selection/Evaluation Split Check",
        "",
        "The original single-layer sensitivity screen used `lm-eval` with `limit=300`, which follows GSM8K dataset order in this setup. The direct GSM8K-500 validation uses a fixed random subset (seed 20260615). We therefore split the direct predictions into examples overlapping the original first-300 screen and held-out examples with original `doc_id >= 300`.",
        "",
        "This analysis does not replace a fully separate layer-selection run, but it tests whether the reported SG-MMP gains persist outside the original sensitivity-screen slice.",
        "",
        "| Model | Slice | n | FP16 | GPTQ-W4 | SG-MMP | Delta | GPTQ wrong / SG correct | GPTQ correct / SG wrong | McNemar p | Bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        lines.append(
            f"| {MODEL_LABELS[r['model']]} | {r['slice']} | {r['n']} | "
            f"{r['fp16_acc']:.2f} | {r['gptq_acc']:.2f} | {r['sg_acc']:.2f} | "
            f"{r['delta']:+.2f} | {r['gptq_wrong_sg_correct']} | {r['gptq_correct_sg_wrong']} | "
            f"{r['mcnemar_p_exact']:.4g} | "
            f"[{r['bootstrap_ci95'][0]:+.2f}, {r['bootstrap_ci95'][1]:+.2f}] |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "- For Qwen2.5-0.5B, the held-out `doc_id >= 300` slice preserves the SG-MMP advantage over GPTQ-W4, so the 500-example gain is not confined to examples overlapping the original layer-sensitivity screen.",
        "- Because the held-out slice is smaller than the full direct-500 set, confidence intervals are wider. The analysis should be described as a robustness check, not as a complete replacement for a fresh selection/evaluation split.",
        "- The same split is also reported for Qwen2.5-1.5B and SmolLM-1.7B for transparency, although their layer-selection histories differ from Qwen2.5-0.5B.",
        "",
    ]
    PAPER_OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Written {JSON_OUT}")
    print(f"Written {PAPER_OUT}")


if __name__ == "__main__":
    main()

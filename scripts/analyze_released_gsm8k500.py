"""Recompute paired GSM8K-500 statistics from public redacted outcomes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path


BOOTSTRAP_SEED = 20260615
BOOTSTRAP_ITERS = 10_000
MODEL_LABELS = {
    "qwen05": "Qwen2.5-0.5B",
    "qwen15": "Qwen2.5-1.5B",
    "smollm": "SmolLM2-1.7B",
    "gemma2": "Gemma-2-2B-it",
    "tinyllama": "TinyLlama-1.1B intermediate",
}


def exact_mcnemar_p(b: int, c: int) -> float:
    total = b + c
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, i) for i in range(min(b, c) + 1)) / (2**total)
    return min(1.0, 2 * tail)


def paired_bootstrap(gptq: list[int], sg: list[int]) -> tuple[float, float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(gptq)
    values = []
    for _ in range(BOOTSTRAP_ITERS):
        delta = 0
        for _ in range(n):
            index = rng.randrange(n)
            delta += sg[index] - gptq[index]
        values.append(100 * delta / n)
    values.sort()
    return values[int(0.025 * BOOTSTRAP_ITERS)], values[int(0.975 * BOOTSTRAP_ITERS)], sum(values) / len(values)


def load_rows(path: Path) -> dict[tuple[str, str], dict[int, int]]:
    groups: dict[tuple[str, str], dict[int, int]] = defaultdict(dict)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[(row["model_key"], row["method"])][int(row["doc_id"])] = int(row["correct"])
    return groups


def analyze(path: Path) -> list[dict]:
    groups = load_rows(path)
    output = []
    for model_key in sorted({model for model, _ in groups}):
        gptq = groups.get((model_key, "gptq"))
        sg = groups.get((model_key, "sg"))
        if not gptq or not sg:
            continue
        ids = sorted(set(gptq) & set(sg))
        if len(ids) != 500:
            raise ValueError(f"{model_key}: expected 500 paired rows, found {len(ids)}")
        a = [gptq[index] for index in ids]
        b = [sg[index] for index in ids]
        repairs = sum(1 for left, right in zip(a, b) if not left and right)
        losses = sum(1 for left, right in zip(a, b) if left and not right)
        low, high, mean = paired_bootstrap(a, b)
        output.append(
            {
                "model_key": model_key,
                "model": MODEL_LABELS.get(model_key, model_key),
                "n": len(ids),
                "gptq_accuracy": round(100 * sum(a) / len(a), 2),
                "sg_mmp_accuracy": round(100 * sum(b) / len(b), 2),
                "delta": round(100 * (sum(b) - sum(a)) / len(a), 2),
                "sg_repairs": repairs,
                "sg_losses": losses,
                "mcnemar_p_exact": exact_mcnemar_p(repairs, losses),
                "paired_bootstrap_mean": round(mean, 3),
                "paired_bootstrap_ci95": [round(low, 2), round(high, 2)],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/gsm8k500/per_example_correctness.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/gsm8k500/recomputed_paired_stats.json"),
    )
    args = parser.parse_args()
    results = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

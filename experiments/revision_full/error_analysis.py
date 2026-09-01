"""Full-test error audit plus blinded manual error-taxonomy preparation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")

from experiments.revision_full.analyze import read_jsonl, sample_path
from experiments.revision_full.protocol import (
    GSM8K_TEST_SIZE,
    MODEL_SPECS,
    RANDOM_CALIB_SEED,
    RESULTS_DIR,
    method_id,
)


OUT = RESULTS_DIR / "error_analysis"
LABELS = {
    "arithmetic",
    "reasoning_setup",
    "state_tracking",
    "extraction_or_format",
    "truncation",
    "hallucination",
    "other",
}


def rows_by_id(model_key: str, method: str) -> dict[int, dict]:
    path = sample_path(model_key, method)
    rows = {int(row["doc_id"]): row for row in read_jsonl(path)}
    if set(rows) != set(range(GSM8K_TEST_SIZE)):
        raise RuntimeError(f"Expected {GSM8K_TEST_SIZE} complete rows in {path}")
    return rows


def transition(a: dict, b: dict) -> str:
    key = (int(a["correct"]), int(b["correct"]))
    return {
        (1, 1): "both_correct",
        (0, 0): "both_wrong",
        (0, 1): "a_wrong_b_correct",
        (1, 0): "a_correct_b_wrong",
    }[key]


def automatic_audit(model_key: str, method_a: str, method_b: str) -> tuple[dict, list[dict]]:
    a_rows = rows_by_id(model_key, method_a)
    b_rows = rows_by_id(model_key, method_b)
    records = []
    for doc_id in range(GSM8K_TEST_SIZE):
        a = a_rows[doc_id]
        b = b_rows[doc_id]
        records.append(
            {
                "doc_id": doc_id,
                "transition": transition(a, b),
                "a_parse_failure": a.get("prediction") is None,
                "b_parse_failure": b.get("prediction") is None,
                "a_truncated": bool(a.get("truncated", False)),
                "b_truncated": bool(b.get("truncated", False)),
            }
        )
    summary = {
        "model_key": model_key,
        "method_a": method_a,
        "method_b": method_b,
        "n": GSM8K_TEST_SIZE,
        "transitions": dict(Counter(row["transition"] for row in records)),
        "parse_failures": {
            "a": sum(row["a_parse_failure"] for row in records),
            "b": sum(row["b_parse_failure"] for row in records),
        },
        "truncations": {
            "a": sum(row["a_truncated"] for row in records),
            "b": sum(row["b_truncated"] for row in records),
        },
    }
    return summary, records


def prepare_annotation(model_key: str, calib_seed: int, sample_size: int) -> None:
    method_a = method_id("gptq_w4", calib_seed)
    method_b = method_id("sg_mmp", calib_seed)
    summary, automatic = automatic_audit(model_key, method_a, method_b)
    a_rows = rows_by_id(model_key, method_a)
    b_rows = rows_by_id(model_key, method_b)
    disagreements = [row for row in automatic if row["transition"] not in {"both_correct"}]
    rng = random.Random(20267001)
    rng.shuffle(disagreements)
    selected = sorted(disagreements[: min(sample_size, len(disagreements))], key=lambda x: x["doc_id"])

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"{model_key}__c{calib_seed}"
    annotation_path = OUT / f"{stem}__blinded_annotation.csv"
    key_path = OUT / f"{stem}__blinding_key.json"
    key = {}
    fields = [
        "annotation_id",
        "doc_id",
        "question",
        "gold",
        "output_a",
        "output_b",
        "rater1_label",
        "rater2_label",
        "consensus_label",
        "notes",
    ]
    with annotation_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for annotation_id, item in enumerate(selected):
            doc_id = item["doc_id"]
            first, second = a_rows[doc_id], b_rows[doc_id]
            if rng.random() < 0.5:
                first, second = second, first
                order = [method_b, method_a]
            else:
                order = [method_a, method_b]
            key[str(annotation_id)] = {"doc_id": doc_id, "output_order": order}
            writer.writerow(
                {
                    "annotation_id": annotation_id,
                    "doc_id": doc_id,
                    "question": first["question"],
                    "gold": first["gold"],
                    "output_a": first["generation"],
                    "output_b": second["generation"],
                    "rater1_label": "",
                    "rater2_label": "",
                    "consensus_label": "",
                    "notes": "",
                }
            )
    key_path.write_text(json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
    summary.update(
        {
            "annotation_sample_size": len(selected),
            "annotation_sampling": "fixed-seed sample from all non-both-correct cases",
            "allowed_labels": sorted(LABELS),
            "annotation_file": str(annotation_path),
            "blinding_key": str(key_path),
        }
    )
    (OUT / f"{stem}__automatic_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def cohen_kappa(first: list[str], second: list[str]) -> float | None:
    if not first or len(first) != len(second):
        return None
    agreement = sum(a == b for a, b in zip(first, second)) / len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        first_counts[label] * second_counts[label] for label in LABELS
    ) / (len(first) ** 2)
    return (agreement - expected) / (1 - expected) if expected < 1 else 1.0


def summarize_annotations(path: Path) -> None:
    with path.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    labeled = [row for row in rows if row["consensus_label"].strip()]
    invalid = sorted(
        {row["consensus_label"].strip() for row in labeled}
        - LABELS
    )
    if invalid:
        raise ValueError(f"Unknown consensus labels: {invalid}; allowed={sorted(LABELS)}")
    invalid_ratings = sorted(
        {
            row[field].strip()
            for row in rows
            for field in ("rater1_label", "rater2_label")
            if row[field].strip()
        }
        - LABELS
    )
    if invalid_ratings:
        raise ValueError(
            f"Unknown rater labels: {invalid_ratings}; allowed={sorted(LABELS)}"
        )
    paired_ratings = [
        row
        for row in rows
        if row["rater1_label"].strip() and row["rater2_label"].strip()
    ]
    summary = {
        "annotation_file": str(path),
        "rows": len(rows),
        "consensus_labeled": len(labeled),
        "consensus_counts": dict(
            Counter(row["consensus_label"].strip() for row in labeled)
        ),
        "double_coded": len(paired_ratings),
        "cohen_kappa": cohen_kappa(
            [row["rater1_label"].strip() for row in paired_ratings],
            [row["rater2_label"].strip() for row in paired_ratings],
        ),
    }
    output = path.with_name(path.stem + "__summary.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--model", choices=MODEL_SPECS, required=True)
    prepare.add_argument("--calib-seed", type=int, default=RANDOM_CALIB_SEED)
    prepare.add_argument("--sample-size", type=int, default=200)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--annotations", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_annotation(args.model, args.calib_seed, args.sample_size)
    else:
        summarize_annotations(args.annotations)


if __name__ == "__main__":
    main()

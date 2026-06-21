"""Export redacted GSM8K per-example outcomes for public release.

The private evaluator logs contain benchmark prompts, reference answers, and
model generations. This exporter intentionally retains only identifiers and
derived evaluation outcomes, so the public artifact does not redistribute the
GSM8K test content or chain-of-thought generations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


RAW_FIELDS = {"question", "answer", "gold", "generation"}


def parse_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__gsm8k500" not in stem or "__" not in stem:
        raise ValueError(f"Not a GSM8K-500 sample file: {path.name}")
    model, remainder = stem.split("__", 1)
    method = remainder.removesuffix("__gsm8k500")
    return model, method


def export(input_dir: Path, output_dir: Path) -> None:
    files = sorted(input_dir.glob("*__gsm8k500.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *__gsm8k500.jsonl files in {input_dir}")

    rows: list[dict[str, str | int]] = []
    summary: dict[str, dict[str, int | float]] = {}
    for path in files:
        model, method = parse_filename(path)
        group = f"{model}/{method}"
        count = correct = 0
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                leaked = RAW_FIELDS.intersection(record)
                if leaked:
                    # Raw fields are expected in the private input. They are
                    # deliberately never copied to the public output.
                    record = {key: value for key, value in record.items() if key not in leaked}
                if not {"doc_id", "prediction", "correct"}.issubset(record):
                    raise ValueError(f"Missing public fields in {path}:{line_no}")
                rows.append(
                    {
                        "model_key": model,
                        "method": method,
                        "doc_id": int(record["doc_id"]),
                        "prediction": "" if record["prediction"] is None else str(record["prediction"]),
                        "correct": int(record["correct"]),
                    }
                )
                count += 1
                correct += int(record["correct"])
        if count != 500:
            raise ValueError(f"Expected 500 examples in {path.name}, found {count}")
        summary[group] = {"n": count, "correct": correct, "accuracy": round(100 * correct / count, 2)}

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "per_example_correctness.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model_key", "method", "doc_id", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (str(row["model_key"]), str(row["method"]), int(row["doc_id"]))))

    manifest = {
        "description": "Redacted derived GSM8K-500 outcomes; no prompts, reference answers, or generated reasoning are included.",
        "groups": summary,
        "row_count": len(rows),
        "group_count": len(summary),
        "fields": ["model_key", "method", "doc_id", "prediction", "correct"],
        "raw_fields_excluded": sorted(RAW_FIELDS),
    }
    (output_dir / "per_example_correctness_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(rows)} rows from {len(files)} groups to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()

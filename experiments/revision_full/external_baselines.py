"""Register externally produced baselines under the canonical full-test schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.revision_full.protocol import (
    GSM8K_TEST_SIZE,
    MODEL_SPECS,
    OUT,
    PROTOCOL_VERSION,
    RESULTS_DIR,
)


REGISTRY = OUT / "external_baselines"
ALLOWED_METHODS = {"tacq", "hawq_v2"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_samples(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [int(row["doc_id"]) for row in rows]
    if len(rows) != GSM8K_TEST_SIZE or set(ids) != set(range(GSM8K_TEST_SIZE)):
        raise RuntimeError(
            f"External samples must contain each GSM8K id 0-{GSM8K_TEST_SIZE - 1} exactly once"
        )
    missing = [
        row["doc_id"] for row in rows if "correct" not in row or "generation" not in row
    ]
    if missing:
        raise RuntimeError(f"External samples lack correctness/generation fields: {missing[:5]}")
    invalid = [
        row["doc_id"]
        for row in rows
        if int(row["correct"]) not in (0, 1) or not isinstance(row["generation"], str)
    ]
    if invalid:
        raise RuntimeError(f"External samples contain invalid fields: {invalid[:5]}")
    return rows


def copy_if_needed(source: Path, destination: Path) -> None:
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def register(
    model_key: str,
    method: str,
    average_bits: float,
    samples: Path,
    source_url: str,
    source_commit: str,
    config: Path,
) -> None:
    if model_key not in MODEL_SPECS:
        raise ValueError(f"unknown model: {model_key}")
    if method not in ALLOWED_METHODS:
        raise ValueError(f"method must be one of {sorted(ALLOWED_METHODS)}")
    if not 4.0 <= average_bits <= 8.0:
        raise ValueError("average bits must be in [4, 8]")
    read_samples(samples)
    if not config.exists():
        raise FileNotFoundError(config)
    selection_path = OUT / "selections" / f"{model_key}.json"
    if not selection_path.exists():
        raise FileNotFoundError(f"Run model selection first: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    sg_bits = float(selection["actual_avg_bits"])
    if abs(average_bits - sg_bits) > 0.05:
        raise RuntimeError(
            f"External baseline is not budget matched: {average_bits:.4f} vs SG {sg_bits:.4f}"
        )

    method_id = f"external_{method}"
    destination = (
        RESULTS_DIR
        / "samples"
        / f"{model_key}__{method_id}__gsm8k{GSM8K_TEST_SIZE}.jsonl"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    copy_if_needed(samples, destination)
    config_destination = REGISTRY / f"{model_key}__{method}__config{config.suffix}"
    config_destination.parent.mkdir(parents=True, exist_ok=True)
    copy_if_needed(config, config_destination)
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "model": MODEL_SPECS[model_key]["display_name"],
        "method": method,
        "method_id": method_id,
        "parameter_weighted_average_bits": average_bits,
        "sg_parameter_weighted_average_bits": sg_bits,
        "canonical_evaluator": "direct 5-shot greedy, complete official GSM8K test set",
        "n": GSM8K_TEST_SIZE,
        "samples": str(destination),
        "samples_sha256": sha256(destination),
        "source_url": source_url,
        "source_commit": source_commit,
        "config": str(config_destination),
        "config_sha256": sha256(config_destination),
    }
    output = REGISTRY / f"{model_key}__{method}.json"
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


def validate() -> None:
    records = []
    for path in sorted(REGISTRY.glob("*.json")):
        if "__config" in path.name:
            continue
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"Stale external baseline registration: {path}")
        sample_path = Path(record["samples"])
        read_samples(sample_path)
        if sha256(sample_path) != record["samples_sha256"]:
            raise RuntimeError(f"External sample hash changed: {sample_path}")
        records.append(record)
    print(json.dumps({"registered": len(records), "records": records}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    item = sub.add_parser("register")
    item.add_argument("--model", choices=MODEL_SPECS, required=True)
    item.add_argument("--method", choices=sorted(ALLOWED_METHODS), required=True)
    item.add_argument("--average-bits", type=float, required=True)
    item.add_argument("--samples", type=Path, required=True)
    item.add_argument("--source-url", required=True)
    item.add_argument("--source-commit", required=True)
    item.add_argument("--config", type=Path, required=True)
    sub.add_parser("validate")
    args = parser.parse_args()
    if args.command == "register":
        register(
            args.model,
            args.method,
            args.average_bits,
            args.samples,
            args.source_url,
            args.source_commit,
            args.config,
        )
    else:
        validate()


if __name__ == "__main__":
    main()

"""Register externally produced baselines under the canonical full-test schema."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    ROOT,
)


REGISTRY = OUT / "external_baselines"
ALLOWED_METHODS = {"tacq", "hawq_v2"}
OFFICIAL_SOURCE_URLS = {
    "tacq": "https://github.com/the-inscrutable-x/tacq",
    "hawq_v2": "https://github.com/zhen-dong/hawq",
}


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        raise RuntimeError(f"Registered evidence must be copied inside the repository: {path}")


def resolve_record_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def validate_source(method: str, source_url: str, source_commit: str) -> None:
    if method not in OFFICIAL_SOURCE_URLS:
        raise RuntimeError(f"Unknown external method {method!r}")
    normalized_source = source_url.lower().rstrip("/")
    if normalized_source.endswith(".git"):
        normalized_source = normalized_source[:-4]
    if normalized_source != OFFICIAL_SOURCE_URLS[method]:
        raise RuntimeError(
            f"{method} must use official source {OFFICIAL_SOURCE_URLS[method]}"
        )
    if re.fullmatch(r"[0-9a-fA-F]{40}", source_commit) is None:
        raise RuntimeError("source_commit must be a full 40-character Git commit SHA")


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
    required = {
        "doc_id",
        "question",
        "gold",
        "prediction",
        "correct",
        "generation",
        "generated_token_count",
        "truncated",
    }
    missing = [row.get("doc_id") for row in rows if not required.issubset(row)]
    if missing:
        raise RuntimeError(f"External samples lack correctness/generation fields: {missing[:5]}")
    invalid = [
        row["doc_id"]
        for row in rows
        if int(row["correct"]) not in (0, 1)
        or not isinstance(row["generation"], str)
        or not isinstance(row["question"], str)
        or not isinstance(row["gold"], str)
        or int(row["generated_token_count"]) < 0
        or not isinstance(row["truncated"], bool)
    ]
    if invalid:
        raise RuntimeError(f"External samples contain invalid fields: {invalid[:5]}")
    return rows


def copy_if_needed(source: Path, destination: Path) -> None:
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def validate_external_config(
    path: Path,
    *,
    method: str,
    model_revision: str,
    source_commit: str,
    average_bits: float,
    expected_parameter_count: int | None = None,
) -> dict:
    if path.suffix.lower() != ".json":
        raise RuntimeError("External baseline config must be a machine-readable JSON file")
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "method",
        "source_commit",
        "model_revision",
        "test_data_used_for_selection",
        "calibration_and_selection_data",
        "command",
        "environment_lock",
        "adaptations_from_official_source",
        "budget_search",
        "bit_width_parameter_counts",
        "bit_accounting_scope",
        "parameter_weighted_average_bits",
        "canonical_evaluator",
    }
    missing = sorted(required - set(config))
    if missing:
        raise RuntimeError(f"External config lacks required fields: {missing}")
    if (
        config["method"] != method
        or config["source_commit"] != source_commit
        or config["model_revision"] != model_revision
        or config["test_data_used_for_selection"] is not False
    ):
        raise RuntimeError("External config method/source/model/test-separation mismatch")
    evaluator = config["canonical_evaluator"]
    if (
        evaluator.get("dataset") != "openai/gsm8k/main:test"
        or int(evaluator.get("n", -1)) != GSM8K_TEST_SIZE
        or evaluator.get("decoding") != "greedy"
        or int(evaluator.get("max_new_tokens", -1)) != 256
    ):
        raise RuntimeError("External config does not use the canonical evaluator")
    counts = {
        int(bits): int(count)
        for bits, count in config["bit_width_parameter_counts"].items()
    }
    if not counts or any(bits <= 0 or count < 0 for bits, count in counts.items()):
        raise RuntimeError("External bit-width parameter counts are invalid")
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("External bit accounting contains no parameters")
    if expected_parameter_count is not None and total != expected_parameter_count:
        raise RuntimeError(
            f"External bit accounting covers {total} parameters; the locked eligible "
            f"module scope contains {expected_parameter_count}"
        )
    computed = sum(bits * count for bits, count in counts.items()) / total
    configured = float(config["parameter_weighted_average_bits"])
    if abs(computed - configured) > 1e-6 or abs(configured - average_bits) > 1e-6:
        raise RuntimeError(
            f"External bit accounting mismatch: computed={computed}, "
            f"config={configured}, CLI={average_bits}"
        )
    return config


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
    validate_source(method, source_url, source_commit)
    read_samples(samples)
    if not config.exists():
        raise FileNotFoundError(config)
    from experiments.revision_full.run import selection_for

    selection = selection_for(model_key)
    sg_bits = float(selection["actual_avg_bits"])
    if abs(average_bits - sg_bits) > 0.05:
        raise RuntimeError(
            f"External baseline is not budget matched: {average_bits:.4f} vs SG {sg_bits:.4f}"
        )
    validate_external_config(
        config,
        method=method,
        model_revision=selection["model_snapshot"]["resolved_revision"],
        source_commit=source_commit,
        average_bits=average_bits,
        expected_parameter_count=sum(
            int(row["n_params"]) for row in selection["module_rows"]
        ),
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
        "samples": portable_path(destination),
        "samples_sha256": sha256(destination),
        "source_url": source_url,
        "source_commit": source_commit,
        "config": portable_path(config_destination),
        "config_sha256": sha256(config_destination),
        "config_schema": "external-baseline-v1",
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
        validate_source(record["method"], record["source_url"], record["source_commit"])
        sample_path = resolve_record_path(record["samples"])
        read_samples(sample_path)
        if sha256(sample_path) != record["samples_sha256"]:
            raise RuntimeError(f"External sample hash changed: {sample_path}")
        config_path = resolve_record_path(record["config"])
        if not config_path.exists() or sha256(config_path) != record["config_sha256"]:
            raise RuntimeError(f"External config changed: {config_path}")
        selection = json.loads(
            (OUT / "selections" / f"{record['model_key']}.json").read_text(
                encoding="utf-8"
            )
        )
        validate_external_config(
            config_path,
            method=record["method"],
            model_revision=selection["model_snapshot"]["resolved_revision"],
            source_commit=record["source_commit"],
            average_bits=float(record["parameter_weighted_average_bits"]),
            expected_parameter_count=sum(
                int(row["n_params"]) for row in selection["module_rows"]
            ),
        )
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

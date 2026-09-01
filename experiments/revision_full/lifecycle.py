"""Fail-closed lifecycle checks for reconstructible quantized state files."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from experiments.revision_full.protocol import (
    CAUSAL_PATCH_N,
    GSM8K_TEST_SIZE,
    MODEL_SPECS,
    OUT,
    PROTOCOL_VERSION,
    RANDOM_ALLOCATIONS,
    RANDOM_CALIB_SEED,
    RESULTS_DIR,
    fixed_causal_patch_indices,
    method_id,
    state_metadata_path,
    state_path,
)


BROAD_TASKS = {
    "arc_challenge",
    "hellaswag",
    "mmlu",
    "mmlu_high_school_mathematics",
}
EXTRA_TASKS = {"svamp", "asdiv", "hendrycks_math500", "truthfulqa_gen"}
CORE_VARIANTS = ("gptq_w4", "gptq_w5", "gptq_w6", "sg_mmp")
CONTROL_VARIANTS = (
    "qkv_only",
    "o_only",
    "ffn_only",
    "qkv_priority_matched",
    "o_priority_matched",
    "ffn_priority_matched",
    "hessian_diag_matched",
)
RECEIPT_DIR = OUT / "lifecycle_receipts"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _has_exact_ids(path: Path, expected_ids: set[int]) -> bool:
    try:
        rows = _read_jsonl(path)
        ids = [int(row["doc_id"]) for row in rows]
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return len(ids) == len(expected_ids) and set(ids) == expected_ids


def gsm8k_sample_path(model_key: str, variant: str, calib_seed: int | None) -> Path:
    method = method_id(variant, calib_seed)
    return (
        RESULTS_DIR
        / "samples"
        / f"{model_key}__{method}__gsm8k{GSM8K_TEST_SIZE}.jsonl"
    )


def gsm8k_complete(model_key: str, variant: str, calib_seed: int | None) -> bool:
    return _has_exact_ids(
        gsm8k_sample_path(model_key, variant, calib_seed),
        set(range(GSM8K_TEST_SIZE)),
    )


def format_result_path(model_key: str, variant: str, calib_seed: int | None) -> Path:
    method = method_id(variant, calib_seed)
    return (
        RESULTS_DIR
        / "format_control"
        / f"{model_key}__{method}__gsm8k_mcq{GSM8K_TEST_SIZE}.jsonl"
    )


def format_complete(model_key: str, variant: str, calib_seed: int | None) -> bool:
    return _has_exact_ids(
        format_result_path(model_key, variant, calib_seed),
        set(range(GSM8K_TEST_SIZE)),
    )


def broad_result_path(model_key: str, variant: str, calib_seed: int | None) -> Path:
    method = method_id(variant, calib_seed)
    return RESULTS_DIR / "broad" / f"{model_key}__{method}.json"


def broad_complete(model_key: str, variant: str, calib_seed: int | None) -> bool:
    path = broad_result_path(model_key, variant, calib_seed)
    try:
        scores = _read_json(path).get("scores", {})
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return set(scores) == BROAD_TASKS and all(
        scores[task] is not None for task in BROAD_TASKS
    )


def extra_result_path(model_key: str, variant: str, calib_seed: int | None) -> Path:
    method = method_id(variant, calib_seed)
    return RESULTS_DIR / "extra" / f"{model_key}__{method}.json"


def extra_complete(model_key: str, variant: str, calib_seed: int | None) -> bool:
    path = extra_result_path(model_key, variant, calib_seed)
    try:
        results = _read_json(path).get("results", {})
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if set(results) != EXTRA_TASKS:
        return False
    for task in EXTRA_TASKS:
        samples = results[task].get("samples", [])
        if not samples or int(results[task].get("n_samples", -1)) != len(samples):
            return False
    return True


def causal_result_paths(model_key: str, calib_seed: int) -> tuple[Path, Path]:
    stem = f"{model_key}__gptq_w4__c{calib_seed}"
    directory = RESULTS_DIR / "causal_patch"
    return directory / f"{stem}.jsonl", directory / f"{stem}__summary.json"


def causal_complete(model_key: str, calib_seed: int) -> bool:
    result, summary = causal_result_paths(model_key, calib_seed)
    expected = set(fixed_causal_patch_indices())
    if len(expected) != CAUSAL_PATCH_N or not _has_exact_ids(result, expected):
        return False
    try:
        rows = _read_jsonl(result)
        layer_sets = []
        for row in rows:
            layers = [int(item["layer"]) for item in row.get("patches", [])]
            if not layers or len(layers) != len(set(layers)):
                return False
            layer_sets.append(set(layers))
        if not layer_sets or any(
            layers != layer_sets[0] for layers in layer_sets[1:]
        ):
            return False
        expected_layers = set(range(max(layer_sets[0]) + 1))
        if layer_sets[0] != expected_layers:
            return False
        summary_record = _read_json(summary)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return int(summary_record.get("n", -1)) == CAUSAL_PATCH_N


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def evidence_paths_for_state(
    model_key: str, calib_seed: int, variant: str
) -> list[Path]:
    sample = gsm8k_sample_path(model_key, variant, calib_seed)
    _require(
        gsm8k_complete(model_key, variant, calib_seed),
        f"canonical GSM8K result is incomplete for {model_key}/{variant}/c{calib_seed}",
    )
    evidence = [sample]
    if calib_seed == RANDOM_CALIB_SEED and variant in {"gptq_w4", "sg_mmp"}:
        _require(
            broad_complete(model_key, variant, calib_seed),
            f"broad panel is incomplete for {model_key}/{variant}/c{calib_seed}",
        )
        _require(
            extra_complete(model_key, variant, calib_seed),
            f"generative transfer panel is incomplete for {model_key}/{variant}/c{calib_seed}",
        )
        _require(
            format_complete(model_key, variant, calib_seed),
            f"format control is incomplete for {model_key}/{variant}/c{calib_seed}",
        )
        evidence.extend(
            [
                broad_result_path(model_key, variant, calib_seed),
                extra_result_path(model_key, variant, calib_seed),
                format_result_path(model_key, variant, calib_seed),
            ]
        )
    if (
        model_key == "qwen05"
        and calib_seed == RANDOM_CALIB_SEED
        and variant == "gptq_w4"
    ):
        _require(
            causal_complete(model_key, calib_seed),
            f"causal patch is incomplete for {model_key}/{variant}/c{calib_seed}",
        )
        evidence.extend(causal_result_paths(model_key, calib_seed))
    return evidence


def state_consumers_complete(model_key: str, calib_seed: int, variant: str) -> bool:
    try:
        evidence_paths_for_state(model_key, calib_seed, variant)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False
    return True


def bank_consumer_variants(model_key: str, calib_seed: int) -> list[str]:
    variants = list(CORE_VARIANTS)
    if calib_seed == RANDOM_CALIB_SEED:
        variants.extend(CONTROL_VARIANTS)
        if MODEL_SPECS[model_key]["role"] == "primary":
            for allocation_id in range(RANDOM_ALLOCATIONS):
                variants.extend(
                    [f"random_{allocation_id}", f"random_modules_{allocation_id}"]
                )
    return variants


def evidence_paths_for_bank(model_key: str, calib_seed: int) -> list[Path]:
    paths: dict[str, Path] = {}
    for variant in bank_consumer_variants(model_key, calib_seed):
        evidence_paths_for_state(model_key, calib_seed, variant)
        receipt = receipt_path(model_key, calib_seed, variant)
        _require(receipt.exists(), f"state cleanup receipt is missing: {receipt}")
        paths[str(receipt)] = receipt
    return list(paths.values())


def bank_consumers_complete(model_key: str, calib_seed: int) -> bool:
    try:
        evidence_paths_for_bank(model_key, calib_seed)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return False
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def receipt_path(model_key: str, calib_seed: int, variant: str) -> Path:
    return RECEIPT_DIR / model_key / f"calib_{calib_seed}" / f"{variant}.json"


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def cleanup_state_artifact(model_key: str, calib_seed: int, variant: str) -> dict:
    if variant == "precision_bank":
        evidence_paths = evidence_paths_for_bank(model_key, calib_seed)
    else:
        evidence_paths = evidence_paths_for_state(model_key, calib_seed, variant)

    metadata = state_metadata_path(model_key, calib_seed, variant)
    _require(metadata.exists(), f"state metadata is missing: {metadata}")
    state = state_path(model_key, calib_seed, variant)
    destination = receipt_path(model_key, calib_seed, variant)
    if not state.exists() and destination.exists():
        receipt = _read_json(destination)
        print(
            "[lifecycle] "
            + json.dumps(
                {
                    "action": "already_clean",
                    "model_key": model_key,
                    "calibration_seed": calib_seed,
                    "variant": variant,
                    "receipt": str(destination),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return receipt
    deleted_bytes = state.stat().st_size if state.exists() else 0
    evidence_records = [_artifact_record(path) for path in evidence_paths]
    metadata_record = _artifact_record(metadata)

    state.unlink(missing_ok=True)
    state.with_suffix(state.suffix + ".tmp").unlink(missing_ok=True)
    receipt = {
        "protocol_version": PROTOCOL_VERSION,
        "action": "reconstructible_state_deleted",
        "deleted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_key": model_key,
        "calibration_seed": calib_seed,
        "variant": variant,
        "state_path": str(state),
        "bytes_deleted": deleted_bytes,
        "state_was_present": deleted_bytes > 0,
        "metadata": metadata_record,
        "evidence": evidence_records,
    }
    _write_json_atomic(destination, receipt)
    print(
        "[lifecycle] "
        + json.dumps(
            {
                "action": receipt["action"],
                "model_key": model_key,
                "calibration_seed": calib_seed,
                "variant": variant,
                "bytes_deleted": deleted_bytes,
                "evidence_files": len(evidence_records),
                "receipt": str(destination),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return receipt

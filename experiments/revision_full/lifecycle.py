"""Fail-closed lifecycle checks for reconstructible quantized state files."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from experiments.revision_full.protocol import (
    CAUSAL_PATCH_N,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_FORMAT_BATCH_SIZE,
    GSM8K_TEST_SIZE,
    MODEL_SPECS,
    MAX_NEW_TOKENS,
    OUT,
    PROTOCOL_VERSION,
    RANDOM_ALLOCATIONS,
    RANDOM_CALIB_SEED,
    RESULTS_DIR,
    ROOT,
    SCREEN_DIR,
    fixed_causal_patch_indices,
    method_id,
    state_metadata_path,
    state_path,
    validate_random_allocation_manifest,
)


BROAD_TASKS = {
    "arc_challenge",
    "hellaswag",
    "mmlu",
    "mmlu_high_school_mathematics",
}
EXTRA_TASKS = {"svamp", "asdiv_gen", "hendrycks_math500", "truthfulqa_gen"}
CAUSAL_INTERVENTIONS = {"block", "attention", "mlp"}
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
    return (
        len(ids) == len(expected_ids)
        and set(ids) == expected_ids
        and all(row.get("protocol_version") == PROTOCOL_VERSION for row in rows)
    )


def _locked_identity(model_key: str) -> tuple[dict, dict]:
    """Return the immutable dataset/model identities frozen for this run."""
    lock = _read_json(OUT / "protocol_lock.json")
    if lock.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("protocol lock is missing or stale")
    dataset = lock.get("dataset_snapshot", {})
    model = lock.get("model_snapshots", {}).get(model_key, {})
    if not dataset.get("manifest_sha256") or not model.get("resolved_revision"):
        raise RuntimeError(f"protocol lock identity is incomplete for {model_key}")
    return dataset, model


def gsm8k_sample_path(model_key: str, variant: str, calib_seed: int | None) -> Path:
    method = method_id(variant, calib_seed)
    return (
        RESULTS_DIR
        / "samples"
        / f"{model_key}__{method}__gsm8k{GSM8K_TEST_SIZE}.jsonl"
    )


def gsm8k_complete(model_key: str, variant: str, calib_seed: int | None) -> bool:
    path = gsm8k_sample_path(model_key, variant, calib_seed)
    if not _has_exact_ids(path, set(range(GSM8K_TEST_SIZE))):
        return False
    try:
        rows = _read_jsonl(path)
        dataset, model = _locked_identity(model_key)
        return all(
            int(row.get("eval_batch_size_per_gpu", -1))
            == DEFAULT_EVAL_BATCH_SIZE
            and int(row.get("max_new_tokens", -1)) == MAX_NEW_TOKENS
            and row.get("dataset_manifest_sha256")
            == dataset["manifest_sha256"]
            and row.get("model_revision") == model["resolved_revision"]
            and row.get("canonical_test_set")
            == "openai/gsm8k/main:test:all-1319"
            for row in rows
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return False


def format_result_path(model_key: str, variant: str, calib_seed: int | None) -> Path:
    method = method_id(variant, calib_seed)
    return (
        RESULTS_DIR
        / "format_control"
        / f"{model_key}__{method}__gsm8k_mcq{GSM8K_TEST_SIZE}.jsonl"
    )


def format_complete(model_key: str, variant: str, calib_seed: int | None) -> bool:
    path = format_result_path(model_key, variant, calib_seed)
    if not _has_exact_ids(path, set(range(GSM8K_TEST_SIZE))):
        return False
    try:
        dataset, model = _locked_identity(model_key)
        return all(
            int(row.get("format_batch_size_per_gpu", -1))
            == DEFAULT_FORMAT_BATCH_SIZE
            and row.get("dataset_manifest_sha256")
            == dataset["manifest_sha256"]
            and row.get("model_revision") == model["resolved_revision"]
            for row in _read_jsonl(path)
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError, json.JSONDecodeError):
        return False


def broad_result_path(model_key: str, variant: str, calib_seed: int | None) -> Path:
    method = method_id(variant, calib_seed)
    return RESULTS_DIR / "broad" / f"{model_key}__{method}.json"


def broad_complete(model_key: str, variant: str, calib_seed: int | None) -> bool:
    path = broad_result_path(model_key, variant, calib_seed)
    try:
        record = _read_json(path)
        dataset, model = _locked_identity(model_key)
        if (
            record.get("protocol_version") != PROTOCOL_VERSION
            or record.get("model_key") != model_key
            or record.get("method") != method_id(variant, calib_seed)
            or record.get("dataset_snapshot") != dataset
            or record.get("model_snapshot") != model
            or int(record.get("batch_size_per_gpu", -1))
            != DEFAULT_EVAL_BATCH_SIZE
            or int(record.get("max_new_tokens", -1)) != MAX_NEW_TOKENS
            or set(record.get("tasks", [])) != BROAD_TASKS
            or len(record.get("tasks", [])) != len(BROAD_TASKS)
        ):
            return False
        scores = record.get("scores", {})
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
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
        record = _read_json(path)
        dataset, model = _locked_identity(model_key)
        if (
            record.get("protocol_version") != PROTOCOL_VERSION
            or record.get("model_key") != model_key
            or record.get("method") != method_id(variant, calib_seed)
            or record.get("dataset_snapshot") != dataset
            or record.get("model_snapshot") != model
            or int(record.get("batch_size_per_gpu", -1))
            != DEFAULT_EVAL_BATCH_SIZE
            or int(record.get("max_new_tokens", -1)) != MAX_NEW_TOKENS
            or set(record.get("tasks", [])) != EXTRA_TASKS
            or len(record.get("tasks", [])) != len(EXTRA_TASKS)
        ):
            return False
        results = record.get("results", {})
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
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
        dataset, model = _locked_identity(model_key)
        pair_sets = []
        for row in rows:
            pairs = [
                (str(item["intervention"]), int(item["layer"]))
                for item in row.get("patches", [])
            ]
            if (
                not pairs
                or len(pairs) != len(set(pairs))
                or row.get("dataset_manifest_sha256")
                != dataset["manifest_sha256"]
                or row.get("model_revision") != model["resolved_revision"]
                or int(row.get("calibration_seed", -1)) != calib_seed
                or int(row.get("w4_correct", -1)) not in (0, 1)
                or int(row.get("final_answer_tokens", 0)) <= 0
            ):
                return False
            pair_sets.append(set(pairs))
        if not pair_sets or any(
            pairs != pair_sets[0] for pairs in pair_sets[1:]
        ):
            return False
        layers = {layer for _, layer in pair_sets[0]}
        expected_layers = set(range(max(layers) + 1))
        expected_pairs = {
            (intervention, layer)
            for intervention in CAUSAL_INTERVENTIONS
            for layer in expected_layers
        }
        if layers != expected_layers or pair_sets[0] != expected_pairs:
            return False
        summary_record = _read_json(summary)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    summary_pairs = {
        (str(row.get("intervention")), int(row.get("layer", -1)))
        for row in summary_record.get("layers", [])
    }
    return (
        int(summary_record.get("n", -1)) == CAUSAL_PATCH_N
        and summary_record.get("protocol_version") == PROTOCOL_VERSION
        and summary_record.get("dataset_manifest_sha256")
        == dataset["manifest_sha256"]
        and summary_record.get("model_revision") == model["resolved_revision"]
        and int(summary_record.get("calibration_seed", -1)) == calib_seed
        and set(summary_record.get("interventions", [])) == CAUSAL_INTERVENTIONS
        and summary_pairs == expected_pairs
    )


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
            counts = {"layer": RANDOM_ALLOCATIONS, "module": RANDOM_ALLOCATIONS}
            try:
                selection = _read_json(OUT / "selections" / f"{model_key}.json")
                counts = validate_random_allocation_manifest(
                    selection.get("random_allocation_manifest", {})
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
            variants.extend(
                f"random_{allocation_id}"
                for allocation_id in range(counts["layer"])
            )
            variants.extend(
                f"random_modules_{allocation_id}"
                for allocation_id in range(counts["module"])
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
    resolved = path.resolve()
    try:
        display_path = str(resolved.relative_to(ROOT))
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
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


def cleanup_screen_state_artifact(
    model_key: str, split_id: int, calib_seed: int
) -> dict:
    """Delete a screen-only GPTQ state after its exact layer screen is durable."""
    variant = f"screen_gptq_w4_split{split_id}"
    metadata = state_metadata_path(model_key, calib_seed, variant)
    _require(metadata.exists(), f"screen state metadata is missing: {metadata}")
    record = _read_json(metadata)
    _require(
        record.get("protocol_version") == PROTOCOL_VERSION,
        f"stale screen state metadata: {metadata}",
    )
    _require(
        int(record.get("split_id", -1)) == split_id
        and int(record.get("calibration_seed", -1)) == calib_seed,
        f"screen metadata does not match requested split/seed: {metadata}",
    )
    expected_layers = {int(value) for value in record.get("screened_layers", [])}
    screen = SCREEN_DIR / model_key / f"split_{split_id}.jsonl"
    rows = _read_jsonl(screen)
    baselines = [row for row in rows if row.get("type") == "baseline"]
    layer_rows = [row for row in rows if row.get("type") == "layer"]
    observed_layers = [int(row["layer"]) for row in layer_rows]
    expected_dataset = record.get("dataset_snapshot", {})
    expected_model = record.get("model_snapshot", {})
    _require(
        bool(expected_dataset.get("manifest_sha256"))
        and bool(expected_model.get("resolved_revision")),
        f"screen state metadata lacks immutable model/data identity: {metadata}",
    )
    expected_common = lambda row: (
        row.get("protocol_version") == PROTOCOL_VERSION
        and row.get("model_key") == model_key
        and int(row.get("split_id", -1)) == split_id
        and int(row.get("calibration_seed", -1)) == calib_seed
        and row.get("dataset_manifest_sha256")
        == expected_dataset.get("manifest_sha256")
        and row.get("model_revision") == expected_model.get("resolved_revision")
        and int(row.get("eval_batch_size_per_gpu", -1))
        == DEFAULT_EVAL_BATCH_SIZE
        and int(row.get("max_new_tokens", -1)) == MAX_NEW_TOKENS
    )
    _require(
        len(baselines) == 1
        and expected_common(baselines[0])
        and baselines[0].get("quantizer") == "GPTQ-W4"
        and int(baselines[0].get("calibration_seed", -1)) == calib_seed,
        f"screen baseline is incomplete or uses the wrong quantizer: {screen}",
    )
    _require(
        len(observed_layers) == len(expected_layers)
        and set(observed_layers) == expected_layers,
        f"screen layer rows are incomplete or duplicated: {screen}",
    )
    _require(
        all(
            expected_common(row)
            and row.get("quantizer") == "GPTQ-W4"
            and int(row.get("calibration_seed", -1)) == calib_seed
            for row in layer_rows
        ),
        f"screen rows do not match the locked GPTQ calibration: {screen}",
    )

    state = state_path(model_key, calib_seed, variant)
    destination = receipt_path(model_key, calib_seed, variant)
    if not state.exists() and destination.exists():
        return _read_json(destination)
    deleted_bytes = state.stat().st_size if state.exists() else 0
    evidence_records = [_artifact_record(screen)]
    metadata_record = _artifact_record(metadata)
    state.unlink(missing_ok=True)
    state.with_suffix(state.suffix + ".tmp").unlink(missing_ok=True)
    receipt = {
        "protocol_version": PROTOCOL_VERSION,
        "action": "reconstructible_screen_state_deleted",
        "deleted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_key": model_key,
        "split_id": split_id,
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
                "split_id": split_id,
                "calibration_seed": calib_seed,
                "bytes_deleted": deleted_bytes,
                "receipt": str(destination),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return receipt

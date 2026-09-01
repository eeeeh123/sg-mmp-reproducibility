"""Fail-closed readiness checks for core execution and resubmission claims."""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.revision_full.protocol import (
    CALIB_HESSIAN_TOKENS,
    CALIB_LENGTH,
    CALIB_SAMPLES,
    CALIB_SEEDS,
    CAUSAL_PATCH_N,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_FORMAT_BATCH_SIZE,
    EXTRA_TASKS as PROTOCOL_EXTRA_TASKS,
    GSM8K_TEST_SIZE,
    MODEL_SPECS,
    MAX_NEW_TOKENS,
    OUT,
    PROTOCOL_VERSION,
    RANDOM_ALLOCATIONS,
    RANDOM_CALIB_SEED,
    RESULTS_DIR,
    ROOT,
    SCREEN_N,
    SCREEN_CALIB_SEEDS,
    SCREEN_DIR,
    SELECTION_BOOTSTRAP_REPLICATES,
    STATE_METADATA_DIR,
    method_id,
    json_sha256,
    state_path,
)
from experiments.revision_full.download_core_datasets import (
    MANIFEST_PATH as DATASET_MANIFEST_PATH,
    snapshot_sha256 as dataset_snapshot_sha256,
)
from experiments.revision_full.download_models import stable_model_record
from experiments.revision_full.external_baselines import (
    read_samples,
    resolve_record_path,
    sha256,
    validate_external_config,
    validate_source,
)
from experiments.revision_full.lifecycle import bank_consumer_variants, receipt_path


POLICY_PATH = Path(__file__).with_name("claim_policy.json")
FORMAT_METHODS = (
    ("fp16", None),
    ("gptq_w4", RANDOM_CALIB_SEED),
    ("sg_mmp", RANDOM_CALIB_SEED),
)
BROAD_TASKS = {
    "arc_challenge",
    "hellaswag",
    "mmlu",
    "mmlu_high_school_mathematics",
}
EXTRA_TASKS = set(PROTOCOL_EXTRA_TASKS)
EXTRA_PRIMARY_METRICS = {
    "svamp": ("exact_match,flexible-extract", "exact_match,none"),
    "asdiv_gen": ("exact_match,flexible-extract", "exact_match,none"),
    "hendrycks_math500": (
        "exact_match,flexible-extract",
        "exact_match,none",
    ),
    "truthfulqa_gen": ("bleu_acc,none", "rouge1_acc,none"),
}


@functools.lru_cache(maxsize=None)
def current_model_snapshot(model_key: str) -> dict:
    manifest = json.loads(
        (OUT / "model_snapshot_manifest.json").read_text(encoding="utf-8")
    )
    return stable_model_record(manifest["models"][model_key])


@functools.lru_cache(maxsize=1)
def current_dataset_snapshot() -> dict:
    manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
    return {
        "manifest": str(DATASET_MANIFEST_PATH.relative_to(OUT.parent)),
        "manifest_sha256": dataset_snapshot_sha256(manifest),
    }


def jsonl_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _resolve_evidence_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def require_cleanup_receipt(
    errors: list[str],
    model_key: str,
    calib_seed: int,
    variant: str,
    expected_action: str = "reconstructible_state_deleted",
) -> None:
    path = receipt_path(model_key, calib_seed, variant)
    if not path.exists():
        errors.append(f"missing cleanup receipt: {model_key}/c{calib_seed}/{variant}")
        return
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (
            record.get("protocol_version") != PROTOCOL_VERSION
            or record.get("model_key") != model_key
            or int(record.get("calibration_seed", -1)) != calib_seed
            or record.get("variant") != variant
            or record.get("action") != expected_action
        ):
            raise RuntimeError("receipt identity/action mismatch")
        if state_path(model_key, calib_seed, variant).exists():
            raise RuntimeError("state still exists after cleanup receipt")
        artifacts = [record["metadata"], *record.get("evidence", [])]
        if len(artifacts) < 2:
            raise RuntimeError("receipt lacks metadata/evidence hashes")
        for artifact in artifacts:
            evidence_path = _resolve_evidence_path(artifact["path"])
            if (
                not evidence_path.exists()
                or evidence_path.stat().st_size != int(artifact["bytes"])
                or sha256(evidence_path) != artifact["sha256"]
            ):
                raise RuntimeError(f"retained evidence changed: {evidence_path}")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        errors.append(
            f"invalid cleanup receipt {model_key}/c{calib_seed}/{variant}: {exc}"
        )


def sample_path(model_key: str, method: str) -> Path:
    return (
        RESULTS_DIR
        / "samples"
        / f"{model_key}__{method}__gsm8k{GSM8K_TEST_SIZE}.jsonl"
    )


def require_complete_sample(errors: list[str], model_key: str, method: str) -> None:
    path = sample_path(model_key, method)
    rows = jsonl_rows(path)
    ids = [int(row["doc_id"]) for row in rows]
    try:
        expected_revision = current_model_snapshot(model_key)["resolved_revision"]
        expected_dataset_hash = current_dataset_snapshot()["manifest_sha256"]
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        errors.append(f"cannot resolve sample provenance for {model_key}: {exc}")
        return
    if (
        len(rows) != GSM8K_TEST_SIZE
        or len(ids) != len(set(ids))
        or set(ids) != set(range(GSM8K_TEST_SIZE))
        or any(int(row.get("correct", -1)) not in (0, 1) for row in rows)
        or any(row.get("protocol_version") != PROTOCOL_VERSION for row in rows)
        or any(
            int(row.get("eval_batch_size_per_gpu", -1))
            != DEFAULT_EVAL_BATCH_SIZE
            or int(row.get("max_new_tokens", -1)) != MAX_NEW_TOKENS
            for row in rows
        )
        or any(
            row.get("model_revision") != expected_revision
            or row.get("dataset_manifest_sha256") != expected_dataset_hash
            for row in rows
        )
    ):
        errors.append(
            f"incomplete/duplicated {model_key}/{method}: "
            f"{len(rows)} rows, {len(set(ids))} unique IDs"
        )


def require_format_control(errors: list[str], model_key: str, method: str) -> None:
    path = (
        RESULTS_DIR
        / "format_control"
        / f"{model_key}__{method}__gsm8k_mcq{GSM8K_TEST_SIZE}.jsonl"
    )
    rows = jsonl_rows(path)
    ids = [int(row["doc_id"]) for row in rows]
    try:
        expected_revision = current_model_snapshot(model_key)["resolved_revision"]
        expected_dataset_hash = current_dataset_snapshot()["manifest_sha256"]
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        errors.append(f"cannot resolve format provenance for {model_key}: {exc}")
        return
    if (
        len(rows) != GSM8K_TEST_SIZE
        or len(ids) != len(set(ids))
        or set(ids) != set(range(GSM8K_TEST_SIZE))
        or any(row.get("protocol_version") != PROTOCOL_VERSION for row in rows)
        or any(
            int(row.get("format_batch_size_per_gpu", -1))
            != DEFAULT_FORMAT_BATCH_SIZE
            for row in rows
        )
        or any(
            row.get("model_revision") != expected_revision
            or row.get("dataset_manifest_sha256") != expected_dataset_hash
            for row in rows
        )
    ):
        errors.append(
            f"incomplete format control {model_key}/{method}: "
            f"{len(rows)} rows, {len(set(ids))} unique IDs"
        )


def require_task_panels(errors: list[str], model_key: str, method: str) -> None:
    expected_model = current_model_snapshot(model_key)
    expected_dataset = current_dataset_snapshot()
    broad_path = RESULTS_DIR / "broad" / f"{model_key}__{method}.json"
    if not broad_path.exists():
        errors.append(f"missing broad panel {model_key}/{method}")
    else:
        broad = json.loads(broad_path.read_text(encoding="utf-8"))
        if broad.get("protocol_version") != PROTOCOL_VERSION:
            errors.append(f"stale broad panel {model_key}/{method}")
        if broad.get("model_snapshot") != expected_model or broad.get(
            "dataset_snapshot"
        ) != expected_dataset:
            errors.append(f"broad panel provenance mismatch {model_key}/{method}")
        scores = broad.get("scores", {})
        if set(scores) != BROAD_TASKS or any(scores[task] is None for task in BROAD_TASKS):
            errors.append(f"incomplete or noncanonical broad panel {model_key}/{method}")
        if "gsm8k" in scores:
            errors.append(f"duplicate GSM8K evaluator in broad panel {model_key}/{method}")

    extra_path = RESULTS_DIR / "extra" / f"{model_key}__{method}.json"
    if not extra_path.exists():
        errors.append(f"missing extra task panel {model_key}/{method}")
        return
    extra = json.loads(extra_path.read_text(encoding="utf-8"))
    if extra.get("protocol_version") != PROTOCOL_VERSION:
        errors.append(f"stale extra task panel {model_key}/{method}")
    if extra.get("model_snapshot") != expected_model or extra.get(
        "dataset_snapshot"
    ) != expected_dataset:
        errors.append(f"extra panel provenance mismatch {model_key}/{method}")
    results = extra.get("results", {})
    if set(results) != EXTRA_TASKS:
        errors.append(f"incomplete extra task panel {model_key}/{method}")
        return
    for task in EXTRA_TASKS:
        item = results[task]
        samples = item.get("samples", [])
        metrics = item.get("metrics", {})
        expected_n = -1
        if DATASET_MANIFEST_PATH.exists():
            manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
            expected_n = int(
                manifest.get("panels", {})
                .get("tasks", {})
                .get(task, {})
                .get("evaluation_docs", -1)
            )
        if (
            not samples
            or int(item.get("n_samples", -1)) != len(samples)
            or (expected_n > 0 and len(samples) != expected_n)
        ):
            errors.append(f"incomplete logged samples {model_key}/{method}/{task}")
        if not any(metric in metrics for metric in EXTRA_PRIMARY_METRICS[task]):
            errors.append(f"missing primary metric {model_key}/{method}/{task}")


def preflight_errors() -> list[str]:
    errors = []
    lock_path = OUT / "protocol_lock.json"
    if not lock_path.exists():
        return [f"missing protocol lock: {lock_path}"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("protocol_version") != PROTOCOL_VERSION:
        errors.append(
            f"protocol lock {lock.get('protocol_version')} != code {PROTOCOL_VERSION}"
        )
    if lock.get("final_test", {}).get("n") != GSM8K_TEST_SIZE:
        errors.append("canonical GSM8K test is not complete")
    try:
        current_models = {
            model_key: current_model_snapshot(model_key) for model_key in MODEL_SPECS
        }
        if lock.get("model_snapshots") != current_models:
            errors.append("protocol lock does not match the frozen model snapshots")
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        errors.append(f"cannot validate frozen model snapshots: {exc}")
    if not DATASET_MANIFEST_PATH.exists():
        errors.append("missing frozen dataset snapshot manifest")
    else:
        manifest = json.loads(DATASET_MANIFEST_PATH.read_text(encoding="utf-8"))
        current_hash = dataset_snapshot_sha256(manifest)
        if manifest.get("snapshot_sha256") != current_hash:
            errors.append("dataset snapshot manifest identity is invalid")
        if lock.get("dataset_snapshot", {}).get("manifest_sha256") != current_hash:
            errors.append("protocol lock does not match the frozen dataset manifest")
    if lock.get("calibration", {}).get("seeds") != list(CALIB_SEEDS):
        errors.append("calibration seed lock does not match code")
    calibration = lock.get("calibration", {})
    if calibration.get("samples") != CALIB_SAMPLES:
        errors.append("calibration sample count does not match code")
    if calibration.get("max_length") != CALIB_LENGTH:
        errors.append("calibration sequence length does not match code")
    if (
        calibration.get("hessian_activation_tokens_per_module")
        != CALIB_HESSIAN_TOKENS
    ):
        errors.append("calibration Hessian token reservoir does not match code")
    screen_splits = lock.get("screen_splits", [])
    if len(screen_splits) != 3 or any(
        split.get("n") != SCREEN_N or len(split.get("indices", [])) != SCREEN_N
        for split in screen_splits
    ):
        errors.append("train-only screen sizes do not match the locked protocol")
    screen_indices = [
        int(index) for split in screen_splits for index in split.get("indices", [])
    ]
    if len(screen_indices) != len(set(screen_indices)):
        errors.append("train-only screen splits are not disjoint")
    if [split.get("calibration_seed") for split in screen_splits] != list(
        SCREEN_CALIB_SEEDS
    ):
        errors.append("screen calibration seeds do not match the locked protocol")
    screen_quantizer = lock.get("screen_quantizer", {})
    if screen_quantizer.get("method") != "GPTQ-W4" or screen_quantizer.get(
        "calibration_seeds"
    ) != list(SCREEN_CALIB_SEEDS):
        errors.append("sensitivity screen is not calibration-repeated GPTQ-W4")
    if lock.get("random_same_budget_allocations_per_model") != RANDOM_ALLOCATIONS:
        errors.append("random-allocation count does not match code")
    if lock.get("uniform_precision_baselines") != [4, 5, 6]:
        errors.append("uniform W4/W5/W6 baseline lock is incomplete")
    if set(lock.get("broad_tasks", [])) != BROAD_TASKS:
        errors.append("broad panel does not include the explicit math-MCQ control")
    if set(lock.get("generative_transfer_tasks", [])) != EXTRA_TASKS:
        errors.append("generative transfer panel does not match code")
    required_external = lock.get("required_external_matched_budget_baselines", {})
    if required_external.get("models") != ["qwen05", "qwen15"] or set(
        required_external.get("methods", [])
    ) != {"tacq", "hawq_v2"}:
        errors.append("required external matched-budget baselines are not locked")
    if lock.get("selection_stability", {}).get(
        "replicates"
    ) != SELECTION_BOOTSTRAP_REPLICATES:
        errors.append("selection-stability bootstrap is not locked")
    if lock.get("canonical_gsm8k_evaluator") != (
        "direct 5-shot greedy, complete official test set"
    ):
        errors.append("canonical GSM8K evaluator is not locked")
    if lock.get("execution") != {
        "eval_batch_size_per_gpu": DEFAULT_EVAL_BATCH_SIZE,
        "format_batch_size_per_gpu": DEFAULT_FORMAT_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "deterministic_greedy": True,
    }:
        errors.append("batch size, generation length, or greedy decoding lock differs")
    manifest_path = RESULTS_DIR / "format_control" / "gsm8k_mcq_manifest.json"
    if not manifest_path.exists():
        errors.append("missing fixed same-item format-control manifest")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ids = {int(item["doc_id"]) for item in manifest.get("items", [])}
        if ids != set(range(GSM8K_TEST_SIZE)):
            errors.append("format-control manifest is incomplete")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("protocol_version") != PROTOCOL_VERSION:
        errors.append("claim policy version does not match protocol")
    for metadata in STATE_METADATA_DIR.glob("**/*.json"):
        record = json.loads(metadata.read_text(encoding="utf-8"))
        if record.get("protocol_version") != PROTOCOL_VERSION:
            errors.append(f"stale state metadata: {metadata}")
        if metadata.name == "precision_bank.json" and record.get(
            "precision_entries"
        ) != [4, 5, 6, 8]:
            errors.append(f"precision bank lacks shared W4/W5/W6/W8 entries: {metadata}")
        model_key = record.get("model_key")
        variant = record.get("variant")
        calib_seed = record.get("calibration_seed")
        if model_key not in MODEL_SPECS or variant is None or calib_seed is None:
            errors.append(f"state metadata identity is incomplete: {metadata}")
            continue
        if (
            record.get("model_snapshot") != current_model_snapshot(model_key)
            or record.get("dataset_snapshot") != current_dataset_snapshot()
        ):
            errors.append(f"state metadata provenance mismatch: {metadata}")
        state = state_path(model_key, int(calib_seed), str(variant))
        if state.exists() and state.stat().st_size != int(record.get("bytes", -1)):
            errors.append(f"state size does not match metadata: {state}")
    return errors


def core_errors() -> list[str]:
    errors = preflight_errors()
    controls = [
        "qkv_only",
        "o_only",
        "ffn_only",
        "qkv_priority_matched",
        "o_priority_matched",
        "ffn_priority_matched",
        "hessian_diag_matched",
    ]
    for model_key, spec in MODEL_SPECS.items():
        selection_path = OUT / "selections" / f"{model_key}.json"
        if not selection_path.exists():
            errors.append(f"missing native train-only selection for {model_key}")
        else:
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            try:
                expected_dataset = current_dataset_snapshot()
                expected_model = current_model_snapshot(model_key)
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                errors.append(f"cannot resolve selection provenance for {model_key}: {exc}")
                expected_dataset = {}
                expected_model = {}
            if selection.get("test_data_used") is not False:
                errors.append(f"selection is not test-clean for {model_key}")
            if (
                selection.get("protocol_version") != PROTOCOL_VERSION
                or selection.get("dataset_snapshot") != expected_dataset
                or selection.get("model_snapshot") != expected_model
            ):
                errors.append(f"selection provenance is stale for {model_key}")
            if selection.get("selection_bootstrap", {}).get(
                "replicates"
            ) != SELECTION_BOOTSTRAP_REPLICATES:
                errors.append(f"missing selection bootstrap stability for {model_key}")
            if selection.get("screen_calibration_seeds") != list(
                SCREEN_CALIB_SEEDS
            ):
                errors.append(f"screen calibration repeats are missing for {model_key}")
            expected_layers = {
                int(row["layer"]) for row in selection.get("module_rows", [])
            }
            for split_id, calib_seed in enumerate(SCREEN_CALIB_SEEDS):
                screen_path = SCREEN_DIR / model_key / f"split_{split_id}.jsonl"
                try:
                    screen_rows = jsonl_rows(screen_path)
                    baseline_rows = [
                        row for row in screen_rows if row.get("type") == "baseline"
                    ]
                    layer_rows = [
                        row for row in screen_rows if row.get("type") == "layer"
                    ]
                    observed = [int(row["layer"]) for row in layer_rows]
                    split = json.loads(
                        (OUT / "protocol_lock.json").read_text(encoding="utf-8")
                    )["screen_splits"][split_id]
                    common_ok = lambda row: (
                        row.get("protocol_version") == PROTOCOL_VERSION
                        and row.get("model_key") == model_key
                        and int(row.get("split_id", -1)) == split_id
                        and int(row.get("split_seed", -1)) == int(split["seed"])
                        and row.get("split_indices_sha256")
                        == split["indices_sha256"]
                        and row.get("dataset_manifest_sha256")
                        == expected_dataset.get("manifest_sha256")
                        and row.get("model_revision")
                        == expected_model.get("resolved_revision")
                        and int(row.get("eval_batch_size_per_gpu", -1))
                        == DEFAULT_EVAL_BATCH_SIZE
                        and int(row.get("max_new_tokens", -1))
                        == MAX_NEW_TOKENS
                    )
                    if (
                        len(baseline_rows) != 1
                        or len(observed) != len(expected_layers)
                        or set(observed) != expected_layers
                        or not common_ok(baseline_rows[0])
                        or any(
                            not common_ok(row)
                            or row.get("quantizer") != "GPTQ-W4"
                            or int(row.get("calibration_seed", -1)) != calib_seed
                            for row in layer_rows
                        )
                    ):
                        errors.append(
                            f"incomplete/noncanonical GPTQ screen {model_key}/split {split_id}"
                        )
                except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    errors.append(f"invalid screen file {model_key}/split {split_id}")
                require_cleanup_receipt(
                    errors,
                    model_key,
                    calib_seed,
                    f"screen_gptq_w4_split{split_id}",
                    expected_action="reconstructible_screen_state_deleted",
                )
            random_manifest = selection.get("random_allocation_manifest", {})
            layer_sets = random_manifest.get("layer_sets", [])
            module_sets = random_manifest.get("module_sets", [])
            if (
                len(layer_sets) != RANDOM_ALLOCATIONS
                or len({tuple(row) for row in layer_sets}) != RANDOM_ALLOCATIONS
                or random_manifest.get("layer_sets_sha256") != json_sha256(layer_sets)
            ):
                errors.append(f"random-layer allocations are not 30 unique locked sets: {model_key}")
            if (
                len(module_sets) != RANDOM_ALLOCATIONS
                or len({tuple(row) for row in module_sets}) != RANDOM_ALLOCATIONS
                or random_manifest.get("module_sets_sha256") != json_sha256(module_sets)
            ):
                errors.append(f"random-module allocations are not 30 unique locked sets: {model_key}")
        require_complete_sample(errors, model_key, "fp16")
        for seed in CALIB_SEEDS:
            for variant in ["gptq_w4", "gptq_w5", "gptq_w6", "sg_mmp"]:
                require_complete_sample(errors, model_key, method_id(variant, seed))
            for variant in bank_consumer_variants(model_key, seed):
                require_cleanup_receipt(errors, model_key, seed, variant)
            require_cleanup_receipt(errors, model_key, seed, "precision_bank")
        for variant in controls:
            require_complete_sample(
                errors, model_key, method_id(variant, RANDOM_CALIB_SEED)
            )
        if spec["role"] == "primary":
            for allocation_id in range(RANDOM_ALLOCATIONS):
                for prefix in ["random", "random_modules"]:
                    require_complete_sample(
                        errors,
                        model_key,
                        method_id(f"{prefix}_{allocation_id}", RANDOM_CALIB_SEED),
                    )
        for variant, seed in FORMAT_METHODS:
            method = method_id(variant, seed)
            require_format_control(errors, model_key, method)
            require_task_panels(errors, model_key, method)
    causal_summary = (
        RESULTS_DIR
        / "causal_patch"
        / f"qwen05__gptq_w4__c{RANDOM_CALIB_SEED}__summary.json"
    )
    if not causal_summary.exists():
        errors.append("missing qwen05 causal patch summary")
    else:
        summary = json.loads(causal_summary.read_text(encoding="utf-8"))
        causal_dataset = current_dataset_snapshot()
        causal_model = current_model_snapshot("qwen05")
        layer_rows = summary.get("layers", [])
        pairs = {
            (row.get("intervention"), int(row.get("layer", -1)))
            for row in layer_rows
        }
        observed_layers = {layer for _, layer in pairs}
        expected_pairs = {
            (intervention, layer)
            for intervention in {"block", "attention", "mlp"}
            for layer in observed_layers
        }
        if (
            summary.get("protocol_version") != PROTOCOL_VERSION
            or summary.get("dataset_manifest_sha256")
            != causal_dataset["manifest_sha256"]
            or summary.get("model_revision")
            != causal_model["resolved_revision"]
            or int(summary.get("calibration_seed", -1)) != RANDOM_CALIB_SEED
            or summary.get("n") != CAUSAL_PATCH_N
            or set(summary.get("interventions", []))
            != {"block", "attention", "mlp"}
            or not observed_layers
            or observed_layers != set(range(max(observed_layers) + 1))
            or pairs != expected_pairs
            or any(
                row.get("final_answer_nll_reduction_sign_flip_p_holm") is None
                for row in layer_rows
            )
        ):
            errors.append("causal patch diagnostic is incomplete")
    analysis_path = OUT / "analysis_full.json"
    analysis_md = OUT / "analysis_full.md"
    if not analysis_path.exists() or not analysis_md.exists():
        errors.append("missing complete core analysis artifacts")
    else:
        try:
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            expected_revisions = {
                key: current_model_snapshot(key)["resolved_revision"]
                for key in MODEL_SPECS
            }
            expected_execution = {
                "eval_batch_size_per_gpu": DEFAULT_EVAL_BATCH_SIZE,
                "format_batch_size_per_gpu": DEFAULT_FORMAT_BATCH_SIZE,
                "max_new_tokens": MAX_NEW_TOKENS,
            }
            if (
                analysis.get("protocol_version") != PROTOCOL_VERSION
                or analysis.get("dataset_manifest_sha256")
                != current_dataset_snapshot()["manifest_sha256"]
                or analysis.get("model_revisions") != expected_revisions
                or analysis.get("execution") != expected_execution
                or int(analysis.get("n", -1)) != GSM8K_TEST_SIZE
                or len(analysis.get("comparisons", []))
                != len(MODEL_SPECS) * len(CALIB_SEEDS)
                or len(analysis.get("run_level_summary", [])) != len(MODEL_SPECS)
                or len(analysis.get("same_item_format_controls", []))
                != len(MODEL_SPECS)
                or len(analysis.get("random_same_budget_controls", [])) != 6
                or len(analysis.get("module_placement_controls", []))
                != 7 * len(MODEL_SPECS)
                or len(analysis.get("cross_task_format_controls", []))
                != len(MODEL_SPECS) * (len(BROAD_TASKS) + len(EXTRA_TASKS))
            ):
                errors.append("core analysis artifact is stale or incomplete")
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid core analysis artifact: {exc}")
    return errors


def resubmission_errors() -> list[str]:
    errors = core_errors()
    for model_key in ["qwen05", "qwen15"]:
        for external_method in ["tacq", "hawq_v2"]:
            path = (
                OUT
                / "external_baselines"
                / f"{model_key}__{external_method}.json"
            )
            if not path.exists():
                errors.append(
                    f"missing official {external_method} baseline registration for {model_key}"
                )
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            try:
                sample_file = resolve_record_path(record["samples"])
                config_file = resolve_record_path(record["config"])
                read_samples(sample_file)
                if sha256(sample_file) != record["samples_sha256"]:
                    errors.append(
                        f"{external_method} sample hash mismatch for {model_key}"
                    )
                if not config_file.exists() or sha256(config_file) != record[
                    "config_sha256"
                ]:
                    errors.append(
                        f"{external_method} config missing or changed for {model_key}"
                    )
                if record.get("protocol_version") != PROTOCOL_VERSION:
                    errors.append(
                        f"stale {external_method} registration for {model_key}"
                    )
                if not record.get("source_url") or not record.get("source_commit"):
                    errors.append(
                        f"{external_method} provenance is incomplete for {model_key}"
                    )
                validate_source(
                    external_method, record["source_url"], record["source_commit"]
                )
                validate_external_config(
                    config_file,
                    method=external_method,
                    model_revision=current_model_snapshot(model_key)[
                        "resolved_revision"
                    ],
                    source_commit=record["source_commit"],
                    average_bits=float(record["parameter_weighted_average_bits"]),
                    expected_parameter_count=sum(
                        int(row["n_params"])
                        for row in json.loads(
                            (OUT / "selections" / f"{model_key}.json").read_text(
                                encoding="utf-8"
                            )
                        )["module_rows"]
                    ),
                )
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                errors.append(
                    f"invalid {external_method} registration for {model_key}: {exc}"
                )
    for model_key, spec in MODEL_SPECS.items():
        if spec["role"] != "primary":
            continue
        path = (
            RESULTS_DIR
            / "error_analysis"
            / f"{model_key}__c{RANDOM_CALIB_SEED}__blinded_annotation__summary.json"
        )
        if not path.exists():
            errors.append(f"missing completed blinded error annotation summary for {model_key}")
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        if int(summary.get("consensus_labeled_cases", 0)) < 200:
            errors.append(f"fewer than 200 consensus error labels for {model_key}")
        if int(summary.get("double_coded_cases", 0)) < 40:
            errors.append(f"fewer than 40 double-coded errors for {model_key}")
        if int(summary.get("required_double_coded_cases", 0)) != 40:
            errors.append(f"preregistered double-code set is not exactly 40 for {model_key}")
        if not summary.get("per_method_consensus_counts"):
            errors.append(f"error labels were not unblinded by method for {model_key}")
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    deployment = RESULTS_DIR / "deployment_metrics.json"
    if (
        policy.get("deployment_efficiency_claim")
        != "disabled_until_packed_kernel_measurements_exist"
        and not deployment.exists()
    ):
        errors.append("deployment claim enabled without packed-kernel measurements")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["preflight", "core", "resubmission"], required=True)
    args = parser.parse_args()
    errors = {
        "preflight": preflight_errors,
        "core": core_errors,
        "resubmission": resubmission_errors,
    }[args.stage]()
    report = {"stage": args.stage, "ready": not errors, "errors": errors}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Fail-closed readiness checks for core execution and resubmission claims."""

from __future__ import annotations

import argparse
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
    GSM8K_TEST_SIZE,
    MODEL_SPECS,
    OUT,
    PROTOCOL_VERSION,
    RANDOM_ALLOCATIONS,
    RANDOM_CALIB_SEED,
    RESULTS_DIR,
    SCREEN_N,
    SELECTION_BOOTSTRAP_REPLICATES,
    STATE_METADATA_DIR,
    method_id,
)
from experiments.revision_full.external_baselines import read_samples, sha256


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
EXTRA_TASKS = {"svamp", "asdiv", "hendrycks_math500", "truthfulqa_gen"}


def jsonl_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as stream:
        return {int(json.loads(line)["doc_id"]) for line in stream if line.strip()}


def sample_path(model_key: str, method: str) -> Path:
    return (
        RESULTS_DIR
        / "samples"
        / f"{model_key}__{method}__gsm8k{GSM8K_TEST_SIZE}.jsonl"
    )


def require_complete_sample(errors: list[str], model_key: str, method: str) -> None:
    path = sample_path(model_key, method)
    ids = jsonl_ids(path)
    if ids != set(range(GSM8K_TEST_SIZE)):
        errors.append(f"incomplete {model_key}/{method}: {len(ids)}/{GSM8K_TEST_SIZE}")


def require_format_control(errors: list[str], model_key: str, method: str) -> None:
    path = (
        RESULTS_DIR
        / "format_control"
        / f"{model_key}__{method}__gsm8k_mcq{GSM8K_TEST_SIZE}.jsonl"
    )
    ids = jsonl_ids(path)
    if ids != set(range(GSM8K_TEST_SIZE)):
        errors.append(
            f"incomplete format control {model_key}/{method}: "
            f"{len(ids)}/{GSM8K_TEST_SIZE}"
        )


def require_task_panels(errors: list[str], model_key: str, method: str) -> None:
    broad_path = RESULTS_DIR / "broad" / f"{model_key}__{method}.json"
    if not broad_path.exists():
        errors.append(f"missing broad panel {model_key}/{method}")
    else:
        broad = json.loads(broad_path.read_text(encoding="utf-8"))
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
    results = extra.get("results", {})
    if set(results) != EXTRA_TASKS:
        errors.append(f"incomplete extra task panel {model_key}/{method}")
        return
    for task in EXTRA_TASKS:
        item = results[task]
        samples = item.get("samples", [])
        if not samples or int(item.get("n_samples", -1)) != len(samples):
            errors.append(f"incomplete logged samples {model_key}/{method}/{task}")


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
            if selection.get("test_data_used") is not False:
                errors.append(f"selection is not test-clean for {model_key}")
            if selection.get("selection_bootstrap", {}).get(
                "replicates"
            ) != SELECTION_BOOTSTRAP_REPLICATES:
                errors.append(f"missing selection bootstrap stability for {model_key}")
        require_complete_sample(errors, model_key, "fp16")
        for seed in CALIB_SEEDS:
            for variant in ["gptq_w4", "gptq_w5", "gptq_w6", "sg_mmp"]:
                require_complete_sample(errors, model_key, method_id(variant, seed))
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
        if summary.get("n") != CAUSAL_PATCH_N:
            errors.append("causal patch diagnostic is incomplete")
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
                sample_file = Path(record["samples"])
                config_file = Path(record["config"])
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
        if int(summary.get("consensus_labeled", 0)) < 200:
            errors.append(f"fewer than 200 consensus error labels for {model_key}")
        if int(summary.get("double_coded", 0)) < 40:
            errors.append(f"fewer than 40 double-coded errors for {model_key}")
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

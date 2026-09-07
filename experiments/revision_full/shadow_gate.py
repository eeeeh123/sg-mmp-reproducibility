"""Pre-test shadow protocol for the generated-question early-stop rule.

This gate replays 50 archived GSM8K items per model and method (200 formal
shadow generations total).  It must pass before any TaCQ test evaluation.  It
does not replace or rewrite the immutable core sample files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.revision_full.protocol import (
    DEFAULT_EVAL_BATCH_SIZE,
    GSM8K_TEST_SIZE,
    MAX_NEW_TOKENS,
    OUT,
    PROTOCOL_VERSION,
    RANDOM_CALIB_SEED,
    RESULTS_DIR,
    method_id,
)
from experiments.revision_full.question_stop import (
    BASE_GENERATION_KWARGS,
    BASE_GENERATION_KWARGS_SHA256,
    STOP_PROTOCOL,
    GeneratedQuestionStopLogitsProcessor,
    canonical_answer_prefix,
    generation_diagnostics,
)


SHADOW_MODELS = ("qwen05", "qwen15")
SHADOW_VARIANTS = ("gptq_w4", "sg_mmp")
SHADOW_N_PER_CELL = 50
SHADOW_DIR = OUT / "gates" / "question_stop_shadow_v1"
MANIFEST_PATH = SHADOW_DIR / "manifest.json"
RECEIPT_PATH = SHADOW_DIR / "PASS.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append one completed batch without exposing a torn trailing row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    payload = existing + "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tracked_worktree_is_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _tacq_has_started() -> bool:
    return (
        (OUT / "tacq" / "frozen_manifest.json").exists()
        or any((RESULTS_DIR / "samples").glob("*__external_tacq__c*__gsm8k1319.jsonl"))
        or any((OUT / "external_baselines").glob("*__tacq__c*.json"))
    )


def source_path(model_key: str, variant: str) -> Path:
    method = method_id(variant, RANDOM_CALIB_SEED)
    return (
        RESULTS_DIR
        / "samples"
        / f"{model_key}__{method}__gsm8k{GSM8K_TEST_SIZE}.jsonl"
    )


def output_path(model_key: str, variant: str) -> Path:
    return SHADOW_DIR / "rows" / f"{model_key}__{variant}.jsonl"


def _complete_rows(path: Path) -> dict[int, dict]:
    rows = read_jsonl(path)
    by_id = {int(row["doc_id"]): row for row in rows}
    if (
        len(rows) != GSM8K_TEST_SIZE
        or len(by_id) != len(rows)
        or set(by_id) != set(range(GSM8K_TEST_SIZE))
    ):
        raise RuntimeError(f"Shadow source is not a complete canonical run: {path}")
    return by_id


def _coverage_signature(w4: dict, sg: dict) -> tuple:
    return (
        bool(w4.get("truncated")),
        bool(sg.get("truncated")),
        int(w4["correct"]),
        int(sg["correct"]),
        canonical_answer_prefix(w4["generation"]).marker_found,
        canonical_answer_prefix(sg["generation"]).marker_found,
    )


def _coverage_select(w4: dict[int, dict], sg: dict[int, dict]) -> list[int]:
    buckets: dict[tuple, list[int]] = {}
    for doc_id in range(GSM8K_TEST_SIZE):
        buckets.setdefault(_coverage_signature(w4[doc_id], sg[doc_id]), []).append(
            doc_id
        )
    selected: list[int] = []
    keys = sorted(buckets, key=repr)
    while len(selected) < SHADOW_N_PER_CELL:
        progressed = False
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop(0))
                progressed = True
                if len(selected) == SHADOW_N_PER_CELL:
                    break
        if not progressed:
            break
    if len(selected) != SHADOW_N_PER_CELL:
        raise RuntimeError("Could not select the locked shadow sample")
    return sorted(selected)


def prepare(force: bool = False) -> dict:
    if MANIFEST_PATH.exists() and not force:
        manifest = require_manifest()
        print("[skip] existing valid Shadow manifest")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return manifest
    if not _tracked_worktree_is_clean():
        raise RuntimeError(
            "Refusing to freeze Shadow with tracked working-tree changes; "
            "the implementation must be retrievable from its Git commit"
        )
    if force and _tacq_has_started():
        raise RuntimeError("Refusing to change Shadow after the TaCQ phase started")
    if force and any((SHADOW_DIR / "rows").glob("*.jsonl")):
        raise RuntimeError("Refusing to change a manifest after shadow rows exist")

    from experiments.revision_full.run import dataset_provenance, model_provenance

    source_records = {}
    selected_ids = {}
    for model_key in SHADOW_MODELS:
        rows = {
            variant: _complete_rows(source_path(model_key, variant))
            for variant in SHADOW_VARIANTS
        }
        selected_ids[model_key] = _coverage_select(
            rows["gptq_w4"], rows["sg_mmp"]
        )
        source_records[model_key] = {
            variant: {
                "path": str(source_path(model_key, variant)),
                "sha256": sha256(source_path(model_key, variant)),
            }
            for variant in SHADOW_VARIANTS
        }
    manifest = {
        "schema": "question-stop-shadow-v1",
        "protocol_version": PROTOCOL_VERSION,
        "implementation_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "implementation_files_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).with_name("protocol.py"),
                Path(__file__).with_name("question_stop.py"),
                Path(__file__).with_name("run.py"),
                Path(__file__).resolve().parents[1]
                / "fix_gsm8k_500"
                / "direct_eval.py",
                Path(__file__).resolve().parents[2]
                / "ptq"
                / "quant"
                / "gptq.py",
                Path(__file__).resolve().parents[2]
                / "ptq"
                / "quant"
                / "mixed_precision.py",
            )
        },
        "purpose": "implementation equivalence only; not an accuracy estimate",
        "models": list(SHADOW_MODELS),
        "variants": list(SHADOW_VARIANTS),
        "calibration_seed": RANDOM_CALIB_SEED,
        "n_per_model_variant": SHADOW_N_PER_CELL,
        "total_formal_generations": SHADOW_N_PER_CELL
        * len(SHADOW_MODELS)
        * len(SHADOW_VARIANTS),
        "selected_doc_ids": selected_ids,
        "selection_rule": "deterministic round-robin over archived truncation/marker/correctness signatures, then doc_id",
        "source_records": source_records,
        "dataset_provenance": dataset_provenance(),
        "model_provenance": {
            key: model_provenance(key) for key in SHADOW_MODELS
        },
        "base_generation_kwargs": BASE_GENERATION_KWARGS,
        "base_generation_kwargs_sha256": BASE_GENERATION_KWARGS_SHA256,
        "online_change": "one generated-only, per-sequence logits processor; force EOS after blank-line Question:",
        "stop_protocol": STOP_PROTOCOL,
        "pass_criteria": {
            "canonical_prefix_matches": 200,
            "prediction_matches": 200,
            "correctness_matches": 200,
            "base_generation_kwargs_unchanged": True,
            "prompt_is_never_scanned": True,
        },
        "test_data_role": "locked post-hoc implementation shadow only; no method selection or tuning",
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def require_manifest() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256")
    actual = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["manifest_sha256"] = claimed
    if claimed != actual:
        raise RuntimeError("Shadow manifest hash mismatch")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != manifest.get("implementation_commit"):
        raise RuntimeError("Repository commit changed after the Shadow freeze")
    root = Path(__file__).resolve().parents[2]
    for relative, expected in manifest.get("implementation_files_sha256", {}).items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"Shadow implementation file changed: {relative}")
    for model_key, variants in manifest["source_records"].items():
        for variant, record in variants.items():
            if sha256(Path(record["path"])) != record["sha256"]:
                raise RuntimeError(f"Shadow source changed: {model_key}/{variant}")
    return manifest


def run_cell(model_key: str, variant: str, force: bool = False) -> None:
    if model_key not in SHADOW_MODELS or variant not in SHADOW_VARIANTS:
        raise ValueError("Shadow gate is locked to qwen05/qwen15 and W4/SG")
    manifest = require_manifest()
    path = output_path(model_key, variant)
    if force:
        if _tacq_has_started():
            raise RuntimeError("Refusing to rerun Shadow after the TaCQ phase started")
        RECEIPT_PATH.unlink(missing_ok=True)
        path.unlink(missing_ok=True)
    existing = read_jsonl(path) if path.exists() else []
    by_id = {int(row["doc_id"]): row for row in existing}
    if len(by_id) != len(existing) or any(
        row.get("manifest_sha256") != manifest["manifest_sha256"]
        for row in existing
    ):
        raise RuntimeError(f"Incompatible shadow resume file: {path}")
    pending = [
        doc_id
        for doc_id in manifest["selected_doc_ids"][model_key]
        if doc_id not in by_id
    ]
    if not pending:
        print(f"[skip] shadow {model_key}/{variant} complete")
        return
    # Any mutation invalidates the aggregate receipt until verify() recomputes
    # all four cells and atomically writes a new PASS.
    RECEIPT_PATH.unlink(missing_ok=True)

    from experiments.revision_full.run import configure_direct_eval

    direct, method = configure_direct_eval(model_key, variant, RANDOM_CALIB_SEED)
    model, tokenizer = direct.load_model(model_key, method)
    train, test = direct.get_dataset()
    prefix = direct.build_fewshot(train, k=5)
    old_rows = _complete_rows(source_path(model_key, variant))
    for start in range(0, len(pending), DEFAULT_EVAL_BATCH_SIZE):
        ids = pending[start : start + DEFAULT_EVAL_BATCH_SIZE]
        prompts = direct.build_model_prompts(
            model_key,
            tokenizer,
            train,
            prefix,
            [test[doc_id]["question"] for doc_id in ids],
        )
        encoded = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=False
        ).to(model.device)
        prompt_width = int(encoded["input_ids"].shape[1])
        outputs = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=[
                GeneratedQuestionStopLogitsProcessor(
                    tokenizer, prompt_width, tokenizer.eos_token_id
                )
            ],
        )
        generated_ids = outputs[:, prompt_width:]
        decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        rows = []
        for doc_id, raw_text, token_row in zip(ids, decoded, generated_ids):
            old = old_rows[doc_id]
            old_prefix = canonical_answer_prefix(old["generation"])
            diagnostics = generation_diagnostics(
                raw_text,
                token_row.tolist(),
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            prediction = direct.extract_prediction(diagnostics["generation"])
            gold = direct.gold_answer(test[doc_id]["answer"])
            rows.append(
                {
                    "manifest_sha256": manifest["manifest_sha256"],
                    "model_key": model_key,
                    "variant": variant,
                    "doc_id": doc_id,
                    "old_canonical_generation": old_prefix.text,
                    "new_canonical_generation": diagnostics["generation"],
                    "canonical_prefix_match": old_prefix.text
                    == diagnostics["generation"],
                    "old_prediction": old["prediction"],
                    "new_prediction": prediction,
                    "prediction_match": old["prediction"] == prediction,
                    "old_correct": int(old["correct"]),
                    "new_correct": direct.is_correct(prediction, gold),
                    "correctness_match": int(old["correct"])
                    == direct.is_correct(prediction, gold),
                    "old_marker_found": old_prefix.marker_found,
                    **diagnostics,
                }
            )
        append_jsonl(path, rows)
        print(f"[shadow] {model_key}/{variant}: {start + len(rows)}/{len(pending)}")
    del model, tokenizer


def verify() -> dict:
    manifest = require_manifest()
    from experiments.fix_gsm8k_500.direct_eval import (
        extract_prediction,
        gold_answer,
        is_correct,
    )
    from experiments.revision_full.run import get_dataset

    _, test = get_dataset()

    rows = []
    errors = []
    for model_key in SHADOW_MODELS:
        for variant in SHADOW_VARIANTS:
            path = output_path(model_key, variant)
            cell = read_jsonl(path) if path.exists() else []
            expected = set(manifest["selected_doc_ids"][model_key])
            observed = {int(row["doc_id"]) for row in cell}
            if len(cell) != SHADOW_N_PER_CELL or observed != expected:
                errors.append(f"incomplete shadow cell {model_key}/{variant}")
            source = _complete_rows(source_path(model_key, variant))
            for row in cell:
                doc_id = int(row["doc_id"])
                old = source[doc_id]
                old_prefix = canonical_answer_prefix(old["generation"]).text
                source_prediction = extract_prediction(old["generation"])
                source_gold = gold_answer(test[doc_id]["answer"])
                if (
                    source_prediction != old.get("prediction")
                    or is_correct(source_prediction, source_gold)
                    != int(old.get("correct", -1))
                    or old.get("question") != test[doc_id]["question"]
                    or old.get("gold") != source_gold
                ):
                    errors.append(
                        f"shadow source row is internally inconsistent "
                        f"{model_key}/{variant}/{doc_id}"
                    )
                new_prefix = row.get("new_canonical_generation")
                new_prediction = extract_prediction(str(new_prefix))
                recomputed = {
                    "manifest_sha256": manifest["manifest_sha256"],
                    "model_key": model_key,
                    "variant": variant,
                    "old_canonical_generation": old_prefix,
                    "canonical_prefix_match": old_prefix == new_prefix,
                    "old_prediction": old["prediction"],
                    "new_prediction": new_prediction,
                    "prediction_match": old["prediction"] == new_prediction,
                    "old_correct": int(old["correct"]),
                    "new_correct": is_correct(new_prediction, old["gold"]),
                    "correctness_match": int(old["correct"])
                    == is_correct(new_prediction, old["gold"]),
                    "stop_protocol": STOP_PROTOCOL,
                    "base_generation_kwargs_sha256": BASE_GENERATION_KWARGS_SHA256,
                }
                mismatched = [
                    key for key, value in recomputed.items() if row.get(key) != value
                ]
                if mismatched:
                    errors.append(
                        f"shadow row evidence mismatch {model_key}/{variant}/{doc_id}: {mismatched}"
                    )
                if (
                    row.get("generation") != new_prefix
                    or canonical_answer_prefix(str(row.get("raw_generation", ""))).text
                    != new_prefix
                ):
                    errors.append(
                        f"shadow raw/canonical generation mismatch {model_key}/{variant}/{doc_id}"
                    )
            rows.extend(cell)
    checks = {
        "rows": len(rows),
        "canonical_prefix_matches": sum(
            bool(row.get("canonical_prefix_match")) for row in rows
        ),
        "prediction_matches": sum(bool(row.get("prediction_match")) for row in rows),
        "correctness_matches": sum(
            bool(row.get("correctness_match")) for row in rows
        ),
        "base_generation_kwargs_unchanged": manifest[
            "base_generation_kwargs_sha256"
        ]
        == BASE_GENERATION_KWARGS_SHA256,
        "prompt_is_never_scanned": True,
    }
    for name in [
        "canonical_prefix_matches",
        "prediction_matches",
        "correctness_matches",
    ]:
        if checks[name] != manifest["total_formal_generations"]:
            errors.append(f"{name}={checks[name]}/200")
    if checks["rows"] != manifest["total_formal_generations"]:
        errors.append(f"shadow rows={checks['rows']}/200")
    if not checks["base_generation_kwargs_unchanged"]:
        errors.append("base generation kwargs changed")
    report = {
        "schema": "question-stop-shadow-receipt-v1",
        "pass": not errors,
        "manifest_sha256": manifest["manifest_sha256"],
        "checks": checks,
        "errors": errors,
        "row_file_sha256": {
            f"{model}/{variant}": sha256(output_path(model, variant))
            for model in SHADOW_MODELS
            for variant in SHADOW_VARIANTS
            if output_path(model, variant).exists()
        },
    }
    if errors:
        RECEIPT_PATH.unlink(missing_ok=True)
    else:
        write_json(RECEIPT_PATH, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--force", action="store_true")
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--model", choices=SHADOW_MODELS, required=True)
    run_parser.add_argument("--variant", choices=SHADOW_VARIANTS, required=True)
    run_parser.add_argument("--force", action="store_true")
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.force)
    elif args.command == "run":
        run_cell(args.model, args.variant, args.force)
    else:
        verify()


if __name__ == "__main__":
    main()

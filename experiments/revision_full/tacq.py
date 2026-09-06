"""Frozen, auditable TaCQ shared-backend adaptation for the two Qwen models.

The official TaCQ saliency is preserved:

    abs(sum_i(abs(grad_i)) * (W_clean - W4_dequantized) * W_clean)

Only the quantization backend and causal-LM data adapter are shared with the
locked SG-MMP experiment.  All degrees of freedom are frozen in a manifest
before any TaCQ GSM8K-test output is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from experiments.revision_full.protocol import (
    CALIB_SEEDS,
    DEFAULT_EVAL_BATCH_SIZE,
    ELIGIBLE_SHORT_NAMES,
    GSM8K_TEST_SIZE,
    GROUP_SIZE,
    MAX_NEW_TOKENS,
    MODEL_SPECS,
    OUT,
    PROTOCOL_VERSION,
    RESULTS_DIR,
    state_metadata_path,
    state_path,
)
from experiments.revision_full.question_stop import (
    BASE_GENERATION_KWARGS_SHA256,
    STOP_PROTOCOL,
    GeneratedQuestionStopLogitsProcessor,
)
from experiments.revision_full.shadow_gate import RECEIPT_PATH as SHADOW_RECEIPT


TACQ_MODELS = ("qwen05", "qwen15")
OFFICIAL_SOURCE_URL = "https://github.com/The-Inscrutable-X/TACQ"
OFFICIAL_SOURCE_COMMIT = "cfc4cccfb6b7d6f7d184c9fbc8f9373c3e74569a"
IMPORTANCE_N = 128
IMPORTANCE_SELECTION_SEED = 20260906
IMPORTANCE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = "sum of per-example absolute gradients"
IMPORTANCE_NORMALIZATION = "none"
IMPORTANCE_CHECKPOINT_EXAMPLES = 32
MASK_GRANULARITY = "global eligible-weight element"
TIE_BREAKING = "score descending, then module name ascending and row-major index ascending"
BUDGET_TOLERANCE_BITS = 0.01
TACQ_DIR = OUT / "tacq"
MANIFEST_PATH = TACQ_DIR / "frozen_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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


def save_torch_atomic(value, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def require_shadow_pass() -> dict:
    if not SHADOW_RECEIPT.exists():
        raise RuntimeError("Shadow question-stop gate has not passed")
    receipt = json.loads(SHADOW_RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("pass") is not True or receipt.get("errors"):
        raise RuntimeError("Shadow question-stop receipt is not a PASS")
    return receipt


def _existing_test_outputs() -> list[Path]:
    return sorted((RESULTS_DIR / "samples").glob("*__external_tacq__c*__gsm8k1319.jsonl"))


def freeze_manifest(source_commit: str, force: bool = False) -> dict:
    if len(source_commit) != 40 or any(c not in "0123456789abcdefABCDEF" for c in source_commit):
        raise ValueError("TaCQ source commit must be a full 40-character SHA")
    shadow = require_shadow_pass()
    if (MANIFEST_PATH.exists() and not force) or (force and _existing_test_outputs()):
        if MANIFEST_PATH.exists() and not force:
            return require_manifest()
        raise RuntimeError("Refusing to replace the TaCQ manifest after test output exists")

    from experiments.revision_full.run import (
        dataset_provenance,
        get_dataset,
        model_provenance,
        selection_for,
    )

    train, _ = get_dataset()
    candidate_ids = list(range(5, len(train)))
    random.Random(IMPORTANCE_SELECTION_SEED).shuffle(candidate_ids)
    importance_ids = sorted(candidate_ids[:IMPORTANCE_N])
    training_identity = [
        {
            "doc_id": doc_id,
            "sha256": json_hash(
                {
                    "question": train[doc_id]["question"],
                    "answer": train[doc_id]["answer"],
                }
            ),
        }
        for doc_id in importance_ids
    ]
    implementation_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    repository_root = Path(__file__).resolve().parents[2]
    implementation_files = [
        Path(__file__),
        Path(__file__).with_name("analyze.py"),
        Path(__file__).with_name("external_baselines.py"),
        Path(__file__).with_name("protocol.py"),
        Path(__file__).with_name("question_stop.py"),
        Path(__file__).with_name("run.py"),
        Path(__file__).with_name("shadow_gate.py"),
        repository_root / "experiments" / "fix_gsm8k_500" / "direct_eval.py",
        repository_root / "ptq" / "quant" / "gptq.py",
        repository_root / "ptq" / "quant" / "mixed_precision.py",
    ]
    manifest = {
        "schema": "tacq-shared-backend-freeze-v1",
        "protocol_version": PROTOCOL_VERSION,
        "method_label": "TaCQ shared-backend adaptation",
        "official_source_url": OFFICIAL_SOURCE_URL,
        "official_source_commit": source_commit.lower(),
        "official_source_usage": "semantic/formula conformance target; this is a disclosed local shared-backend adaptation, not an unmodified official-script run",
        "implementation_commit": implementation_commit,
        "implementation_files_sha256": {
            str(path.relative_to(repository_root)): sha256(path)
            for path in implementation_files
        },
        "models": list(TACQ_MODELS),
        "calibration_seeds": list(CALIB_SEEDS),
        "dataset_provenance": dataset_provenance(),
        "model_provenance": {key: model_provenance(key) for key in TACQ_MODELS},
        "sg_selections_sha256": {
            key: json_hash(selection_for(key)) for key in TACQ_MODELS
        },
        "shadow_receipt_sha256": sha256(SHADOW_RECEIPT),
        "test_data_used_for_importance_allocation_or_tuning": False,
        "test_access": "one locked final 1319-item evaluation per model and calibration seed after all gates pass",
        "importance": {
            "dataset": "openai/gsm8k/main:train",
            "sample_count": IMPORTANCE_N,
            "sample_selection_seed": IMPORTANCE_SELECTION_SEED,
            "sample_records": training_identity,
            "batch_size": IMPORTANCE_BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "loss": "full causal next-token cross entropy over five-shot prompt plus worked answer",
            "max_length": 2048,
            "shuffle": False,
            "accumulator_dtype": "float32",
            "normalization": IMPORTANCE_NORMALIZATION,
            "formula": "abs(sum(abs(per_example_gradient)) * (W_clean - W4_dequantized) * W_clean)",
            "checkpoint_examples": IMPORTANCE_CHECKPOINT_EXAMPLES,
            "recomputed_per_calibration_seed": False,
        },
        "allocation": {
            "eligible_modules": sorted(ELIGIBLE_SHORT_NAMES),
            "base_quantizer": "the same GPTQ-W4 group_size=128 state in each calibration-seed precision bank",
            "group_size": GROUP_SIZE,
            "preserved_precision": "FP16",
            "mask_granularity": MASK_GRANULARITY,
            "importance_normalization": IMPORTANCE_NORMALIZATION,
            "tie_breaking": TIE_BREAKING,
            "budget_rounding": "k=floor((SG_average_bits-4)*eligible_parameter_count/12); never exceed SG logical budget",
            "logical_budget_tolerance_bits": BUDGET_TOLERANCE_BITS,
            "engineering_ledger": "W4 payload, scale/zero tensors, bit-packed mask, FP16 exceptions, serialized bytes",
        },
        "fixed_hyperparameters": {
            "importance_train_samples": IMPORTANCE_N,
            "importance_batch_size": IMPORTANCE_BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "importance_normalization": IMPORTANCE_NORMALIZATION,
            "mask_granularity": MASK_GRANULARITY,
            "tie_breaking": TIE_BREAKING,
            "budget_rounding": "floor; non-exceeding",
            "importance_recomputed_per_calibration_seed": False,
            "quantized_bits": 4,
            "preserved_bits": 16,
            "group_size": GROUP_SIZE,
            "hyperparameters_frozen_before_test": True,
        },
        "validity_gates": [
            "official source URL and full commit",
            "formula conformance unit test",
            "exact eligible-module coverage",
            "finite gradients and scores",
            "deterministic mask digest",
            "state save/reload and 32 train-only generations",
            "logical average bits within 0.01 of and not above SG",
            "shadow question-stop PASS receipt",
        ],
        "inference": {
            "per_seed": "paired bootstrap and exact McNemar; diagnostic only",
            "primary": "model-level hierarchical bootstrap over calibration seeds and paired examples",
            "multiplicity": "Holm over exactly two model-level SG-minus-TaCQ hypotheses",
        },
        "generation": {
            "dataset": "openai/gsm8k/main:test",
            "n": GSM8K_TEST_SIZE,
            "prompt": "locked direct 5-shot",
            "decoding": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "online_stop": STOP_PROTOCOL,
            "base_generation_kwargs_sha256": BASE_GENERATION_KWARGS_SHA256,
        },
    }
    manifest["manifest_sha256"] = json_hash(manifest)
    write_json(MANIFEST_PATH, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def require_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise RuntimeError("Run tacq.py freeze before TaCQ construction")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    claimed = manifest.pop("manifest_sha256", None)
    actual = json_hash(manifest)
    manifest["manifest_sha256"] = claimed
    if claimed != actual:
        raise RuntimeError("TaCQ frozen manifest hash mismatch")
    if manifest["official_source_url"] != OFFICIAL_SOURCE_URL:
        raise RuntimeError("TaCQ source URL changed")
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != manifest.get("implementation_commit"):
        raise RuntimeError("Repository commit changed after the TaCQ freeze")
    repository_root = Path(__file__).resolve().parents[2]
    for relative, expected in manifest.get("implementation_files_sha256", {}).items():
        if sha256(repository_root / relative) != expected:
            raise RuntimeError(f"TaCQ implementation file changed after freeze: {relative}")
    require_shadow_pass()
    if manifest.get("shadow_receipt_sha256") != sha256(SHADOW_RECEIPT):
        raise RuntimeError("Shadow PASS receipt changed after the TaCQ freeze")
    return manifest


def _gradient_dir(model_key: str) -> Path:
    return TACQ_DIR / "gradients" / model_key


def _gradient_chunk_path(model_key: str, start: int, stop: int) -> Path:
    return _gradient_dir(model_key) / f"abs_grad_{start:03d}_{stop:03d}.pt"


def _gradient_chunk_metadata(path: Path) -> Path:
    return path.with_suffix(".json")


def _valid_chunk(path: Path, manifest: dict, model_key: str, ids: list[int]) -> bool:
    metadata_path = _gradient_chunk_metadata(path)
    if not path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("manifest_sha256") == manifest["manifest_sha256"]
            and metadata.get("model_key") == model_key
            and metadata.get("train_doc_ids") == ids
            and int(metadata.get("bytes", -1)) == path.stat().st_size
            and metadata.get("sha256") == sha256(path)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def capture_importance(model_key: str) -> None:
    if model_key not in TACQ_MODELS:
        raise ValueError("TaCQ is frozen to qwen05 and qwen15")
    manifest = require_manifest()
    import torch
    import torch.nn as nn

    from experiments.revision_full.run import get_dataset, load_model_tokenizer
    from experiments.fix_gsm8k_500 import direct_eval as direct

    train, _ = get_dataset()
    ids = [row["doc_id"] for row in manifest["importance"]["sample_records"]]
    model, tokenizer = load_model_tokenizer(model_key)
    eligible = {
        name: module.weight
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.split(".")[-1] in ELIGIBLE_SHORT_NAMES
        and "lm_head" not in name
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in eligible.values():
        parameter.requires_grad_(True)
    model.eval()
    prefix = direct.build_fewshot(train, k=5)

    for start in range(0, len(ids), IMPORTANCE_CHECKPOINT_EXAMPLES):
        chunk_ids = ids[start : start + IMPORTANCE_CHECKPOINT_EXAMPLES]
        stop = start + len(chunk_ids)
        path = _gradient_chunk_path(model_key, start, stop)
        if _valid_chunk(path, manifest, model_key, chunk_ids):
            print(f"[skip] valid TaCQ gradient chunk {model_key} {start}:{stop}")
            continue
        accumulator = {
            name: torch.zeros_like(parameter, dtype=torch.float32, device=parameter.device)
            for name, parameter in eligible.items()
        }
        for ordinal, doc_id in enumerate(chunk_ids, start=1):
            prompt = direct.build_model_prompts(
                model_key,
                tokenizer,
                train,
                prefix,
                [train[doc_id]["question"]],
            )[0]
            full_text = prompt + " " + train[doc_id]["answer"]
            encoded = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(model.device)
            model.zero_grad(set_to_none=True)
            output = model(
                **encoded, labels=encoded["input_ids"], use_cache=False
            )
            if not torch.isfinite(output.loss):
                raise RuntimeError(f"Non-finite TaCQ loss at train doc {doc_id}")
            output.loss.backward()
            for name, parameter in eligible.items():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Missing/non-finite TaCQ gradient: {name}")
                accumulator[name].add_(parameter.grad.detach().abs().float())
            print(
                f"[tacq-gradient] {model_key} {start + ordinal}/{len(ids)}",
                flush=True,
            )
            del encoded, output
        model.zero_grad(set_to_none=True)
        cpu_accumulator = {name: value.cpu() for name, value in accumulator.items()}
        save_torch_atomic(cpu_accumulator, path)
        metadata = {
            "manifest_sha256": manifest["manifest_sha256"],
            "model_key": model_key,
            "train_doc_ids": chunk_ids,
            "module_names": sorted(eligible),
            "accumulation": GRADIENT_ACCUMULATION,
            "dtype": "float32",
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        write_json(_gradient_chunk_metadata(path), metadata)
        del accumulator, cpu_accumulator
        torch.cuda.empty_cache()

    records = []
    for start in range(0, len(ids), IMPORTANCE_CHECKPOINT_EXAMPLES):
        chunk_ids = ids[start : start + IMPORTANCE_CHECKPOINT_EXAMPLES]
        stop = start + len(chunk_ids)
        path = _gradient_chunk_path(model_key, start, stop)
        if not _valid_chunk(path, manifest, model_key, chunk_ids):
            raise RuntimeError(f"Incomplete TaCQ gradient capture: {path}")
        records.append(json.loads(_gradient_chunk_metadata(path).read_text(encoding="utf-8")))
    write_json(
        _gradient_dir(model_key) / "COMPLETE.json",
        {
            "manifest_sha256": manifest["manifest_sha256"],
            "model_key": model_key,
            "chunks": records,
            "sample_count": len(ids),
        },
    )
    del model, tokenizer


def official_importance_score(abs_gradient, clean_weight, corrupt_weight):
    """Exact TaCQ weight-product contrastive post-processing formula."""

    return (abs_gradient * (clean_weight - corrupt_weight) * clean_weight).abs()


def _score_name(module_name: str) -> str:
    return hashlib.sha256(module_name.encode()).hexdigest()[:20] + ".pt"


def _score_dir(model_key: str, calib_seed: int) -> Path:
    return TACQ_DIR / "scores" / model_key / f"calib_{calib_seed}"


def _score_metadata_path(score_path: Path) -> Path:
    return score_path.with_suffix(".json")


def _valid_score(
    score_path: Path,
    *,
    manifest_sha256: str,
    model_key: str,
    calib_seed: int,
    module_name: str,
    gradient_chunk_sha256: list[str],
    precision_bank_sha256: str,
) -> bool:
    metadata_path = _score_metadata_path(score_path)
    if not score_path.exists() or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return (
            metadata.get("manifest_sha256") == manifest_sha256
            and metadata.get("model_key") == model_key
            and int(metadata.get("calibration_seed", -1)) == calib_seed
            and metadata.get("module_name") == module_name
            and metadata.get("gradient_chunk_sha256") == gradient_chunk_sha256
            and metadata.get("precision_bank_sha256") == precision_bank_sha256
            and int(metadata.get("bytes", -1)) == score_path.stat().st_size
            and metadata.get("sha256") == sha256(score_path)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _chunk_paths(model_key: str, manifest: dict) -> list[Path]:
    ids = [row["doc_id"] for row in manifest["importance"]["sample_records"]]
    paths = []
    for start in range(0, len(ids), IMPORTANCE_CHECKPOINT_EXAMPLES):
        chunk_ids = ids[start : start + IMPORTANCE_CHECKPOINT_EXAMPLES]
        path = _gradient_chunk_path(model_key, start, start + len(chunk_ids))
        if not _valid_chunk(path, manifest, model_key, chunk_ids):
            raise RuntimeError(f"Missing or changed gradient chunk: {path}")
        paths.append(path)
    return paths


def _float32_threshold(paths: list[Path], k: int) -> float:
    """Find the exact kth-largest nonnegative float32 value in bounded RAM."""

    import torch

    if k <= 0:
        return math.inf
    rank = int(k)
    prefix = 0
    prefix_bits = 0
    for shift in (24, 16, 8, 0):
        counts = np.zeros(256, dtype=np.int64)
        for path in paths:
            tensor = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            values = tensor.detach().contiguous().numpy().reshape(-1).view(np.uint32)
            if prefix_bits:
                keep = (values >> (32 - prefix_bits)) == prefix
                values = values[keep]
            byte = ((values >> shift) & 0xFF).astype(np.int64, copy=False)
            counts += np.bincount(byte, minlength=256)
            del tensor, values, byte
        chosen = None
        for value in range(255, -1, -1):
            count = int(counts[value])
            if rank > count:
                rank -= count
            else:
                chosen = value
                break
        if chosen is None:
            raise RuntimeError("Could not locate the exact TaCQ score threshold")
        prefix = (prefix << 8) | chosen
        prefix_bits += 8
    return np.asarray([prefix], dtype=np.uint32).view(np.float32)[0].item()


def _tensor_bytes(value) -> int:
    return int(value.numel() * value.element_size()) if hasattr(value, "numel") else 0


def fp16_count_for_budget(total_parameters: int, sg_average_bits: float) -> tuple[int, float]:
    if total_parameters <= 0:
        raise ValueError("eligible parameter count must be positive")
    count = math.floor((float(sg_average_bits) - 4.0) * total_parameters / 12.0)
    if not 0 <= count <= total_parameters:
        raise ValueError("SG budget cannot be represented by a W4/FP16 mixture")
    logical_bits = (4 * (total_parameters - count) + 16 * count) / total_parameters
    return count, logical_bits


def build_state(model_key: str, calib_seed: int, force: bool = False) -> Path:
    if model_key not in TACQ_MODELS or calib_seed not in CALIB_SEEDS:
        raise ValueError("TaCQ model/calibration seed is outside the frozen design")
    manifest = require_manifest()
    import torch
    import torch.nn as nn

    from experiments.revision_full.run import (
        dataset_provenance,
        load_model_tokenizer,
        model_provenance,
        require_current_state_metadata,
        selection_for,
    )
    from ptq.quant.gptq import dequantize_gptq

    output = state_path(model_key, calib_seed, "tacq")
    metadata_path = state_metadata_path(model_key, calib_seed, "tacq")
    test_output = (
        RESULTS_DIR
        / "samples"
        / f"{model_key}__external_tacq__c{calib_seed}__gsm8k{GSM8K_TEST_SIZE}.jsonl"
    )
    if force and test_output.exists():
        raise RuntimeError("Refusing --force after the corresponding TaCQ test output exists")
    if output.exists() and not force:
        existing = require_current_state_metadata(
            metadata_path, output, model_key, calib_seed, "tacq"
        )
        if (
            existing.get("manifest_sha256") != manifest["manifest_sha256"]
            or existing.get("state_sha256") != sha256(output)
        ):
            raise RuntimeError("Existing TaCQ state is stale; do not resume it")
        return output
    selection = selection_for(model_key)
    module_rows = sorted(selection["module_rows"], key=lambda row: row["name"])
    expected_names = [row["name"] for row in module_rows]
    gradient_paths = _chunk_paths(model_key, manifest)
    chunk_states = [
        torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        for path in gradient_paths
    ]
    if any(sorted(state) != expected_names for state in chunk_states):
        raise RuntimeError("TaCQ gradient module coverage differs from locked selection")
    bank_path = state_path(model_key, calib_seed, "precision_bank")
    require_current_state_metadata(
        state_metadata_path(model_key, calib_seed, "precision_bank"),
        bank_path,
        model_key,
        calib_seed,
        "precision_bank",
    )
    bank = torch.load(bank_path, map_location="cpu", weights_only=False, mmap=True)
    bank_sha256 = sha256(bank_path)
    if any(name not in bank or "w4" not in bank[name] for name in expected_names):
        raise RuntimeError("Precision bank lacks a locked eligible W4 module")
    model, tokenizer = load_model_tokenizer(model_key, device="cpu")
    modules = dict(model.named_modules())

    score_dir = _score_dir(model_key, calib_seed)
    if force and score_dir.exists():
        shutil.rmtree(score_dir)
    score_dir.mkdir(parents=True, exist_ok=True)
    score_records = []
    gradient_chunk_sha256 = [
        json.loads(_gradient_chunk_metadata(path).read_text(encoding="utf-8"))[
            "sha256"
        ]
        for path in gradient_paths
    ]
    for ordinal, name in enumerate(expected_names, start=1):
        score_path = score_dir / _score_name(name)
        score_valid = _valid_score(
            score_path,
            manifest_sha256=manifest["manifest_sha256"],
            model_key=model_key,
            calib_seed=calib_seed,
            module_name=name,
            gradient_chunk_sha256=gradient_chunk_sha256,
            precision_bank_sha256=bank_sha256,
        )
        if score_path.exists() and not score_valid:
            raise RuntimeError(
                f"Stale/incompatible TaCQ score: {score_path}; use --force only before test"
            )
        if not score_valid:
            gradient = sum(state[name].float() for state in chunk_states)
            clean = modules[name].weight.detach().float().cpu()
            w4 = bank[name]["w4"]
            corrupt = dequantize_gptq(
                w4["w_q"], w4["scale"], w4["zero"], int(w4["group_size"])
            ).float()
            score = official_importance_score(gradient, clean, corrupt).float()
            if not torch.isfinite(score).all():
                raise RuntimeError(f"Non-finite TaCQ score: {name}")
            save_torch_atomic(score, score_path)
            write_json(
                _score_metadata_path(score_path),
                {
                    "manifest_sha256": manifest["manifest_sha256"],
                    "model_key": model_key,
                    "calibration_seed": calib_seed,
                    "module_name": name,
                    "gradient_chunk_sha256": gradient_chunk_sha256,
                    "precision_bank_sha256": bank_sha256,
                    "bytes": score_path.stat().st_size,
                    "sha256": sha256(score_path),
                },
            )
            del gradient, clean, corrupt, score
        score_metadata = json.loads(
            _score_metadata_path(score_path).read_text(encoding="utf-8")
        )
        score_records.append(
            {
                "name": name,
                "path": str(score_path),
                "sha256": score_metadata["sha256"],
            }
        )
        print(f"[tacq-score] {model_key}/c{calib_seed} {ordinal}/{len(expected_names)}")

    total = sum(int(row["n_params"]) for row in module_rows)
    sg_bits = float(selection["actual_avg_bits"])
    k, logical_bits = fp16_count_for_budget(total, sg_bits)
    threshold = _float32_threshold([Path(row["path"]) for row in score_records], k)
    greater_count = 0
    equal_count = 0
    for row in score_records:
        score = torch.load(row["path"], map_location="cpu", weights_only=False, mmap=True)
        greater_count += int((score > threshold).sum().item())
        equal_count += int((score == threshold).sum().item())
    tie_remaining = k - greater_count
    if not 0 <= tie_remaining <= equal_count:
        raise RuntimeError("Invalid deterministic TaCQ threshold/tie accounting")

    state = {}
    selected_total = 0
    mask_digest = hashlib.sha256()
    base_payload_bytes = 0
    exception_value_bytes = 0
    packed_mask_bytes = 0
    for row in score_records:
        name = row["name"]
        score = torch.load(row["path"], map_location="cpu", weights_only=False, mmap=True)
        flat_score = score.reshape(-1)
        mask = flat_score > threshold
        if tie_remaining:
            # A zero threshold can create hundreds of millions of ties.  Walk
            # fixed-size row-major chunks so the deterministic tie rule never
            # materializes one checkpoint-sized int64 index vector.
            for offset in range(0, int(flat_score.numel()), 1_000_000):
                if not tie_remaining:
                    break
                local = flat_score[offset : offset + 1_000_000] == threshold
                local_positions = torch.nonzero(local, as_tuple=False).reshape(-1)
                take = min(tie_remaining, int(local_positions.numel()))
                if take:
                    mask[offset + local_positions[:take]] = True
                    tie_remaining -= take
                del local, local_positions
        clean = modules[name].weight.detach().reshape(-1).cpu()
        values = clean[mask].to(torch.float16).contiguous()
        packed_np = np.packbits(mask.numpy(), bitorder="big")
        packed = torch.from_numpy(packed_np.copy())
        choice = dict(bank[name]["w4"])
        choice.update(
            {
                "method": "tacq_w4_fp16",
                "fp16_mask_packbits": packed,
                "mask_numel": int(mask.numel()),
                "fp16_values": values,
            }
        )
        state[name] = choice
        chosen = int(mask.sum().item())
        selected_total += chosen
        mask_digest.update(name.encode())
        mask_digest.update(packed.numpy().tobytes())
        packed_mask_bytes += _tensor_bytes(packed)
        exception_value_bytes += _tensor_bytes(values)
        base_payload_bytes += sum(
            _tensor_bytes(choice[key]) for key in ["w_q", "scale", "zero"]
        )
        del score, flat_score, mask, clean, values, packed, packed_np
    if tie_remaining or selected_total != k:
        raise RuntimeError("TaCQ deterministic mask did not select exactly k weights")

    if logical_bits > sg_bits + 1e-12 or sg_bits - logical_bits > BUDGET_TOLERANCE_BITS:
        raise RuntimeError(
            f"TaCQ logical budget mismatch: {logical_bits:.8f} vs SG {sg_bits:.8f}"
        )
    save_torch_atomic(state, output)
    state_sha = sha256(output)
    adaptation_freeze = manifest["fixed_hyperparameters"]
    config_path = TACQ_DIR / "configs" / f"{model_key}__c{calib_seed}.json"
    config = {
        "method": "tacq",
        "source_commit": manifest["official_source_commit"],
        "model_revision": model_provenance(model_key)["resolved_revision"],
        "calibration_seed": calib_seed,
        "test_data_used_for_selection": False,
        "calibration_and_selection_data": {
            "dataset": "openai/gsm8k/main:train",
            "sample_count": IMPORTANCE_N,
            "sample_records_sha256": json_hash(manifest["importance"]["sample_records"]),
            "test_split_access": "final canonical evaluation only",
        },
        "command": f"python experiments/revision_full/tacq.py build --model {model_key} --calib-seed {calib_seed}",
        "environment_lock": {"repository_server_env": "server_env.sh", "manifest_sha256": manifest["manifest_sha256"]},
        "adaptations_from_official_source": [
            "Qwen2.5-0.5B/1.5B causal-LM loader",
            "shared locked GPTQ-W4 group-128 backend for controlled allocation comparison",
            "global element mask materialized as bit-packed FP16 exceptions",
            "canonical direct GSM8K evaluator with shadow-validated generated-question stop",
        ],
        "adaptation_freeze": adaptation_freeze,
        "budget_search": {
            "candidate_settings_tried": [k],
            "selection_rule": "closed-form largest integer k not exceeding frozen SG logical budget",
            "uses_test_accuracy": False,
        },
        "bit_width_parameter_counts": {"4": total - k, "16": k},
        "bit_accounting_scope": "exactly the eligible q/k/v/o/gate/up/down projection weights listed in the locked model selection",
        "parameter_weighted_average_bits": logical_bits,
        "engineering_ledger": {
            "base_w4_scale_zero_tensor_bytes": base_payload_bytes,
            "bitpacked_mask_bytes": packed_mask_bytes,
            "fp16_exception_value_bytes": exception_value_bytes,
            "serialized_state_bytes": output.stat().st_size,
            "deployment_speed_or_memory_claim_enabled": False,
        },
        "canonical_evaluator": {
            "dataset": "openai/gsm8k/main:test",
            "n": GSM8K_TEST_SIZE,
            "prompt": "direct 5-shot",
            "decoding": "greedy",
            "max_new_tokens": MAX_NEW_TOKENS,
            "online_stop": STOP_PROTOCOL,
        },
    }
    write_json(config_path, config)
    metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "variant": "tacq",
        "calibration_seed": calib_seed,
        "method_label": "TaCQ shared-backend adaptation",
        "manifest_sha256": manifest["manifest_sha256"],
        "source_precision_bank": str(bank_path),
        "source_precision_bank_sha256": bank_sha256,
        "gradient_chunk_sha256": gradient_chunk_sha256,
        "module_names": expected_names,
        "selected_fp16_parameters": k,
        "eligible_parameters": total,
        "parameter_weighted_average_bits": logical_bits,
        "sg_parameter_weighted_average_bits": sg_bits,
        "budget_gap_bits": sg_bits - logical_bits,
        "score_threshold": threshold,
        "tie_count_at_threshold": equal_count,
        "mask_sha256": mask_digest.hexdigest(),
        "state_sha256": state_sha,
        "config": str(config_path),
        "bytes": output.stat().st_size,
        "model_snapshot": model_provenance(model_key),
        "dataset_snapshot": dataset_provenance(),
    }
    write_json(metadata_path, metadata)
    write_json(score_dir / "manifest.json", {"records": score_records, "threshold": threshold})
    del state, bank, chunk_states, model, tokenizer
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return output


def _configure_eval(model_key: str, calib_seed: int):
    from experiments.fix_gsm8k_500 import direct_eval as direct
    from experiments.revision_full.run import dataset_provenance, model_provenance

    method = f"external_tacq__c{calib_seed}"
    state = state_path(model_key, calib_seed, "tacq")
    direct.OUT = RESULTS_DIR / "runtime" / model_key
    direct.SAMPLE_DIR = RESULTS_DIR / "samples"
    direct.LOG_DIR = RESULTS_DIR / "logs"
    for directory in (direct.OUT, direct.SAMPLE_DIR, direct.LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    direct.get_dataset = __import__(
        "experiments.revision_full.run", fromlist=["get_dataset"]
    ).get_dataset
    direct.MODEL_SPECS = {
        model_key: {
            "name": MODEL_SPECS[model_key]["display_name"],
            "path": MODEL_SPECS[model_key]["path"],
            "prompt_style": MODEL_SPECS[model_key]["prompt_style"],
        }
    }
    direct.METHOD_SPECS = {
        method: {
            "label": "TaCQ shared-backend adaptation",
            "kind": "mixed",
            "state": state,
            "models": {model_key},
        }
    }
    direct.CORE_METHODS = [method]
    direct.ONLINE_QUESTION_STOP = True
    direct.ROW_METADATA = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_manifest_sha256": dataset_provenance()["manifest_sha256"],
        "model_revision": model_provenance(model_key)["resolved_revision"],
        "canonical_test_set": "openai/gsm8k/main:test:all-1319",
        "tacq_manifest_sha256": require_manifest()["manifest_sha256"],
        "calibration_seed": calib_seed,
    }
    return direct, method


def smoke(model_key: str, calib_seed: int) -> None:
    import torch

    manifest = require_manifest()
    metadata_path = state_metadata_path(model_key, calib_seed, "tacq")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    state = state_path(model_key, calib_seed, "tacq")
    if metadata.get("state_sha256") != sha256(state):
        raise RuntimeError("TaCQ state changed before smoke test")
    direct, method = _configure_eval(model_key, calib_seed)
    model, tokenizer = direct.load_model(model_key, method)
    train, _ = direct.get_dataset()
    ids = [row["doc_id"] for row in manifest["importance"]["sample_records"][:32]]
    prefix = direct.build_fewshot(train, k=5)
    generated = 0
    marker_count = 0
    with torch.no_grad():
        for start in range(0, len(ids), DEFAULT_EVAL_BATCH_SIZE):
            batch_ids = ids[start : start + DEFAULT_EVAL_BATCH_SIZE]
            prompts = direct.build_model_prompts(
                model_key,
                tokenizer,
                train,
                prefix,
                [train[index]["question"] for index in batch_ids],
            )
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            width = int(encoded["input_ids"].shape[1])
            processor = GeneratedQuestionStopLogitsProcessor(
                tokenizer, width, tokenizer.eos_token_id
            )
            outputs = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                logits_processor=[processor],
            )
            texts = tokenizer.batch_decode(outputs[:, width:], skip_special_tokens=True)
            if len(texts) != len(batch_ids) or any(not isinstance(text, str) for text in texts):
                raise RuntimeError("TaCQ smoke generation returned invalid output")
            marker_count += sum(processor.finished or [])
            generated += len(texts)
    receipt = {
        "schema": "tacq-train-smoke-v1",
        "pass": generated == 32,
        "manifest_sha256": manifest["manifest_sha256"],
        "model_key": model_key,
        "calibration_seed": calib_seed,
        "train_only": True,
        "generated": generated,
        "marker_stops": marker_count,
        "state_sha256": metadata["state_sha256"],
        "save_reload_validated": True,
    }
    path = TACQ_DIR / "smoke" / f"{model_key}__c{calib_seed}.json"
    write_json(path, receipt)
    del model, tokenizer
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    if not receipt["pass"]:
        raise SystemExit(1)


def evaluate(model_key: str, calib_seed: int, force: bool = False) -> None:
    manifest = require_manifest()
    metadata = json.loads(
        state_metadata_path(model_key, calib_seed, "tacq").read_text(encoding="utf-8")
    )
    smoke_path = TACQ_DIR / "smoke" / f"{model_key}__c{calib_seed}.json"
    smoke_receipt = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (
        smoke_receipt.get("pass") is not True
        or smoke_receipt.get("train_only") is not True
        or smoke_receipt.get("manifest_sha256") != manifest["manifest_sha256"]
        or smoke_receipt.get("state_sha256") != metadata.get("state_sha256")
    ):
        raise RuntimeError("TaCQ train-only smoke gate is missing or stale")
    direct, method = _configure_eval(model_key, calib_seed)
    direct.evaluate(
        model_key,
        method,
        GSM8K_TEST_SIZE,
        DEFAULT_EVAL_BATCH_SIZE,
        MAX_NEW_TOKENS,
        force=force,
    )
    samples = direct.sample_path(model_key, method, GSM8K_TEST_SIZE)
    from experiments.revision_full.external_baselines import register

    register(
        model_key,
        "tacq",
        float(metadata["parameter_weighted_average_bits"]),
        samples,
        OFFICIAL_SOURCE_URL,
        manifest["official_source_commit"],
        Path(metadata["config"]),
        calibration_seed,
    )


def cleanup(model_key: str, calib_seed: int | None = None) -> None:
    """Remove only reconstructible TaCQ intermediates after registered outputs."""

    from experiments.revision_full.external_baselines import validate

    validate()
    seeds = [calib_seed] if calib_seed is not None else list(CALIB_SEEDS)
    for seed in seeds:
        record = OUT / "external_baselines" / f"{model_key}__tacq__c{seed}.json"
        if not record.exists():
            raise RuntimeError(f"Refusing cleanup before registration: {record}")
        score_dir = _score_dir(model_key, seed)
        if score_dir.exists():
            shutil.rmtree(score_dir)
        state_path(model_key, seed, "tacq").unlink(missing_ok=True)
    if calib_seed is None:
        gradient_dir = _gradient_dir(model_key)
        if gradient_dir.exists():
            shutil.rmtree(gradient_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--source-commit", default=OFFICIAL_SOURCE_COMMIT)
    freeze_parser.add_argument("--force", action="store_true")
    capture_parser = sub.add_parser("capture-importance")
    capture_parser.add_argument("--model", choices=TACQ_MODELS, required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--model", choices=TACQ_MODELS, required=True)
    build_parser.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)
    build_parser.add_argument("--force", action="store_true")
    smoke_parser = sub.add_parser("smoke")
    smoke_parser.add_argument("--model", choices=TACQ_MODELS, required=True)
    smoke_parser.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)
    eval_parser = sub.add_parser("evaluate")
    eval_parser.add_argument("--model", choices=TACQ_MODELS, required=True)
    eval_parser.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)
    eval_parser.add_argument("--force", action="store_true")
    cleanup_parser = sub.add_parser("cleanup")
    cleanup_parser.add_argument("--model", choices=TACQ_MODELS, required=True)
    cleanup_parser.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_manifest(args.source_commit, args.force)
    elif args.command == "capture-importance":
        capture_importance(args.model)
    elif args.command == "build":
        build_state(args.model, args.calib_seed, args.force)
    elif args.command == "smoke":
        smoke(args.model, args.calib_seed)
    elif args.command == "evaluate":
        evaluate(args.model, args.calib_seed, args.force)
    else:
        cleanup(args.model, args.calib_seed)


if __name__ == "__main__":
    main()

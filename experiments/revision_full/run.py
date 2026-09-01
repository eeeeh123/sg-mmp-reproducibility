"""Server-ready rejection-revision pipeline for SG-MMP.

This pipeline hard-separates GSM8K train development from the complete official
test evaluation. It intentionally does not reuse any layer selection produced
from the historical GSM8K test subsets.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") in (None, "", "expandable_segments:True"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from experiments.revision_full.protocol import (
    CALIB_LENGTH,
    CALIB_HESSIAN_TOKENS,
    CALIB_SAMPLES,
    CALIB_SEEDS,
    CAUSAL_PATCH_N,
    CAUSAL_PATCH_SEED,
    DEFAULT_EVAL_BATCH_SIZE,
    ELIGIBLE_SHORT_NAMES,
    GSM8K_TEST_SIZE,
    GROUP_SIZE,
    MODEL_SPECS,
    OUT,
    PROTOCOL_VERSION,
    QKV_SHORT_NAMES,
    RANDOM_ALLOCATIONS,
    RANDOM_CALIB_SEED,
    ROLE_SHORT_NAMES,
    RESULTS_DIR,
    SCREEN_DIR,
    SCREEN_N,
    SCREEN_SEEDS,
    SELECTION_BOOTSTRAP_REPLICATES,
    SELECTION_BOOTSTRAP_SEED,
    STATE_DIR,
    STATE_METADATA_DIR,
    TARGET_AVG_BITS,
    average_bits,
    fixed_causal_patch_indices,
    json_sha256,
    make_disjoint_screen_splits,
    method_id,
    role_priority_budget_match,
    scored_budget_match,
    select_layers_under_budget,
    state_metadata_path,
    state_path,
)
from experiments.revision_full.lifecycle import (
    bank_consumers_complete,
    broad_complete,
    cleanup_state_artifact,
    extra_complete,
    gsm8k_complete,
    state_consumers_complete,
)


for directory in [OUT, STATE_DIR, STATE_METADATA_DIR, SCREEN_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

LOCK_PATH = OUT / "protocol_lock.json"
ROLE_PRIORITY_VARIANTS = {
    "qkv_priority_matched": "qkv",
    "o_priority_matched": "o",
    "ffn_priority_matched": "ffn",
}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def status(stage: str, **extra) -> None:
    payload = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "protocol": PROTOCOL_VERSION,
        "stage": stage,
        **extra,
    }
    print("[status]", json.dumps(payload, ensure_ascii=False), flush=True)
    write_json(OUT / "status.json", payload)


def get_dataset():
    import experiments.fix_gsm8k_500.direct_eval as direct

    direct.OUT = OUT / "dataset_io"
    direct.OUT.mkdir(parents=True, exist_ok=True)
    return direct.get_dataset()


def prepare_protocol(force: bool = False) -> dict:
    if LOCK_PATH.exists() and not force:
        lock = read_json(LOCK_PATH)
        if lock.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"Existing lock uses another protocol: {LOCK_PATH}")
        return lock

    train, test = get_dataset()
    if len(test) != GSM8K_TEST_SIZE:
        raise RuntimeError(
            f"Expected {GSM8K_TEST_SIZE} GSM8K test examples, found {len(test)}"
        )
    splits = make_disjoint_screen_splits(len(train))
    full_test_indices = list(range(len(test)))
    causal_patch_indices = fixed_causal_patch_indices(len(test))
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "openai/gsm8k/main",
        "train_size": len(train),
        "test_size": len(test),
        "development_split": "train only",
        "screen_splits": splits,
        "screen_quantizer": {
            "method": "RTN-W4",
            "scheme": "group-wise asymmetric per-output-channel min-max",
            "group_size": GROUP_SIZE,
        },
        "final_test": {
            "selection": "all official test examples in dataset order",
            "n": len(full_test_indices),
            "indices_sha256": json_sha256(full_test_indices),
        },
        "causal_patch_diagnostic": {
            "selection": "fixed random test subset, independent of model outputs",
            "n": CAUSAL_PATCH_N,
            "seed": CAUSAL_PATCH_SEED,
            "indices": causal_patch_indices,
            "indices_sha256": json_sha256(causal_patch_indices),
            "inference_scope": "mechanistic diagnostic only; never a headline accuracy estimate",
        },
        "calibration": {
            "dataset": "wikitext train",
            "seeds": list(CALIB_SEEDS),
            "samples": CALIB_SAMPLES,
            "max_length": CALIB_LENGTH,
            "construction": "deterministic packed token stream with exact-length segments and no synthetic zero padding",
            "hessian_activation_tokens_per_module": CALIB_HESSIAN_TOKENS,
            "sample_balancing": "every calibration sequence contributes floor/ceil of the fixed activation-token reservoir",
        },
        "selection_budget": {
            "target_parameter_weighted_average_bits": TARGET_AVG_BITS,
            "rule": "rank aggregation across train splits, then accept ranked layers while budget allows",
        },
        "selection_stability": {
            "method": "bootstrap the three disjoint train-screen units with replacement and rerun budgeted selection",
            "replicates": SELECTION_BOOTSTRAP_REPLICATES,
            "seed": SELECTION_BOOTSTRAP_SEED,
            "inference_scope": "descriptive stability diagnostic; three split-level units",
        },
        "random_same_budget_allocations_per_model": RANDOM_ALLOCATIONS,
        "random_allocation_calibration_seed": RANDOM_CALIB_SEED,
        "uniform_precision_baselines": [4, 5, 6],
        "matched_module_controls": [
            *ROLE_PRIORITY_VARIANTS,
            "hessian_diag_matched",
        ],
        "broad_tasks": [
            "arc_challenge",
            "hellaswag",
            "mmlu",
            "mmlu_high_school_mathematics",
        ],
        "generative_transfer_tasks": [
            "svamp",
            "asdiv",
            "hendrycks_math500",
            "truthfulqa_gen",
        ],
        "required_external_matched_budget_baselines": {
            "models": ["qwen05", "qwen15"],
            "methods": ["tacq", "hawq_v2"],
            "canonical_test_n": GSM8K_TEST_SIZE,
        },
        "canonical_gsm8k_evaluator": "direct 5-shot greedy, complete official test set",
        "models": {
            key: {
                "name": spec["name"],
                "display_name": spec["display_name"],
                "role": spec["role"],
            }
            for key, spec in MODEL_SPECS.items()
        },
    }
    write_json(LOCK_PATH, lock)
    status("protocol_locked", path=str(LOCK_PATH), test_n=len(test))
    return lock


def require_protocol() -> dict:
    if not LOCK_PATH.exists():
        raise RuntimeError("Run `prepare` before any GPU experiment")
    lock = read_json(LOCK_PATH)
    if lock.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(
            f"Protocol lock is {lock.get('protocol_version')!r}, expected {PROTOCOL_VERSION!r}; "
            "rerun `prepare --force` before GPU experiments"
        )
    if lock.get("final_test", {}).get("n") != GSM8K_TEST_SIZE:
        raise RuntimeError("Protocol lock does not specify the complete GSM8K test set")
    return lock


def configure_determinism(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model_tokenizer(model_key: str, device: str = "cuda:0"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ptq.eval import cleanup_gpu

    spec = MODEL_SPECS[model_key]
    if device.startswith("cuda"):
        cleanup_gpu()
    kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    if device.startswith("cuda") and model_key in {"smollm", "gemma2"}:
        model = AutoModelForCausalLM.from_pretrained(str(spec["path"]), **kwargs)
        model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(spec["path"]),
            device_map=device if device.startswith("cuda") else None,
            **kwargs,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        str(spec["path"]), trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer


def transformer_layers(model) -> list[tuple[int, list[tuple[str, object]]]]:
    import torch.nn as nn
    from ptq.quant.mixed_precision import parse_layer_num

    grouped: dict[int, list[tuple[str, object]]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or "lm_head" in name:
            continue
        layer = parse_layer_num(name)
        short = name.split(".")[-1]
        if layer >= 0 and short in ELIGIBLE_SHORT_NAMES:
            grouped.setdefault(layer, []).append((name, module))
    return [(layer, grouped[layer]) for layer in sorted(grouped)]


def module_rows(model) -> list[dict]:
    rows = []
    for layer, specs in transformer_layers(model):
        for name, module in specs:
            rows.append(
                {
                    "name": name,
                    "layer": layer,
                    "short": name.split(".")[-1],
                    "n_params": int(module.weight.numel()),
                }
            )
    return rows


def screen_file(model_key: str, split_id: int) -> Path:
    return SCREEN_DIR / model_key / f"split_{split_id}.jsonl"


def _screen_records(path: Path) -> dict:
    return {(row["type"], row.get("layer")): row for row in read_jsonl(path)}


def _eval_loaded_model(
    model_key: str,
    model,
    tokenizer,
    examples: list[dict],
    batch_size: int,
    max_new_tokens: int,
) -> dict:
    from experiments.fix_gsm8k_500.direct_eval import (
        build_fewshot,
        build_model_prompts,
        extract_prediction,
        gold_answer,
        is_correct,
    )
    import experiments.fix_gsm8k_500.direct_eval as direct

    train, _ = get_dataset()
    direct.MODEL_SPECS[model_key] = {
        "prompt_style": MODEL_SPECS[model_key]["prompt_style"],
    }
    prefix = build_fewshot(train, k=5)
    correct = 0
    t0 = time.time()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = build_model_prompts(
            model_key, tokenizer, train, prefix, [example["question"] for example in batch]
        )
        encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        prompt_len = encoded["input_ids"].shape[1]
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
        for example, text in zip(batch, decoded):
            gold = gold_answer(example["answer"])
            correct += is_correct(extract_prediction(text), gold)
        done = start + len(batch)
        print(f"  screen {done}/{len(examples)} accuracy={100 * correct / done:.2f}", flush=True)
        del encoded, generated
    return {
        "n": len(examples),
        "correct": correct,
        "accuracy": 100 * correct / len(examples),
        "elapsed_s": time.time() - t0,
    }


def screen_model(
    model_key: str,
    split_id: int,
    batch_size: int,
    max_new_tokens: int,
    force: bool,
) -> None:
    import torch
    from ptq.eval import cleanup_gpu
    from ptq.quant.rtn import dequantize_tensor_rtn, quantize_tensor_rtn

    lock = require_protocol()
    split = lock["screen_splits"][split_id]
    path = screen_file(model_key, split_id)
    if force and path.exists():
        path.unlink()
    records = _screen_records(path)
    train, _ = get_dataset()
    examples = [train[index] for index in split["indices"]]
    configure_determinism(split["seed"])
    model, tokenizer = load_model_tokenizer(model_key)
    layers = transformer_layers(model)

    if ("baseline", None) not in records:
        baseline = _eval_loaded_model(
            model_key, model, tokenizer, examples, batch_size, max_new_tokens
        )
        append_jsonl(
            path,
            {
                "type": "baseline",
                "model_key": model_key,
                "split_id": split_id,
                "split_seed": split["seed"],
                **baseline,
            },
        )
    else:
        baseline = records[("baseline", None)]

    for ordinal, (layer, specs) in enumerate(layers, start=1):
        if ("layer", layer) in records:
            continue
        status(
            "screen_layer_start",
            model=model_key,
            split_id=split_id,
            layer=layer,
            ordinal=ordinal,
            total=len(layers),
        )
        saved = [(module, module.weight.detach().cpu().clone()) for _, module in specs]
        try:
            for _, module in specs:
                w_q, scale, zero = quantize_tensor_rtn(
                    module.weight.data, bits=4, group_size=GROUP_SIZE
                )
                dequantized = dequantize_tensor_rtn(w_q, scale, zero, GROUP_SIZE)
                module.weight.data.copy_(dequantized.to(module.weight.dtype))
                del w_q, scale, zero, dequantized
            result = _eval_loaded_model(
                model_key, model, tokenizer, examples, batch_size, max_new_tokens
            )
            append_jsonl(
                path,
                {
                    "type": "layer",
                    "model_key": model_key,
                    "split_id": split_id,
                    "split_seed": split["seed"],
                    "layer": layer,
                    "quantizer": "RTN-W4",
                    "group_size": GROUP_SIZE,
                    "drop_vs_fp16": float(baseline["accuracy"]) - result["accuracy"],
                    **result,
                },
            )
        finally:
            for module, weight in saved:
                module.weight.data.copy_(weight.to(module.weight.device, dtype=module.weight.dtype))
            del saved
            gc.collect()
            torch.cuda.empty_cache()

    del model, tokenizer
    cleanup_gpu()
    status("screen_complete", model=model_key, split_id=split_id, path=str(path))


def select_model(model_key: str) -> dict:
    lock = require_protocol()
    split_rows: dict[int, list[dict]] = {}
    for split in lock["screen_splits"]:
        path = screen_file(model_key, split["split_id"])
        rows = [row for row in read_jsonl(path) if row["type"] == "layer"]
        if not rows:
            raise RuntimeError(f"Missing completed screen: {path}")
        split_rows[int(split["split_id"])] = rows

    all_layers = sorted({int(row["layer"]) for rows in split_rows.values() for row in rows})
    by_split_layer = {
        (split_id, int(row["layer"])): row
        for split_id, rows in split_rows.items()
        for row in rows
    }
    for split_id, rows in split_rows.items():
        if {int(row["layer"]) for row in rows} != set(all_layers):
            raise RuntimeError(f"Incomplete layer screen for {model_key}/split {split_id}")

    ranks = {}
    for split_id, rows in split_rows.items():
        ordered = sorted(rows, key=lambda row: (-float(row["drop_vs_fp16"]), int(row["layer"])))
        ranks[split_id] = {int(row["layer"]): rank + 1 for rank, row in enumerate(ordered)}

    ranking = []
    for layer in all_layers:
        drops = [
            float(by_split_layer[(split_id, layer)]["drop_vs_fp16"])
            for split_id in split_rows
        ]
        layer_ranks = [ranks[split_id][layer] for split_id in split_rows]
        ranking.append(
            {
                "layer": layer,
                "mean_drop": statistics.mean(drops),
                "drop_std": statistics.pstdev(drops),
                "mean_rank": statistics.mean(layer_ranks),
                "per_split_drop": drops,
                "per_split_rank": layer_ranks,
            }
        )
    ranking.sort(key=lambda row: (row["mean_rank"], -row["mean_drop"], row["layer"]))

    model, tokenizer = load_model_tokenizer(model_key, device="cpu")
    modules = module_rows(model)
    selection = select_layers_under_budget(ranking, modules, TARGET_AVG_BITS)
    selected = set(selection["selected_layers"])
    top_sets = []
    for split_id in split_rows:
        ordered = sorted(
            all_layers,
            key=lambda layer: (ranks[split_id][layer], layer),
        )
        top_sets.append(set(ordered[: len(selected)]))
    jaccards = [
        len(selected & top_set) / len(selected | top_set) if selected | top_set else 1.0
        for top_set in top_sets
    ]
    split_ids = sorted(split_rows)
    bootstrap_rng = random.Random(SELECTION_BOOTSTRAP_SEED)
    bootstrap_jaccards = []
    bootstrap_exact = 0
    bootstrap_inclusion = {layer: 0 for layer in all_layers}
    for _ in range(SELECTION_BOOTSTRAP_REPLICATES):
        sampled_split_ids = [bootstrap_rng.choice(split_ids) for _ in split_ids]
        sampled_ranking = []
        for layer in all_layers:
            sampled_ranks = [ranks[split_id][layer] for split_id in sampled_split_ids]
            sampled_drops = [
                float(by_split_layer[(split_id, layer)]["drop_vs_fp16"])
                for split_id in sampled_split_ids
            ]
            sampled_ranking.append(
                {
                    "layer": layer,
                    "mean_rank": statistics.mean(sampled_ranks),
                    "mean_drop": statistics.mean(sampled_drops),
                }
            )
        sampled_ranking.sort(
            key=lambda row: (row["mean_rank"], -row["mean_drop"], row["layer"])
        )
        bootstrap_selected = set(
            select_layers_under_budget(
                sampled_ranking, modules, TARGET_AVG_BITS
            )["selected_layers"]
        )
        for layer in bootstrap_selected:
            bootstrap_inclusion[layer] += 1
        union = selected | bootstrap_selected
        bootstrap_jaccards.append(
            len(selected & bootstrap_selected) / len(union) if union else 1.0
        )
        bootstrap_exact += int(bootstrap_selected == selected)
    bootstrap_stability = {
        "unit": "three disjoint GSM8K-train screen splits",
        "replicates": SELECTION_BOOTSTRAP_REPLICATES,
        "seed": SELECTION_BOOTSTRAP_SEED,
        "aggregate_selected_layers": sorted(selected),
        "mean_jaccard_vs_aggregate": statistics.mean(bootstrap_jaccards),
        "jaccard_ci95": [
            sorted(bootstrap_jaccards)[
                int(0.025 * SELECTION_BOOTSTRAP_REPLICATES)
            ],
            sorted(bootstrap_jaccards)[
                int(0.975 * SELECTION_BOOTSTRAP_REPLICATES)
            ],
        ],
        "exact_set_rate": bootstrap_exact / SELECTION_BOOTSTRAP_REPLICATES,
        "layer_inclusion_probability": {
            str(layer): bootstrap_inclusion[layer]
            / SELECTION_BOOTSTRAP_REPLICATES
            for layer in all_layers
        },
        "interpretation_limit": "three split-level units; report descriptively and do not claim universal selection invariance",
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "model": MODEL_SPECS[model_key]["display_name"],
        "selection_data": "GSM8K train development splits only",
        "test_data_used": False,
        "screen_files": [
            str(screen_file(model_key, split_id).relative_to(OUT.parent))
            for split_id in split_rows
        ],
        "screen_file_sha256": {
            str(split_id): json_sha256(read_jsonl(screen_file(model_key, split_id)))
            for split_id in split_rows
        },
        "ranking_rule": "ascending mean rank across three disjoint train splits; ties use mean drop",
        "ranking": ranking,
        "top_set_jaccard_vs_aggregate": jaccards,
        "mean_top_set_jaccard": statistics.mean(jaccards),
        "selection_bootstrap": bootstrap_stability,
        "module_rows": modules,
        **selection,
    }
    path = OUT / "selections" / f"{model_key}.json"
    write_json(path, payload)
    del model, tokenizer
    status(
        "selection_locked",
        model=model_key,
        selected_layers=selection["selected_layers"],
        avg_bits=selection["actual_avg_bits"],
        path=str(path),
    )
    return payload


def selection_for(model_key: str) -> dict:
    path = OUT / "selections" / f"{model_key}.json"
    if not path.exists():
        raise RuntimeError(f"Run `select --model {model_key}` first")
    selection = read_json(path)
    if selection.get("test_data_used") is not False:
        raise RuntimeError(f"Selection is not test-clean: {path}")
    return selection


def save_torch_atomic(value, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def build_bank(model_key: str, calib_seed: int, force: bool) -> Path:
    from ptq.data import get_calib_dataset
    from ptq.eval import cleanup_gpu
    from ptq.quant.mixed_precision import quantize_model_precision_bank

    require_protocol()
    path = state_path(model_key, calib_seed, "precision_bank")
    metadata_path = state_metadata_path(model_key, calib_seed, "precision_bank")
    if path.exists() and not force:
        if not metadata_path.exists():
            raise RuntimeError(
                f"Precision bank exists without metadata; rerun build-bank --force: {path}"
            )
        return path
    if not force and bank_consumers_complete(model_key, calib_seed):
        status(
            "precision_bank_not_needed",
            model=model_key,
            calib_seed=calib_seed,
            reason="all dependent evidence is complete",
        )
        return path
    configure_determinism(calib_seed)
    model, tokenizer = load_model_tokenizer(model_key)
    calibration = get_calib_dataset(
        tokenizer,
        n_samples=CALIB_SAMPLES,
        max_length=CALIB_LENGTH,
        dataset_name="wikitext",
        seed=calib_seed,
    )
    bank = quantize_model_precision_bank(
        model,
        calibration,
        bits_w4=4,
        group_size=GROUP_SIZE,
        uniform_bits=(5, 6),
        max_calib_tokens=CALIB_HESSIAN_TOKENS,
    )
    save_torch_atomic(bank, path)
    write_json(
        metadata_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "model_key": model_key,
            "calibration_dataset": "wikitext train",
            "calibration_seed": calib_seed,
            "calibration_samples": CALIB_SAMPLES,
            "calibration_length": CALIB_LENGTH,
            "calibration_construction": "packed exact-length segments; no synthetic zero padding",
            "hessian_activation_tokens_per_module": CALIB_HESSIAN_TOKENS,
            "hessian_sample_balancing": "all calibration samples contribute",
            "group_size": GROUP_SIZE,
            "precision_entries": [4, 5, 6, 8],
            "modules": len(bank),
            "selection_score": "calibration-weighted diagonal-Hessian reconstruction NMSE",
            "bytes": path.stat().st_size,
        },
    )
    del bank, calibration, model, tokenizer
    cleanup_gpu()
    status("precision_bank_saved", model=model_key, calib_seed=calib_seed, path=str(path))
    return path


def _random_module_allocation(selection: dict, allocation_id: int) -> set[str]:
    rows = selection["module_rows"]
    target_names = set(selection["w8_module_names"])
    target_params = sum(int(row["n_params"]) for row in rows if row["name"] in target_names)
    best_names: set[str] = set()
    best_gap = target_params
    candidates = list(rows)
    rng = random.Random(20262001 + allocation_id)
    for _ in range(5000):
        rng.shuffle(candidates)
        chosen = set()
        total = 0
        for row in candidates:
            size = int(row["n_params"])
            if total + size <= target_params:
                chosen.add(row["name"])
                total += size
        gap = target_params - total
        if gap < best_gap:
            best_names = chosen
            best_gap = gap
            if gap == 0:
                break
    return best_names


def _policy_for_variant(
    model_key: str,
    variant: str,
    selected_modules_override: set[str] | None = None,
):
    from ptq.quant.mixed_precision import parse_layer_num

    selection = selection_for(model_key)
    selected = set(int(layer) for layer in selection["selected_layers"])
    selected_modules: set[str] | None = selected_modules_override
    allocation_details: dict | None = None
    if variant.startswith("random_modules_"):
        allocation_id = int(variant.rsplit("_", 1)[1])
        selected_modules = _random_module_allocation(selection, allocation_id)
        selected = set()
    elif variant in ROLE_PRIORITY_VARIANTS:
        role = ROLE_PRIORITY_VARIANTS[variant]
        allocation_details = role_priority_budget_match(
            selection["module_rows"],
            set(selection["w8_module_names"]),
            ROLE_SHORT_NAMES[role],
            seed=20263001 + list(ROLE_PRIORITY_VARIANTS).index(variant),
        )
        selected_modules = set(allocation_details["selected_module_names"])
        selected = set()
    elif variant == "hessian_diag_matched":
        if selected_modules is None:
            raise ValueError("hessian_diag_matched requires a scored module allocation")
        selected = set()
    elif variant.startswith("random_"):
        allocation_id = int(variant.split("_", 1)[1])
        all_layers = sorted({int(row["layer"]) for row in selection["module_rows"]})
        rng = random.Random(20261001 + allocation_id)
        selected = set(rng.sample(all_layers, len(selected)))

    def policy(module_idx: int, name: str, short: str) -> str:
        if short not in ELIGIBLE_SHORT_NAMES:
            return "w4"
        layer = parse_layer_num(name)
        if variant == "gptq_w4":
            return "w4"
        if variant == "qkv_only":
            return "w8" if short in QKV_SHORT_NAMES else "w4"
        if variant == "o_only":
            return "w8" if short == "o_proj" else "w4"
        if variant == "ffn_only":
            return "w8" if short in {"gate_proj", "up_proj", "down_proj"} else "w4"
        if variant.startswith("random_modules_"):
            return "w8" if name in selected_modules else "w4"
        if variant in ROLE_PRIORITY_VARIANTS or variant == "hessian_diag_matched":
            return "w8" if name in selected_modules else "w4"
        if variant == "sg_mmp" or variant.startswith("random_"):
            return "w8" if layer in selected or short in QKV_SHORT_NAMES else "w4"
        raise ValueError(f"Unknown bank variant: {variant}")

    return policy, selected, selected_modules, allocation_details


def materialize(model_key: str, calib_seed: int, variant: str, force: bool) -> Path:
    import torch
    from ptq.quant.mixed_precision import compose_precision_state

    if variant.startswith("random_modules_"):
        allocation_id = int(variant.rsplit("_", 1)[1])
        if not 0 <= allocation_id < RANDOM_ALLOCATIONS:
            raise ValueError(f"Random module allocation id must be in [0, {RANDOM_ALLOCATIONS - 1}]")
    elif variant.startswith("random_"):
        allocation_id = int(variant.split("_", 1)[1])
        if not 0 <= allocation_id < RANDOM_ALLOCATIONS:
            raise ValueError(f"Random allocation id must be in [0, {RANDOM_ALLOCATIONS - 1}]")
    elif variant not in {
        "gptq_w4",
        "qkv_only",
        "o_only",
        "ffn_only",
        "sg_mmp",
        *ROLE_PRIORITY_VARIANTS,
        "hessian_diag_matched",
    }:
        raise ValueError(
            "variant must be a core, role-priority, Hessian, "
            f"random_0..{RANDOM_ALLOCATIONS - 1}, or "
            f"random_modules_0..{RANDOM_ALLOCATIONS - 1}"
        )

    output_path = state_path(model_key, calib_seed, variant)
    metadata_path = state_metadata_path(model_key, calib_seed, variant)
    if output_path.exists() and not force:
        if not metadata_path.exists():
            raise RuntimeError(
                f"State exists without metadata; rerun materialize --force: {output_path}"
            )
        return output_path
    if not force and state_consumers_complete(model_key, calib_seed, variant):
        status(
            "state_not_needed",
            model=model_key,
            calib_seed=calib_seed,
            variant=variant,
            reason="all dependent evidence is complete",
        )
        return output_path
    bank_path = state_path(model_key, calib_seed, "precision_bank")
    if not bank_path.exists():
        raise FileNotFoundError(f"Run build-bank first: {bank_path}")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False, mmap=True)
    scored_allocation = None
    selected_override = None
    if variant == "hessian_diag_matched":
        score_key = "hessian_diag_reconstruction_nmse"
        scores = {
            name: float(entry.get("scores", {}).get(score_key, 0.0))
            for name, entry in bank.items()
        }
        if not scores or max(scores.values(), default=0.0) <= 0:
            raise RuntimeError(
                f"Precision bank has no Hessian-diagonal scores; rebuild it under {PROTOCOL_VERSION}"
            )
        selection = selection_for(model_key)
        scored_allocation = scored_budget_match(
            selection["module_rows"], set(selection["w8_module_names"]), scores
        )
        selected_override = set(scored_allocation["selected_module_names"])
    policy, selected, selected_modules, allocation_details = _policy_for_variant(
        model_key, variant, selected_override
    )
    state = compose_precision_state(bank, policy)
    save_torch_atomic(state, output_path)

    selection = selection_for(model_key)
    modules = selection["module_rows"]
    w8_names = {name for name, entry in state.items() if entry.get("method") == "w8_perchannel"}
    avg = average_bits(modules, w8_names & {row["name"] for row in modules})
    if variant.startswith("random_") and not variant.startswith("random_modules_") and abs(avg - selection["actual_avg_bits"]) > 1e-6:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Random allocation is not budget matched: {avg} vs {selection['actual_avg_bits']}"
        )
    matched_module_variant = (
        variant.startswith("random_modules_")
        or variant in ROLE_PRIORITY_VARIANTS
        or variant == "hessian_diag_matched"
    )
    if matched_module_variant and abs(avg - selection["actual_avg_bits"]) > 0.01:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Matched module allocation differs from the SG-MMP budget by more than 0.01 bits: "
            f"{avg} vs {selection['actual_avg_bits']}"
        )
    write_json(
        metadata_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "model_key": model_key,
            "calibration_seed": calib_seed,
            "variant": variant,
            "selected_layers": sorted(selected),
            "selected_module_names": sorted(selected_modules) if selected_modules is not None else None,
            "allocation_details": scored_allocation or allocation_details,
            "parameter_weighted_average_bits": avg,
            "state_entries": len(state),
            "bytes": output_path.stat().st_size,
        },
    )
    del state, bank
    status(
        "state_materialized",
        model=model_key,
        calib_seed=calib_seed,
        variant=variant,
        avg_bits=avg,
        path=str(output_path),
    )
    return output_path


def quantize_uniform(model_key: str, calib_seed: int, bits: int, force: bool) -> Path:
    import torch
    from ptq.quant.mixed_precision import compose_precision_state

    require_protocol()
    if bits not in {5, 6}:
        raise ValueError("Uniform comparison bits must be 5 or 6")
    variant = f"gptq_w{bits}"
    path = state_path(model_key, calib_seed, variant)
    metadata_path = state_metadata_path(model_key, calib_seed, variant)
    if path.exists() and not force:
        if not metadata_path.exists():
            raise RuntimeError(
                f"State exists without metadata; rerun quantize-uniform --force: {path}"
            )
        return path
    if not force and state_consumers_complete(model_key, calib_seed, variant):
        status(
            "state_not_needed",
            model=model_key,
            calib_seed=calib_seed,
            variant=variant,
            reason="all dependent evidence is complete",
        )
        return path
    bank_path = state_path(model_key, calib_seed, "precision_bank")
    if not bank_path.exists():
        raise FileNotFoundError(f"Run build-bank first: {bank_path}")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False, mmap=True)
    action = f"w{bits}"
    state = compose_precision_state(bank, lambda *_: action)
    save_torch_atomic(state, path)
    write_json(
        metadata_path,
        {
            "protocol_version": PROTOCOL_VERSION,
            "model_key": model_key,
            "variant": variant,
            "calibration_seed": calib_seed,
            "source_precision_bank": str(bank_path),
            "shared_calibration_activations_and_hessian": True,
            "parameter_weighted_average_bits": float(bits),
            "state_entries": len(state),
            "bytes": path.stat().st_size,
        },
    )
    del state, bank
    status(
        "uniform_state_saved",
        model=model_key,
        calib_seed=calib_seed,
        bits=bits,
        path=str(path),
    )
    return path


def configure_direct_eval(
    model_key: str,
    variant: str,
    calib_seed: int | None,
):
    import experiments.fix_gsm8k_500.direct_eval as direct

    method = method_id(variant, calib_seed)
    spec = MODEL_SPECS[model_key]
    direct.OUT = RESULTS_DIR
    direct.SAMPLE_DIR = RESULTS_DIR / "samples"
    direct.LOG_DIR = RESULTS_DIR / "logs"
    for path in [direct.OUT, direct.SAMPLE_DIR, direct.LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    direct.MODEL_SPECS = {
        model_key: {
            "name": spec["display_name"],
            "path": spec["path"],
            "prompt_style": spec["prompt_style"],
        }
    }
    if variant == "fp16":
        direct.METHOD_SPECS = {method: {"label": "FP16", "kind": "fp16"}}
    else:
        if calib_seed is None:
            raise ValueError(f"{variant} requires --calib-seed")
        path = state_path(model_key, calib_seed, variant)
        if not path.exists():
            raise FileNotFoundError(path)
        kind = "gptq" if variant in {"gptq_w5", "gptq_w6"} else "mixed"
        direct.METHOD_SPECS = {
            method: {
                "label": variant,
                "kind": kind,
                "state": path,
                "models": {model_key},
            }
        }
    direct.CORE_METHODS = [method]
    return direct, method


def evaluate_full(
    model_key: str,
    variant: str,
    calib_seed: int | None,
    batch_size: int,
    max_new_tokens: int,
    force: bool,
) -> None:
    require_protocol()
    if not force and gsm8k_complete(model_key, variant, calib_seed):
        status(
            "full_evaluation_already_complete",
            model=model_key,
            variant=variant,
            calib_seed=calib_seed,
        )
        return
    direct, method = configure_direct_eval(model_key, variant, calib_seed)
    direct.evaluate(
        model_key,
        method,
        GSM8K_TEST_SIZE,
        batch_size,
        max_new_tokens,
        force=force,
    )


def evaluate_allocation(
    model_key: str,
    variant: str,
    calib_seed: int,
    batch_size: int,
    keep_state: bool,
) -> None:
    if not (variant.startswith("random_") or variant.startswith("random_modules_")):
        raise ValueError("evaluate-allocation only accepts random_* or random_modules_* variants")
    completed = gsm8k_complete(model_key, variant, calib_seed)
    if completed and not keep_state:
        cleanup_state_artifact(model_key, calib_seed, variant)
        return
    materialize(model_key, calib_seed, variant, force=completed and keep_state)
    if completed:
        return
    evaluate_full(
        model_key,
        variant,
        calib_seed,
        batch_size,
        max_new_tokens=256,
        force=False,
    )
    if not gsm8k_complete(model_key, variant, calib_seed):
        raise RuntimeError(
            f"Keeping allocation state because evaluation is incomplete: {model_key}/{variant}"
        )
    if not keep_state:
        cleanup_state_artifact(model_key, calib_seed, variant)


def evaluate_broad(
    model_key: str,
    variant: str,
    calib_seed: int | None,
    batch_size: int,
    force: bool,
) -> None:
    from ptq.eval import cleanup_gpu, run_eval_on_model

    require_protocol()
    if not force and broad_complete(model_key, variant, calib_seed):
        status(
            "broad_evaluation_already_complete",
            model=model_key,
            variant=variant,
            calib_seed=calib_seed,
        )
        return
    direct, method = configure_direct_eval(model_key, variant, calib_seed)
    model, tokenizer = direct.load_model(model_key, method)
    path = RESULTS_DIR / "broad" / f"{model_key}__{method}.json"
    if force and path.exists():
        path.unlink()
    tasks = [
        "arc_challenge",
        "hellaswag",
        "mmlu",
        "mmlu_high_school_mathematics",
    ]
    record = read_json(path) if path.exists() else {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "method": method,
        "canonical_gsm8k_source": "evaluate-full only; GSM8K intentionally excluded here",
        "limits": {task: None for task in tasks},
        "scores": {},
    }
    for task in tasks:
        if task in record["scores"]:
            continue
        score = run_eval_on_model(
            model,
            tokenizer,
            [task],
            batch_size=batch_size,
            max_gen_toks=256,
            limit=None,
        )
        record["scores"].update(score)
        write_json(path, record)
    del model, tokenizer
    cleanup_gpu()
    status("broad_full_complete", model=model_key, method=method, path=str(path))


def evaluate_extra(
    model_key: str,
    variant: str,
    calib_seed: int | None,
    batch_size: int,
    force: bool,
) -> None:
    import torch
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from ptq.eval import cleanup_gpu

    require_protocol()
    if not force and extra_complete(model_key, variant, calib_seed):
        status(
            "extra_evaluation_already_complete",
            model=model_key,
            variant=variant,
            calib_seed=calib_seed,
        )
        return
    direct, method = configure_direct_eval(model_key, variant, calib_seed)
    model, tokenizer = direct.load_model(model_key, method)
    lm_model = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_batch_size=batch_size,
    )
    tasks = ["svamp", "asdiv", "hendrycks_math500", "truthfulqa_gen"]
    task_manager = TaskManager(
        include_path=str(OUT.parents[1] / "fix_svamp_ood")
    )
    path = RESULTS_DIR / "extra" / f"{model_key}__{method}.json"
    if force and path.exists():
        path.unlink()
    record = read_json(path) if path.exists() else {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "method": method,
        "limits": {task: None for task in tasks},
        "results": {},
    }
    for task in tasks:
        if task in record["results"]:
            continue
        result = simple_evaluate(
            model=lm_model,
            tasks=[task],
            task_manager=task_manager,
            batch_size=batch_size,
            limit=None,
            log_samples=True,
            gen_kwargs={
                "temperature": 0.0,
                "max_new_tokens": 256,
                "do_sample": False,
            },
        )
        samples = result.get("samples", {}).get(task, [])
        record["results"][task] = {
            "metrics": result.get("results", {}).get(task, {}),
            "n_samples": len(samples),
            "samples": samples,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        torch.cuda.empty_cache()
    del lm_model, model, tokenizer
    cleanup_gpu()
    status("extra_full_complete", model=model_key, method=method, path=str(path))


def parse_model(value: str) -> str:
    if value not in MODEL_SPECS:
        raise ValueError(f"Unknown model {value}; choose from {list(MODEL_SPECS)}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("screen")
    p.add_argument("--model", required=True)
    p.add_argument("--split-id", type=int, choices=range(len(SCREEN_SEEDS)), required=True)
    p.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("select")
    p.add_argument("--model", required=True)

    p = sub.add_parser("build-bank")
    p.add_argument("--model", required=True)
    p.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("materialize")
    p.add_argument("--model", required=True)
    p.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("quantize-uniform")
    p.add_argument("--model", required=True)
    p.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)
    p.add_argument("--bits", type=int, choices=[5, 6], required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("evaluate-allocation")
    p.add_argument("--model", required=True)
    p.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    p.add_argument("--keep-state", action="store_true")

    p = sub.add_parser("cleanup-state")
    p.add_argument("--model", required=True)
    p.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)
    p.add_argument("--variant", required=True)

    p = sub.add_parser("cleanup-bank")
    p.add_argument("--model", required=True)
    p.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)

    for command in ["evaluate-full", "evaluate-broad", "evaluate-extra"]:
        p = sub.add_parser(command)
        p.add_argument("--model", required=True)
        p.add_argument("--variant", required=True)
        p.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS)
        p.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
        p.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        lock = prepare_protocol(force=args.force)
        print(
            json.dumps(
                {
                    "protocol_version": lock["protocol_version"],
                    "train_size": lock["train_size"],
                    "test_size": lock["test_size"],
                    "screen_split_sizes": [row["n"] for row in lock["screen_splits"]],
                    "lock_path": str(LOCK_PATH),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    model_key = parse_model(args.model)
    if args.command == "screen":
        screen_model(
            model_key,
            args.split_id,
            args.batch_size,
            args.max_new_tokens,
            args.force,
        )
    elif args.command == "select":
        selection = select_model(model_key)
        print(
            json.dumps(
                {
                    "model": selection["model"],
                    "selected_layers": selection["selected_layers"],
                    "actual_avg_bits": selection["actual_avg_bits"],
                    "mean_top_set_jaccard": selection["mean_top_set_jaccard"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "build-bank":
        print(build_bank(model_key, args.calib_seed, args.force))
    elif args.command == "materialize":
        print(materialize(model_key, args.calib_seed, args.variant, args.force))
    elif args.command == "quantize-uniform":
        print(quantize_uniform(model_key, args.calib_seed, args.bits, args.force))
    elif args.command == "evaluate-allocation":
        evaluate_allocation(
            model_key,
            args.variant,
            args.calib_seed,
            args.batch_size,
            args.keep_state,
        )
    elif args.command == "cleanup-state":
        cleanup_state_artifact(model_key, args.calib_seed, args.variant)
    elif args.command == "cleanup-bank":
        cleanup_state_artifact(model_key, args.calib_seed, "precision_bank")
    elif args.command == "evaluate-full":
        evaluate_full(
            model_key,
            args.variant,
            args.calib_seed,
            args.batch_size,
            256,
            args.force,
        )
    elif args.command == "evaluate-broad":
        evaluate_broad(
            model_key, args.variant, args.calib_seed, args.batch_size, args.force
        )
    elif args.command == "evaluate-extra":
        evaluate_extra(
            model_key, args.variant, args.calib_seed, args.batch_size, args.force
        )


if __name__ == "__main__":
    main()

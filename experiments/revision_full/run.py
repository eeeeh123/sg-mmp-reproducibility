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
from contextlib import contextmanager
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
    BROAD_TASKS,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_FORMAT_BATCH_SIZE,
    ELIGIBLE_SHORT_NAMES,
    EXTRA_TASKS,
    GSM8K_TEST_SIZE,
    GROUP_SIZE,
    MAX_CONCURRENT_RAM_BUILDERS,
    MAX_NEW_TOKENS,
    MIN_AVAILABLE_RAM_GIB,
    MODEL_SPECS,
    OUT,
    PROTOCOL_VERSION,
    QKV_SHORT_NAMES,
    RANDOM_ALLOCATIONS,
    RANDOM_CALIB_SEED,
    RAM_BUILDER_WAIT_POLL_SECONDS,
    RAM_BUILDER_WAIT_TIMEOUT_SECONDS,
    ROLE_SHORT_NAMES,
    RESULTS_DIR,
    SCREEN_DIR,
    SCREEN_N,
    SCREEN_CALIB_SEEDS,
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
    make_random_layer_allocation_plan,
    make_unique_random_module_allocations,
    method_id,
    role_priority_budget_match,
    scored_budget_match,
    select_layers_under_budget,
    state_metadata_path,
    state_path,
    validate_random_allocation_manifest,
)
from experiments.revision_full.lifecycle import (
    bank_consumers_complete,
    broad_complete,
    cleanup_state_artifact,
    extra_complete,
    extra_task_complete,
    gsm8k_complete,
    gsm8k_sample_path,
    state_consumers_complete,
    cleanup_screen_state_artifact,
)
from experiments.revision_full.download_core_datasets import (
    MANIFEST_PATH as DATASET_MANIFEST_PATH,
    snapshot_sha256 as dataset_snapshot_sha256,
)
from experiments.revision_full.download_models import stable_model_record


for directory in [OUT, STATE_DIR, STATE_METADATA_DIR, SCREEN_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

LOCK_PATH = OUT / "protocol_lock.json"
ROLE_PRIORITY_VARIANTS = {
    "qkv_priority_matched": "qkv",
    "o_priority_matched": "o",
    "ffn_priority_matched": "ffn",
}


def write_json(path: Path, value, *, default=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    dump_options = {"default": default} if default is not None else {}
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, **dump_options),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require_current_state_metadata(
    metadata_path: Path,
    state_file: Path,
    model_key: str,
    calib_seed: int,
    variant: str,
) -> dict:
    if not metadata_path.exists():
        raise RuntimeError(f"State exists without metadata: {state_file}")
    record = read_json(metadata_path)
    if (
        record.get("protocol_version") != PROTOCOL_VERSION
        or record.get("model_key") != model_key
        or int(record.get("calibration_seed", -1)) != calib_seed
        or record.get("variant") != variant
        or record.get("model_snapshot") != model_provenance(model_key)
        or record.get("dataset_snapshot") != dataset_provenance()
        or int(record.get("bytes", -1)) != state_file.stat().st_size
    ):
        raise RuntimeError(
            f"State metadata/provenance mismatch; do not reuse or mix it: {state_file}"
        )
    return record


def dataset_provenance() -> dict:
    if not DATASET_MANIFEST_PATH.exists():
        raise RuntimeError(
            "Missing frozen dataset manifest; run "
            "experiments/revision_full/download_core_datasets.py first"
        )
    manifest = read_json(DATASET_MANIFEST_PATH)
    if manifest.get("schema_version") != 2:
        raise RuntimeError(f"Unsupported dataset manifest: {DATASET_MANIFEST_PATH}")
    snapshot_hash = dataset_snapshot_sha256(manifest)
    if manifest.get("snapshot_sha256") != snapshot_hash:
        raise RuntimeError(f"Dataset snapshot identity is invalid: {DATASET_MANIFEST_PATH}")
    return {
        "manifest": str(DATASET_MANIFEST_PATH.relative_to(OUT.parent)),
        "manifest_sha256": snapshot_hash,
    }


def model_provenance(model_key: str) -> dict:
    path = OUT / "model_snapshot_manifest.json"
    if not path.exists():
        raise RuntimeError(
            "Missing model snapshot manifest; run download_models.py on the server first"
        )
    record = read_json(path).get("models", {}).get(model_key)
    if not record or len(str(record.get("resolved_revision", ""))) != 40:
        raise RuntimeError(f"Missing immutable checkpoint revision for {model_key}")
    if not record.get("weight_file_records"):
        raise RuntimeError(
            f"Model manifest for {model_key} lacks file hashes; rerun download_models.py"
        )
    return stable_model_record(record)


def frozen_core_cache_path(dataset_key: str, filename: str) -> Path:
    manifest = read_json(DATASET_MANIFEST_PATH)
    records = manifest.get("core", {}).get(dataset_key, {}).get("cache_files", [])
    matches = [
        Path(record["path"])
        for record in records
        if Path(record["path"]).name == filename
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Frozen dataset manifest must contain exactly one {filename} for {dataset_key}"
        )
    path = matches[0]
    if not path.exists():
        raise FileNotFoundError(f"Frozen dataset cache is missing: {path}")
    return path


def frozen_arrow_rows(dataset_key: str, filename: str) -> list[dict]:
    import pyarrow.ipc as pa_ipc

    path = frozen_core_cache_path(dataset_key, filename)
    with pa_ipc.open_stream(str(path)) as reader:
        return reader.read_all().to_pylist()


def frozen_wikitext_calibration(tokenizer, calib_seed: int):
    from ptq.data import _packed_random_segments

    rows = frozen_arrow_rows(
        "Salesforce/wikitext/wikitext-2-raw-v1/train", "wikitext-train.arrow"
    )
    texts = [row["text"] for row in rows if row.get("text") and row["text"].strip()]
    return _packed_random_segments(
        tokenizer,
        texts,
        n_samples=CALIB_SAMPLES,
        max_length=CALIB_LENGTH,
        seed=calib_seed,
    )


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
    suffix = f"__{extra['model']}" if extra.get("model") in MODEL_SPECS else ""
    write_json(OUT / f"status{suffix}.json", payload)


def system_available_ram_gib() -> float | None:
    """Return Linux MemAvailable without treating swap as usable experiment RAM."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024**2
    return None


def supports_posix_file_lock() -> bool:
    return os.name == "posix"


@contextmanager
def ram_builder_slot(stage: str, model_key: str, calib_seed: int):
    """Serialize activation-heavy state builders across the two GPU workers."""
    if MAX_CONCURRENT_RAM_BUILDERS != 1:
        yield
        return
    if not supports_posix_file_lock():
        raise RuntimeError(
            "The single RAM-builder mode requires a POSIX server with fcntl locking"
        )

    import fcntl

    lock_path = OUT / "ram_builder.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as stream:
        status(
            "ram_builder_wait",
            model=model_key,
            calib_seed=calib_seed,
            builder_stage=stage,
        )
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            wait_started = time.monotonic()
            available = system_available_ram_gib()
            while (
                available is not None
                and available < MIN_AVAILABLE_RAM_GIB
            ):
                waited_seconds = time.monotonic() - wait_started
                if (
                    RAM_BUILDER_WAIT_TIMEOUT_SECONDS > 0
                    and waited_seconds >= RAM_BUILDER_WAIT_TIMEOUT_SECONDS
                ):
                    raise RuntimeError(
                        f"Only {available:.1f} GiB system RAM is available before "
                        f"{stage} after waiting {waited_seconds:.0f} seconds; at least "
                        f"{MIN_AVAILABLE_RAM_GIB:.1f} GiB is required. Stop competing "
                        "RAM-heavy jobs and resume; swap is not counted."
                    )
                status(
                    "ram_builder_memory_wait",
                    model=model_key,
                    calib_seed=calib_seed,
                    builder_stage=stage,
                    available_ram_gib=round(available, 1),
                    required_ram_gib=MIN_AVAILABLE_RAM_GIB,
                    waited_seconds=round(waited_seconds, 1),
                    timeout_seconds=(
                        None
                        if RAM_BUILDER_WAIT_TIMEOUT_SECONDS == 0
                        else RAM_BUILDER_WAIT_TIMEOUT_SECONDS
                    ),
                )
                sleep_seconds = RAM_BUILDER_WAIT_POLL_SECONDS
                if RAM_BUILDER_WAIT_TIMEOUT_SECONDS > 0:
                    sleep_seconds = min(
                        sleep_seconds,
                        RAM_BUILDER_WAIT_TIMEOUT_SECONDS - waited_seconds,
                    )
                time.sleep(sleep_seconds)
                available = system_available_ram_gib()
            status(
                "ram_builder_acquired",
                model=model_key,
                calib_seed=calib_seed,
                builder_stage=stage,
                available_ram_gib=(
                    None if available is None else round(available, 1)
                ),
            )
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def get_dataset():
    train = frozen_arrow_rows(
        "openai/gsm8k/main/train", "gsm8k-train.arrow"
    )
    test = frozen_arrow_rows(
        "openai/gsm8k/main/test", "gsm8k-test.arrow"
    )
    return train, test


def prepare_protocol(force: bool = False) -> dict:
    if LOCK_PATH.exists() and not force:
        lock = read_json(LOCK_PATH)
        if lock.get("protocol_version") != PROTOCOL_VERSION:
            raise RuntimeError(f"Existing lock uses another protocol: {LOCK_PATH}")
        return require_protocol()

    data_provenance = dataset_provenance()
    train, test = get_dataset()
    if len(test) != GSM8K_TEST_SIZE:
        raise RuntimeError(
            f"Expected {GSM8K_TEST_SIZE} GSM8K test examples, found {len(test)}"
        )
    splits = make_disjoint_screen_splits(len(train))
    for split, calibration_seed in zip(splits, SCREEN_CALIB_SEEDS):
        split["calibration_seed"] = calibration_seed
    full_test_indices = list(range(len(test)))
    causal_patch_indices = fixed_causal_patch_indices(len(test))
    lock = {
        "protocol_version": PROTOCOL_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "openai/gsm8k/main",
        "dataset_snapshot": data_provenance,
        "train_size": len(train),
        "test_size": len(test),
        "development_split": "train only",
        "screen_splits": splits,
        "screen_quantizer": {
            "method": "GPTQ-W4",
            "scheme": "group-wise asymmetric GPTQ using the split-locked WikiText calibration seed",
            "group_size": GROUP_SIZE,
            "calibration_seeds": list(SCREEN_CALIB_SEEDS),
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
        "broad_tasks": list(BROAD_TASKS),
        "generative_transfer_tasks": list(EXTRA_TASKS),
        "required_external_matched_budget_baselines": {
            "models": ["qwen05", "qwen15"],
            "methods": ["tacq", "hawq_v2"],
            "canonical_test_n": GSM8K_TEST_SIZE,
        },
        "canonical_gsm8k_evaluator": "direct 5-shot greedy, complete official test set",
        "execution": {
            "eval_batch_size_per_gpu": DEFAULT_EVAL_BATCH_SIZE,
            "format_batch_size_per_gpu": DEFAULT_FORMAT_BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "deterministic_greedy": True,
        },
        "models": {
            key: {
                "name": spec["name"],
                "display_name": spec["display_name"],
                "role": spec["role"],
            }
            for key, spec in MODEL_SPECS.items()
        },
        "model_snapshots": {
            key: model_provenance(key) for key in MODEL_SPECS
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
    current_models = {key: model_provenance(key) for key in MODEL_SPECS}
    if lock.get("model_snapshots") != current_models:
        raise RuntimeError(
            "Model snapshot manifest changed after protocol lock; rerun prepare --force "
            "before GPU work and never mix existing results from another checkpoint"
        )
    current_data = dataset_provenance()
    if lock.get("dataset_snapshot", {}).get("manifest_sha256") != current_data[
        "manifest_sha256"
    ]:
        raise RuntimeError(
            "Dataset manifest changed after protocol lock; rerun prepare --force before GPU work"
        )
    expected_execution = {
        "eval_batch_size_per_gpu": DEFAULT_EVAL_BATCH_SIZE,
        "format_batch_size_per_gpu": DEFAULT_FORMAT_BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "deterministic_greedy": True,
    }
    if lock.get("execution") != expected_execution:
        raise RuntimeError(
            "Execution batch/token settings differ from the protocol lock; "
            "do not resume into an existing result set with new settings"
        )
    return lock


def require_locked_batch(batch_size: int, *, format_control: bool = False) -> None:
    lock = require_protocol()
    key = "format_batch_size_per_gpu" if format_control else "eval_batch_size_per_gpu"
    expected = int(lock["execution"][key])
    if batch_size != expected:
        raise RuntimeError(
            f"batch_size={batch_size} differs from locked {key}={expected}; "
            "change it only before formal outputs and rerun prepare --force"
        )


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


def screen_state_variant(split_id: int) -> str:
    return f"screen_gptq_w4_split{split_id}"


def build_screen_bank(
    model_key: str, split_id: int, calib_seed: int, force: bool
) -> Path:
    """Build the calibration-specific GPTQ-W4 state used by one screen run."""
    from ptq.eval import cleanup_gpu
    from ptq.quant.gptq import quantize_model_gptq

    lock = require_protocol()
    split = lock["screen_splits"][split_id]
    if int(split["calibration_seed"]) != calib_seed:
        raise RuntimeError(
            f"screen split {split_id} is locked to calibration seed "
            f"{split['calibration_seed']}, not {calib_seed}"
        )
    variant = screen_state_variant(split_id)
    path = state_path(model_key, calib_seed, variant)
    metadata_path = state_metadata_path(model_key, calib_seed, variant)
    if path.exists() and not force:
        require_current_state_metadata(
            metadata_path, path, model_key, calib_seed, variant
        )
        return path

    with ram_builder_slot("build-screen-bank", model_key, calib_seed):
        if path.exists() and not force:
            require_current_state_metadata(
                metadata_path, path, model_key, calib_seed, variant
            )
            return path
        configure_determinism(calib_seed)
        model, tokenizer = load_model_tokenizer(model_key)
        calibration = frozen_wikitext_calibration(tokenizer, calib_seed)
        state = quantize_model_gptq(
            model, calibration, bits=4, group_size=GROUP_SIZE
        )
        save_torch_atomic(state, path)
        screened_layers = sorted(
            {
                int(row["layer"])
                for row in module_rows(model)
            }
        )
        write_json(
            metadata_path,
            {
                "protocol_version": PROTOCOL_VERSION,
                "model_key": model_key,
                "variant": variant,
                "split_id": split_id,
                "screen_seed": int(split["seed"]),
                "calibration_seed": calib_seed,
                "quantizer": "GPTQ-W4",
                "group_size": GROUP_SIZE,
                "calibration_samples": CALIB_SAMPLES,
                "calibration_length": CALIB_LENGTH,
                "screened_layers": screened_layers,
                "state_entries": len(state),
                "bytes": path.stat().st_size,
                "model_snapshot": model_provenance(model_key),
                "dataset_snapshot": dataset_provenance(),
            },
        )
        del state, calibration, model, tokenizer
        cleanup_gpu()
        status(
            "screen_bank_saved",
            model=model_key,
            split_id=split_id,
            calib_seed=calib_seed,
            path=str(path),
        )
    return path


def _screen_records(path: Path) -> dict:
    records = {}
    for row in read_jsonl(path):
        key = (row["type"], row.get("layer"))
        if key in records:
            raise RuntimeError(f"Duplicate screen row {key} in {path}")
        records[key] = row
    return records


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
    from ptq.quant.gptq import dequantize_gptq

    require_locked_batch(batch_size)
    if max_new_tokens != MAX_NEW_TOKENS:
        raise RuntimeError(
            f"screen max_new_tokens={max_new_tokens} differs from locked {MAX_NEW_TOKENS}"
        )
    lock = require_protocol()
    split = lock["screen_splits"][split_id]
    calib_seed = int(split["calibration_seed"])
    variant = screen_state_variant(split_id)
    quantized_path = state_path(model_key, calib_seed, variant)
    if not quantized_path.exists():
        raise FileNotFoundError(
            f"Run build-screen-bank for {model_key}/split {split_id} first: {quantized_path}"
        )
    require_current_state_metadata(
        state_metadata_path(model_key, calib_seed, variant),
        quantized_path,
        model_key,
        calib_seed,
        variant,
    )
    quantized_state = torch.load(
        quantized_path, map_location="cpu", weights_only=False, mmap=True
    )
    path = screen_file(model_key, split_id)
    if force and path.exists():
        path.unlink()
    records = _screen_records(path)
    screen_identity = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "split_id": split_id,
        "split_seed": int(split["seed"]),
        "split_indices_sha256": split["indices_sha256"],
        "calibration_seed": calib_seed,
        "dataset_manifest_sha256": dataset_provenance()["manifest_sha256"],
        "model_revision": model_provenance(model_key)["resolved_revision"],
        "eval_batch_size_per_gpu": batch_size,
        "max_new_tokens": max_new_tokens,
        "quantizer": "GPTQ-W4",
        "group_size": GROUP_SIZE,
    }
    if any(
        any(row.get(key) != value for key, value in screen_identity.items())
        for row in records.values()
    ):
        raise RuntimeError(
            f"Existing screen rows have stale model/data/batch provenance: {path}. "
            "Inspect them or rerun this split with --force."
        )
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
                **screen_identity,
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
            for name, module in specs:
                if name not in quantized_state:
                    raise RuntimeError(f"Screen GPTQ state lacks module {name}")
                entry = quantized_state[name]
                w_q = entry["w_q"].to(module.weight.device)
                scale = entry["scale"].to(module.weight.device)
                zero = entry["zero"].to(module.weight.device)
                dequantized = dequantize_gptq(w_q, scale, zero, GROUP_SIZE)
                module.weight.data.copy_(dequantized.to(module.weight.dtype))
                del w_q, scale, zero, dequantized
            result = _eval_loaded_model(
                model_key, model, tokenizer, examples, batch_size, max_new_tokens
            )
            append_jsonl(
                path,
                {
                    "type": "layer",
                    **screen_identity,
                    "layer": layer,
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

    del model, tokenizer, quantized_state
    cleanup_gpu()
    status("screen_complete", model=model_key, split_id=split_id, path=str(path))


def select_model(model_key: str) -> dict:
    lock = require_protocol()
    path = OUT / "selections" / f"{model_key}.json"
    if path.exists():
        # Allocation ids are persistent result identities, not disposable draws.
        # In particular, upgrading a sampler must not relabel completed evidence.
        selection = selection_for(model_key)
        current_hashes = {
            str(split["split_id"]): json_sha256(
                read_jsonl(screen_file(model_key, split["split_id"]))
            )
            for split in lock["screen_splits"]
        }
        if (
            selection.get("screen_file_sha256") != current_hashes
            or selection.get("screen_calibration_seeds")
            != [int(split["calibration_seed"]) for split in lock["screen_splits"]]
            or selection.get("screen_quantizer") != lock["screen_quantizer"]
        ):
            raise RuntimeError(
                f"Locked selection screen evidence has changed: {path}. "
                "Preserve existing results and inspect the mismatch; "
                "do not overwrite the allocation manifest to resume."
            )
        status("selection_already_locked", model=model_key, path=str(path))
        return selection

    split_rows: dict[int, list[dict]] = {}
    for split in lock["screen_splits"]:
        path = screen_file(model_key, split["split_id"])
        all_rows = read_jsonl(path)
        baselines = [row for row in all_rows if row.get("type") == "baseline"]
        rows = [row for row in all_rows if row.get("type") == "layer"]
        metadata = read_json(
            state_metadata_path(
                model_key,
                int(split["calibration_seed"]),
                screen_state_variant(int(split["split_id"])),
            )
        )
        current_dataset = dataset_provenance()
        current_model = model_provenance(model_key)
        if (
            metadata.get("protocol_version") != PROTOCOL_VERSION
            or metadata.get("dataset_snapshot") != current_dataset
            or metadata.get("model_snapshot") != current_model
            or int(metadata.get("split_id", -1)) != int(split["split_id"])
            or int(metadata.get("calibration_seed", -1))
            != int(split["calibration_seed"])
        ):
            raise RuntimeError(f"Screen-state metadata is stale: {path}")
        expected_layers = {int(value) for value in metadata.get("screened_layers", [])}
        observed_layers = [int(row["layer"]) for row in rows]
        valid_common = lambda row: (
            row.get("protocol_version") == PROTOCOL_VERSION
            and row.get("model_key") == model_key
            and int(row.get("split_id", -1)) == int(split["split_id"])
            and int(row.get("split_seed", -1)) == int(split["seed"])
            and row.get("split_indices_sha256") == split["indices_sha256"]
            and int(row.get("calibration_seed", -1))
            == int(split["calibration_seed"])
            and row.get("dataset_manifest_sha256")
            == current_dataset["manifest_sha256"]
            and row.get("model_revision") == current_model["resolved_revision"]
            and int(row.get("eval_batch_size_per_gpu", -1))
            == DEFAULT_EVAL_BATCH_SIZE
            and int(row.get("max_new_tokens", -1)) == MAX_NEW_TOKENS
            and row.get("quantizer") == "GPTQ-W4"
            and int(row.get("group_size", -1)) == GROUP_SIZE
        )
        if (
            len(baselines) != 1
            or not expected_layers
            or len(observed_layers) != len(set(observed_layers))
            or set(observed_layers) != expected_layers
            or len(all_rows) != len(rows) + 1
            or not valid_common(baselines[0])
            or any(not valid_common(row) for row in rows)
        ):
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
    random_layer_plan = make_random_layer_allocation_plan(
        modules,
        set(selection["w8_module_names"]),
        selection["selected_layers"],
    )
    random_layer_allocations = random_layer_plan["sets"]
    random_module_allocations = make_unique_random_module_allocations(
        modules, set(selection["w8_module_names"])
    )
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
        "screen_calibration_seeds": [
            int(split["calibration_seed"]) for split in lock["screen_splits"]
        ],
        "screen_quantizer": lock["screen_quantizer"],
        "model_snapshot": model_provenance(model_key),
        "dataset_snapshot": dataset_provenance(),
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
        "random_allocation_manifest": {
            "requested_count_per_family": RANDOM_ALLOCATIONS,
            "count_per_family": RANDOM_ALLOCATIONS,
            "layer_seed": 20261001,
            "module_seed": 20262001,
            "layer_feasibility": {
                key: value
                for key, value in random_layer_plan.items()
                if key != "sets"
            },
            "layer_sets": random_layer_allocations,
            "module_sets": random_module_allocations,
            "layer_sets_sha256": json_sha256(random_layer_allocations),
            "module_sets_sha256": json_sha256(random_module_allocations),
        },
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
    if selection.get("model_key") != model_key:
        raise RuntimeError(f"Selection belongs to a different model: {path}")
    if selection.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"Selection protocol is stale: {path}")
    if selection.get("test_data_used") is not False:
        raise RuntimeError(f"Selection is not test-clean: {path}")
    if selection.get("dataset_snapshot") != dataset_provenance():
        raise RuntimeError(f"Selection dataset provenance has changed: {path}")
    if selection.get("model_snapshot") != model_provenance(model_key):
        raise RuntimeError(f"Selection model provenance has changed: {path}")
    manifest = selection.get("random_allocation_manifest", {})
    try:
        validate_random_allocation_manifest(manifest)
    except (TypeError, ValueError):
        raise RuntimeError(f"Selection random-allocation manifest is invalid: {path}")
    return selection


def allocation_ids(model_key: str, family: str) -> list[int]:
    """Return only the preregistered allocation ids for a model/family."""
    if family not in {"layer", "module"}:
        raise ValueError("family must be layer or module")
    selection = selection_for(model_key)
    counts = validate_random_allocation_manifest(
        selection["random_allocation_manifest"]
    )
    return list(range(counts[family]))


def save_torch_atomic(value, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_bank(model_key: str, calib_seed: int, force: bool) -> Path:
    from ptq.eval import cleanup_gpu
    from ptq.quant.mixed_precision import quantize_model_precision_bank

    require_protocol()
    path = state_path(model_key, calib_seed, "precision_bank")
    metadata_path = state_metadata_path(model_key, calib_seed, "precision_bank")
    if path.exists() and not force:
        require_current_state_metadata(
            metadata_path, path, model_key, calib_seed, "precision_bank"
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
    with ram_builder_slot("build-bank", model_key, calib_seed):
        if path.exists() and not force:
            require_current_state_metadata(
                metadata_path, path, model_key, calib_seed, "precision_bank"
            )
            return path
        configure_determinism(calib_seed)
        model, tokenizer = load_model_tokenizer(model_key)
        calibration = frozen_wikitext_calibration(tokenizer, calib_seed)
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
                "variant": "precision_bank",
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
                "model_snapshot": model_provenance(model_key),
                "dataset_snapshot": dataset_provenance(),
            },
        )
        del bank, calibration, model, tokenizer
        cleanup_gpu()
        status(
            "precision_bank_saved",
            model=model_key,
            calib_seed=calib_seed,
            path=str(path),
        )
    return path


def _random_module_allocation(selection: dict, allocation_id: int) -> set[str]:
    manifest = selection.get("random_allocation_manifest", {})
    allocations = manifest.get("module_sets", [])
    counts = validate_random_allocation_manifest(manifest)
    if not 0 <= allocation_id < counts["module"]:
        raise ValueError(
            f"Random module allocation id must be in [0, {counts['module'] - 1}]"
        )
    return set(allocations[allocation_id])


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
        manifest = selection.get("random_allocation_manifest", {})
        allocations = manifest.get("layer_sets", [])
        counts = validate_random_allocation_manifest(manifest)
        if not 0 <= allocation_id < counts["layer"]:
            raise ValueError(
                f"Random layer allocation id must be in [0, {counts['layer'] - 1}]"
            )
        selected = {int(layer) for layer in allocations[allocation_id]}

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
        int(variant.rsplit("_", 1)[1])
    elif variant.startswith("random_"):
        int(variant.split("_", 1)[1])
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
        require_current_state_metadata(
            metadata_path, output_path, model_key, calib_seed, variant
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
    require_current_state_metadata(
        state_metadata_path(model_key, calib_seed, "precision_bank"),
        bank_path,
        model_key,
        calib_seed,
        "precision_bank",
    )
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
            "model_snapshot": model_provenance(model_key),
            "dataset_snapshot": dataset_provenance(),
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
        require_current_state_metadata(
            metadata_path, path, model_key, calib_seed, variant
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
    require_current_state_metadata(
        state_metadata_path(model_key, calib_seed, "precision_bank"),
        bank_path,
        model_key,
        calib_seed,
        "precision_bank",
    )
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
            "model_snapshot": model_provenance(model_key),
            "dataset_snapshot": dataset_provenance(),
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

    # Core v4 rows retain their historical generation contract.  The
    # shadow-validated online stop is enabled only by the TaCQ adapter.
    direct.ONLINE_QUESTION_STOP = False
    method = method_id(variant, calib_seed)
    spec = MODEL_SPECS[model_key]
    direct.OUT = RESULTS_DIR / "runtime" / model_key
    direct.SAMPLE_DIR = RESULTS_DIR / "samples"
    direct.LOG_DIR = RESULTS_DIR / "logs"
    direct.get_dataset = get_dataset
    for path in [direct.OUT, direct.SAMPLE_DIR, direct.LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    direct.ROW_METADATA = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_manifest_sha256": dataset_provenance()["manifest_sha256"],
        "model_revision": model_provenance(model_key)["resolved_revision"],
        "canonical_test_set": "openai/gsm8k/main:test:all-1319",
    }
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
        require_current_state_metadata(
            state_metadata_path(model_key, calib_seed, variant),
            path,
            model_key,
            calib_seed,
            variant,
        )
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
    require_locked_batch(batch_size)
    if max_new_tokens != MAX_NEW_TOKENS:
        raise RuntimeError(
            f"max_new_tokens={max_new_tokens} differs from locked {MAX_NEW_TOKENS}"
        )
    if not force and gsm8k_complete(model_key, variant, calib_seed):
        completed_path = gsm8k_sample_path(model_key, variant, calib_seed)
        expected_row_metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "dataset_manifest_sha256": dataset_provenance()["manifest_sha256"],
            "model_revision": model_provenance(model_key)["resolved_revision"],
            "canonical_test_set": "openai/gsm8k/main:test:all-1319",
            "eval_batch_size_per_gpu": batch_size,
            "max_new_tokens": max_new_tokens,
        }
        if any(
            any(row.get(key) != value for key, value in expected_row_metadata.items())
            for row in read_jsonl(completed_path)
        ):
            raise RuntimeError(
                f"Completed canonical result has stale provenance: {completed_path}"
            )
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
    if completed:
        evaluate_full(
            model_key,
            variant,
            calib_seed,
            batch_size,
            max_new_tokens=MAX_NEW_TOKENS,
            force=False,
        )
        if keep_state:
            allocation_state = state_path(model_key, calib_seed, variant)
            materialize(
                model_key,
                calib_seed,
                variant,
                force=not allocation_state.exists(),
            )
        else:
            cleanup_state_artifact(model_key, calib_seed, variant)
        return
    materialize(model_key, calib_seed, variant, force=False)
    evaluate_full(
        model_key,
        variant,
        calib_seed,
        batch_size,
        max_new_tokens=MAX_NEW_TOKENS,
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

    require_locked_batch(batch_size)
    expected_method = method_id(variant, calib_seed)
    expected_path = RESULTS_DIR / "broad" / f"{model_key}__{expected_method}.json"
    expected_metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "method": expected_method,
        "dataset_snapshot": dataset_provenance(),
        "model_snapshot": model_provenance(model_key),
        "batch_size_per_gpu": batch_size,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tasks": list(BROAD_TASKS),
    }
    if not force and broad_complete(model_key, variant, calib_seed):
        record = read_json(expected_path)
        if any(record.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(
                f"Completed broad result has stale provenance: {expected_path}"
            )
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
    tasks = list(BROAD_TASKS)
    locked_metadata = expected_metadata
    record = read_json(path) if path.exists() else {
        **locked_metadata,
        "canonical_gsm8k_source": "evaluate-full only; GSM8K intentionally excluded here",
        "limits": {task: None for task in tasks},
        "scores": {},
    }
    if any(record.get(key) != value for key, value in locked_metadata.items()):
        raise RuntimeError(
            f"Broad result metadata changed; inspect or rerun with --force: {path}"
        )
    for task in tasks:
        if task in record["scores"]:
            continue
        score = run_eval_on_model(
            model,
            tokenizer,
            [task],
            batch_size=batch_size,
            max_gen_toks=MAX_NEW_TOKENS,
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
    from ptq.eval import cleanup_gpu

    from experiments.revision_full.lm_eval_compat import RevisionTaskManager

    require_locked_batch(batch_size)
    expected_method = method_id(variant, calib_seed)
    expected_path = RESULTS_DIR / "extra" / f"{model_key}__{expected_method}.json"
    expected_metadata = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "method": expected_method,
        "dataset_snapshot": dataset_provenance(),
        "model_snapshot": model_provenance(model_key),
        "batch_size_per_gpu": batch_size,
        "max_new_tokens": MAX_NEW_TOKENS,
        "tasks": list(EXTRA_TASKS),
    }
    if not force and extra_complete(model_key, variant, calib_seed):
        record = read_json(expected_path)
        if any(record.get(key) != value for key, value in expected_metadata.items()):
            raise RuntimeError(
                f"Completed extra-panel result has stale provenance: {expected_path}"
            )
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
    tasks = list(EXTRA_TASKS)
    task_manager = RevisionTaskManager(
        include_path=str(OUT.parents[1] / "fix_svamp_ood")
    )
    path = RESULTS_DIR / "extra" / f"{model_key}__{method}.json"
    if force and path.exists():
        path.unlink()
    locked_metadata = expected_metadata
    record = read_json(path) if path.exists() else {
        **locked_metadata,
        "limits": {task: None for task in tasks},
        "results": {},
    }
    if any(record.get(key) != value for key, value in locked_metadata.items()):
        raise RuntimeError(
            f"Extra-panel metadata changed; inspect or rerun with --force: {path}"
        )
    manifest = read_json(DATASET_MANIFEST_PATH)
    expected_docs = {
        task: int(
            manifest.get("panels", {})
            .get("tasks", {})
            .get(task, {})
            .get("evaluation_docs", -1)
        )
        for task in tasks
    }
    for task in tasks:
        existing = record["results"].get(task)
        if existing is not None and extra_task_complete(existing, expected_docs[task]):
            continue
        if existing is not None:
            status(
                "extra_task_incomplete_rerun",
                model=model_key,
                method=method,
                task=task,
            )
        result = simple_evaluate(
            model=lm_model,
            tasks=[task],
            task_manager=task_manager,
            batch_size=batch_size,
            limit=None,
            log_samples=True,
            gen_kwargs={
                "temperature": 0.0,
                "max_new_tokens": MAX_NEW_TOKENS,
                "do_sample": False,
            },
        )
        samples = result.get("samples", {}).get(task, [])
        record["results"][task] = {
            "metrics": result.get("results", {}).get(task, {}),
            "n_samples": len(samples),
            "samples": samples,
        }
        write_json(path, record, default=str)
        torch.cuda.empty_cache()
    del lm_model, model, tokenizer
    cleanup_gpu()
    status("extra_full_complete", model=model_key, method=method, path=str(path))


def parse_model(value: str) -> str:
    if value not in MODEL_SPECS:
        raise ValueError(f"Unknown model {value}; choose from {list(MODEL_SPECS)}")
    return value


def smoke_eval(
    model_key: str,
    batch_size: int,
    format_batch_size: int,
    max_new_tokens: int,
) -> dict:
    """Use GSM8K train only to validate 24-GiB evaluation memory before locking."""
    import torch
    from experiments.fix_gsm8k_500.direct_eval import gold_answer
    from experiments.revision_full.format_control import (
        chat_prompt,
        make_item,
        raw_prompt,
        score_choice_batch,
    )
    from ptq.eval import cleanup_gpu

    if batch_size <= 0 or format_batch_size <= 0 or max_new_tokens <= 0:
        raise ValueError("smoke-test batch sizes and max_new_tokens must be positive")
    dataset_provenance()
    model_provenance(model_key)
    train, _ = get_dataset()
    generation_examples = [train[index] for index in range(5, 5 + batch_size)]
    configure_determinism(20260901)
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_model_tokenizer(model_key)
    generation = _eval_loaded_model(
        model_key,
        model,
        tokenizer,
        generation_examples,
        batch_size,
        max_new_tokens,
    )

    train_answers = [gold_answer(row["answer"]) for row in train[5:]]
    demos = [
        make_item(row["question"], row["answer"], train_answers, 20260831 + index)
        for index, row in enumerate(train[:5])
    ]
    targets = [
        make_item(
            train[index]["question"],
            train[index]["answer"],
            train_answers,
            20261831 + index,
        )
        for index in range(5, 5 + format_batch_size)
    ]
    prompts = [
        chat_prompt(tokenizer, demos, item)
        if MODEL_SPECS[model_key]["prompt_style"] == "chat"
        else raw_prompt(demos, item)
        for item in targets
    ]
    scores = score_choice_batch(model, tokenizer, prompts)
    if len(scores) != format_batch_size or any(len(row) != 4 for row in scores):
        raise RuntimeError("format-control smoke test did not return four scores per item")
    peak_gib = torch.cuda.max_memory_allocated() / 1024**3
    result = {
        "model_key": model_key,
        "source_split": "GSM8K train only",
        "generation_batch_size": batch_size,
        "format_batch_size": format_batch_size,
        "max_new_tokens": max_new_tokens,
        "generation_items": generation["n"],
        "peak_cuda_allocated_gib": round(peak_gib, 3),
        "status": "passed",
    }
    del model, tokenizer
    cleanup_gpu()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("smoke-eval")
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    p.add_argument("--format-batch-size", type=int, default=DEFAULT_FORMAT_BATCH_SIZE)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)

    p = sub.add_parser("build-screen-bank")
    p.add_argument("--model", required=True)
    p.add_argument("--split-id", type=int, choices=range(len(SCREEN_SEEDS)), required=True)
    p.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("cleanup-screen-bank")
    p.add_argument("--model", required=True)
    p.add_argument("--split-id", type=int, choices=range(len(SCREEN_SEEDS)), required=True)
    p.add_argument("--calib-seed", type=int, choices=CALIB_SEEDS, required=True)

    p = sub.add_parser("screen")
    p.add_argument("--model", required=True)
    p.add_argument("--split-id", type=int, choices=range(len(SCREEN_SEEDS)), required=True)
    p.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("select")
    p.add_argument("--model", required=True)

    p = sub.add_parser("allocation-ids")
    p.add_argument("--model", required=True)
    p.add_argument("--family", choices=["layer", "module"], required=True)

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
    if args.command == "smoke-eval":
        smoke_eval(
            model_key,
            args.batch_size,
            args.format_batch_size,
            args.max_new_tokens,
        )
    elif args.command == "build-screen-bank":
        print(
            build_screen_bank(
                model_key, args.split_id, args.calib_seed, args.force
            )
        )
    elif args.command == "cleanup-screen-bank":
        cleanup_screen_state_artifact(model_key, args.split_id, args.calib_seed)
    elif args.command == "screen":
        screen_model(
            model_key,
            args.split_id,
            args.batch_size,
            args.max_new_tokens,
            args.force,
        )
    elif args.command == "select":
        selection = select_model(model_key)
        allocation_counts = validate_random_allocation_manifest(
            selection["random_allocation_manifest"]
        )
        print(
            json.dumps(
                {
                    "model": selection["model"],
                    "selected_layers": selection["selected_layers"],
                    "actual_avg_bits": selection["actual_avg_bits"],
                    "mean_top_set_jaccard": selection["mean_top_set_jaccard"],
                    "random_allocation_counts": allocation_counts,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "allocation-ids":
        print(" ".join(str(value) for value in allocation_ids(model_key, args.family)))
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
            MAX_NEW_TOKENS,
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

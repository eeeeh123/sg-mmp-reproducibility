"""Frozen protocol constants and CPU-only helpers for the rejection revision."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "revision_full" / "outputs"


def _configured_state_dir() -> Path:
    value = os.environ.get("REVISION_FULL_STATE_DIR")
    path = Path(value).expanduser() if value else OUT / "states"
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


STATE_DIR = _configured_state_dir()
STATE_METADATA_DIR = OUT / "state_metadata"
SCREEN_DIR = OUT / "screens"
RESULTS_DIR = OUT / "results"

PROTOCOL_VERSION = "revision-full-v4"
GSM8K_TEST_SIZE = 1319
CAUSAL_PATCH_N = 200
CAUSAL_PATCH_SEED = 20268001
SCREEN_N = 256
SCREEN_SEEDS = (20260831, 20260901, 20260902)
SELECTION_BOOTSTRAP_REPLICATES = 2000
SELECTION_BOOTSTRAP_SEED = 20269001
CALIB_SEEDS = (41, 97, 193)
SCREEN_CALIB_SEEDS = CALIB_SEEDS
CALIB_SAMPLES = 128
CALIB_LENGTH = 2048
CALIB_HESSIAN_TOKENS = 4096
TARGET_AVG_BITS = 5.0
RANDOM_ALLOCATIONS = 30
RANDOM_CALIB_SEED = CALIB_SEEDS[0]
GROUP_SIZE = 128
DEFAULT_EVAL_BATCH_SIZE = int(os.environ.get("REVISION_FULL_EVAL_BATCH_SIZE", "4"))
DEFAULT_FORMAT_BATCH_SIZE = int(
    os.environ.get("REVISION_FULL_FORMAT_BATCH_SIZE", "2")
)
MAX_CONCURRENT_RAM_BUILDERS = int(
    os.environ.get("REVISION_FULL_MAX_CONCURRENT_RAM_BUILDERS", "1")
)
MIN_AVAILABLE_RAM_GIB = float(
    os.environ.get("REVISION_FULL_MIN_AVAILABLE_RAM_GIB", "24")
)
MAX_NEW_TOKENS = 256
FEWSHOT_TRAIN_INDICES = tuple(range(5))

if DEFAULT_EVAL_BATCH_SIZE <= 0 or DEFAULT_FORMAT_BATCH_SIZE <= 0:
    raise ValueError("Evaluation batch sizes must be positive")
if MAX_CONCURRENT_RAM_BUILDERS <= 0:
    raise ValueError("REVISION_FULL_MAX_CONCURRENT_RAM_BUILDERS must be positive")
if MIN_AVAILABLE_RAM_GIB < 0:
    raise ValueError("REVISION_FULL_MIN_AVAILABLE_RAM_GIB cannot be negative")

BROAD_TASKS = (
    "arc_challenge",
    "hellaswag",
    "mmlu",
    "mmlu_high_school_mathematics",
)
EXTRA_TASKS = ("svamp", "asdiv_gen", "hendrycks_math500", "truthfulqa_gen")

MODEL_SPECS = {
    "qwen05": {
        "name": "Qwen2.5-0.5B",
        "display_name": "Qwen2.5-0.5B",
        "path": ROOT / "models" / "Qwen2.5-0.5B",
        "prompt_style": "raw",
        "role": "primary",
    },
    "qwen15": {
        "name": "Qwen2.5-1.5B",
        "display_name": "Qwen2.5-1.5B",
        "path": ROOT / "models" / "Qwen2.5-1.5B",
        "prompt_style": "raw",
        "role": "primary",
    },
    "smollm": {
        "name": "SmolLM-1.7B",
        "display_name": "SmolLM2-1.7B",
        "path": ROOT / "models" / "SmolLM-1.7B",
        "prompt_style": "raw",
        "role": "primary",
    },
    "gemma2": {
        "name": "gemma-2-2b-it",
        "display_name": "Gemma-2-2B-it",
        "path": ROOT / "models" / "gemma-2-2b-it",
        "prompt_style": "chat",
        "role": "family_check",
    },
}

ELIGIBLE_SHORT_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
QKV_SHORT_NAMES = {"q_proj", "k_proj", "v_proj"}
ROLE_SHORT_NAMES = {
    "qkv": QKV_SHORT_NAMES,
    "o": {"o_proj"},
    "ffn": {"gate_proj", "up_proj", "down_proj"},
}


def json_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_unique_random_layer_allocations(
    module_rows: list[dict],
    target_w8_names: set[str],
    selected_layers: list[int],
    count: int = RANDOM_ALLOCATIONS,
    seed: int = 20261001,
) -> list[list[int]]:
    """Pre-register distinct layer allocations at the SG-MMP W8 budget."""
    qkv_names = {
        row["name"] for row in module_rows if row["short"] in QKV_SHORT_NAMES
    }
    target_bits = average_bits(module_rows, target_w8_names)
    all_layers = sorted({int(row["layer"]) for row in module_rows})
    layer_count = len(selected_layers)
    if not 0 < layer_count <= len(all_layers):
        raise ValueError("selected_layers must define a non-empty feasible allocation")
    rng = random.Random(seed)
    allocations: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    attempts = 0
    while len(allocations) < count and attempts < 200_000:
        attempts += 1
        candidate = tuple(sorted(rng.sample(all_layers, layer_count)))
        if candidate in seen:
            continue
        names = qkv_names | {
            row["name"]
            for row in module_rows
            if int(row["layer"]) in candidate
            and row["short"] not in QKV_SHORT_NAMES
        }
        if abs(average_bits(module_rows, names) - target_bits) > 0.01:
            continue
        seen.add(candidate)
        allocations.append(list(candidate))
    if len(allocations) != count:
        raise RuntimeError(
            f"Could only construct {len(allocations)}/{count} distinct matched layer allocations"
        )
    return allocations


def make_unique_random_module_allocations(
    module_rows: list[dict],
    target_w8_names: set[str],
    count: int = RANDOM_ALLOCATIONS,
    seed: int = 20262001,
) -> list[list[str]]:
    """Pre-register distinct module allocations within 0.01 average bit."""
    target_bits = average_bits(module_rows, target_w8_names)
    target_params = sum(
        int(row["n_params"])
        for row in module_rows
        if row["name"] in target_w8_names
    )
    rng = random.Random(seed)
    allocations: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(allocations) < count and attempts < 500_000:
        attempts += 1
        candidates = list(module_rows)
        rng.shuffle(candidates)
        chosen: set[str] = set()
        used = 0
        for row in candidates:
            size = int(row["n_params"])
            if used + size <= target_params:
                chosen.add(row["name"])
                used += size
        key = tuple(sorted(chosen))
        if not key or key in seen:
            continue
        if abs(average_bits(module_rows, chosen) - target_bits) > 0.01:
            continue
        seen.add(key)
        allocations.append(list(key))
    if len(allocations) != count:
        raise RuntimeError(
            f"Could only construct {len(allocations)}/{count} distinct matched module allocations"
        )
    return allocations


def make_disjoint_screen_splits(
    train_size: int,
    n: int = SCREEN_N,
    seeds: tuple[int, ...] = SCREEN_SEEDS,
) -> list[dict]:
    """Create deterministic, non-overlapping development splits from GSM8K train."""
    reserved = set(FEWSHOT_TRAIN_INDICES)
    remaining = [i for i in range(train_size) if i not in reserved]
    if n * len(seeds) > len(remaining):
        raise ValueError(
            f"Need {n * len(seeds)} development examples but only {len(remaining)} are available"
        )

    splits = []
    for split_id, seed in enumerate(seeds):
        rng = random.Random(seed)
        rng.shuffle(remaining)
        indices = sorted(remaining[:n])
        used = set(indices)
        remaining = [i for i in remaining if i not in used]
        splits.append(
            {
                "split_id": split_id,
                "seed": seed,
                "n": n,
                "indices": indices,
                "indices_sha256": json_sha256(indices),
            }
        )
    return splits


def fixed_causal_patch_indices(
    test_size: int = GSM8K_TEST_SIZE,
    n: int = CAUSAL_PATCH_N,
    seed: int = CAUSAL_PATCH_SEED,
) -> list[int]:
    """Pre-register a model-output-independent diagnostic test subset."""
    if not 1 <= n <= test_size:
        raise ValueError(f"causal patch n must be in [1, {test_size}], got {n}")
    indices = list(range(test_size))
    random.Random(seed).shuffle(indices)
    return sorted(indices[:n])


def average_bits(module_rows: list[dict], w8_names: set[str]) -> float:
    total = sum(int(row["n_params"]) for row in module_rows)
    if total <= 0:
        raise ValueError("No eligible quantized parameters")
    w8 = sum(int(row["n_params"]) for row in module_rows if row["name"] in w8_names)
    return (8 * w8 + 4 * (total - w8)) / total


def _best_subset_under_budget(
    module_rows: list[dict], budget_params: int, seed: int, attempts: int = 5000
) -> set[str]:
    """Deterministically approximate the largest module subset under a budget."""
    if budget_params <= 0 or not module_rows:
        return set()
    best: set[str] = set()
    best_total = 0
    rng = random.Random(seed)
    for _ in range(attempts):
        candidates = list(module_rows)
        rng.shuffle(candidates)
        chosen: set[str] = set()
        total = 0
        for row in candidates:
            size = int(row["n_params"])
            if total + size <= budget_params:
                chosen.add(row["name"])
                total += size
        if total > best_total:
            best = chosen
            best_total = total
            if total == budget_params:
                break
    return best


def role_priority_budget_match(
    module_rows: list[dict],
    target_w8_names: set[str],
    preferred_shorts: set[str],
    seed: int,
) -> dict:
    """Match the SG W8 budget while prioritizing one module family.

    Pure q/k/v-only or o-only policies often cannot consume the SG budget. This
    control therefore maximizes W8 parameters from the requested family first,
    then uses explicitly reported complementary filler modules to close the
    budget. It is a matched-budget role-priority control, not a pure-family one.
    """
    target_params = sum(
        int(row["n_params"]) for row in module_rows if row["name"] in target_w8_names
    )
    preferred = [row for row in module_rows if row["short"] in preferred_shorts]
    filler = [row for row in module_rows if row["short"] not in preferred_shorts]

    chosen_preferred = _best_subset_under_budget(preferred, target_params, seed)
    preferred_params = sum(
        int(row["n_params"]) for row in preferred if row["name"] in chosen_preferred
    )
    remaining = target_params - preferred_params
    chosen_filler = _best_subset_under_budget(filler, remaining, seed + 1)
    selected = chosen_preferred | chosen_filler
    actual_params = sum(
        int(row["n_params"]) for row in module_rows if row["name"] in selected
    )
    return {
        "selected_module_names": sorted(selected),
        "preferred_module_names": sorted(chosen_preferred),
        "filler_module_names": sorted(chosen_filler),
        "target_w8_params": target_params,
        "actual_w8_params": actual_params,
        "gap_params": target_params - actual_params,
    }


def scored_budget_match(
    module_rows: list[dict], target_w8_names: set[str], scores: dict[str, float]
) -> dict:
    """Greedily allocate W8 by score-per-parameter under the SG budget."""
    target_params = sum(
        int(row["n_params"]) for row in module_rows if row["name"] in target_w8_names
    )
    ordered = sorted(
        module_rows,
        key=lambda row: (
            -float(scores.get(row["name"], 0.0)) / max(1, int(row["n_params"])),
            row["name"],
        ),
    )
    selected: set[str] = set()
    used = 0
    for row in ordered:
        size = int(row["n_params"])
        if used + size <= target_params:
            selected.add(row["name"])
            used += size
    remaining_rows = [row for row in module_rows if row["name"] not in selected]
    filler = _best_subset_under_budget(
        remaining_rows, target_params - used, seed=20264001
    )
    selected |= filler
    used = sum(
        int(row["n_params"]) for row in module_rows if row["name"] in selected
    )
    return {
        "selected_module_names": sorted(selected),
        "budget_filler_module_names": sorted(filler),
        "target_w8_params": target_params,
        "actual_w8_params": used,
        "gap_params": target_params - used,
    }


def select_layers_under_budget(
    ranking: list[dict],
    module_rows: list[dict],
    target_avg_bits: float = TARGET_AVG_BITS,
) -> dict:
    """Select ranked layers without exceeding the parameter-weighted bit budget."""
    if not 4.0 <= target_avg_bits <= 8.0:
        raise ValueError(f"target_avg_bits must be in [4, 8], got {target_avg_bits}")

    qkv_names = {row["name"] for row in module_rows if row["short"] in QKV_SHORT_NAMES}
    base_bits = average_bits(module_rows, qkv_names)
    if base_bits > target_avg_bits + 1e-12:
        raise ValueError(
            f"q/k/v-only policy already uses {base_bits:.4f} bits, above target {target_avg_bits:.4f}"
        )

    selected: list[int] = []
    w8_names = set(qkv_names)
    decisions = []
    for item in ranking:
        layer = int(item["layer"])
        upgrade = {
            row["name"]
            for row in module_rows
            if int(row["layer"]) == layer and row["short"] not in QKV_SHORT_NAMES
        }
        proposed = w8_names | upgrade
        proposed_bits = average_bits(module_rows, proposed)
        accepted = proposed_bits <= target_avg_bits + 1e-12
        decisions.append(
            {
                "layer": layer,
                "accepted": accepted,
                "proposed_avg_bits": round(proposed_bits, 6),
                "incremental_params": sum(
                    int(row["n_params"]) for row in module_rows if row["name"] in upgrade
                ),
            }
        )
        if accepted:
            selected.append(layer)
            w8_names = proposed

    return {
        "selected_layers": selected,
        "actual_avg_bits": average_bits(module_rows, w8_names),
        "base_qkv_avg_bits": base_bits,
        "w8_module_names": sorted(w8_names),
        "decisions": decisions,
    }


def state_path(model_key: str, calib_seed: int, variant: str) -> Path:
    return STATE_DIR / model_key / f"calib_{calib_seed}" / f"{variant}.pt"


def state_metadata_path(model_key: str, calib_seed: int, variant: str) -> Path:
    return STATE_METADATA_DIR / model_key / f"calib_{calib_seed}" / f"{variant}.json"


def method_id(variant: str, calib_seed: int | None = None) -> str:
    return variant if calib_seed is None else f"{variant}__c{calib_seed}"

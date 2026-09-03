"""Frozen protocol constants and CPU-only helpers for the rejection revision."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
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
RANDOM_LAYER_ENUMERATION_LIMIT = 1_000_000
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
RAM_BUILDER_WAIT_POLL_SECONDS = float(
    os.environ.get("REVISION_FULL_RAM_BUILDER_WAIT_POLL_SECONDS", "30")
)
RAM_BUILDER_WAIT_TIMEOUT_SECONDS = float(
    os.environ.get("REVISION_FULL_RAM_BUILDER_WAIT_TIMEOUT_SECONDS", "0")
)
MAX_NEW_TOKENS = 256
FEWSHOT_TRAIN_INDICES = tuple(range(5))

if DEFAULT_EVAL_BATCH_SIZE <= 0 or DEFAULT_FORMAT_BATCH_SIZE <= 0:
    raise ValueError("Evaluation batch sizes must be positive")
if MAX_CONCURRENT_RAM_BUILDERS <= 0:
    raise ValueError("REVISION_FULL_MAX_CONCURRENT_RAM_BUILDERS must be positive")
if MIN_AVAILABLE_RAM_GIB < 0:
    raise ValueError("REVISION_FULL_MIN_AVAILABLE_RAM_GIB cannot be negative")
if RAM_BUILDER_WAIT_POLL_SECONDS <= 0:
    raise ValueError("REVISION_FULL_RAM_BUILDER_WAIT_POLL_SECONDS must be positive")
if RAM_BUILDER_WAIT_TIMEOUT_SECONDS < 0:
    raise ValueError("REVISION_FULL_RAM_BUILDER_WAIT_TIMEOUT_SECONDS cannot be negative")
if MAX_CONCURRENT_RAM_BUILDERS == 1 and MIN_AVAILABLE_RAM_GIB < 24:
    raise ValueError(
        "single RAM-builder mode requires REVISION_FULL_MIN_AVAILABLE_RAM_GIB >= 24"
    )

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


def make_random_layer_allocation_plan(
    module_rows: list[dict],
    target_w8_names: set[str],
    selected_layers: list[int],
    count: int = RANDOM_ALLOCATIONS,
    seed: int = 20261001,
) -> dict:
    """Pre-register up to ``count`` feasible, distinct layer allocations.

    When fewer than ``count`` whole-layer placements exist, keep every
    feasible placement and record the exhaustive shortfall. Duplicating a
    placement would inflate the apparent random-control sample size.
    """
    qkv_names = {
        row["name"] for row in module_rows if row["short"] in QKV_SHORT_NAMES
    }
    target_params = sum(
        int(row["n_params"])
        for row in module_rows
        if row["name"] in target_w8_names
    )
    all_layers = sorted({int(row["layer"]) for row in module_rows})
    layer_count = len(selected_layers)
    if not 0 <= layer_count <= len(all_layers):
        raise ValueError("selected_layers must define a feasible allocation")
    if count <= 0:
        raise ValueError("count must be positive")

    def is_matched(candidate: tuple[int, ...]) -> bool:
        names = qkv_names | {
            row["name"]
            for row in module_rows
            if int(row["layer"]) in candidate
            and row["short"] not in QKV_SHORT_NAMES
        }
        actual_params = sum(
            int(row["n_params"])
            for row in module_rows
            if row["name"] in names
        )
        return actual_params == target_params

    rng = random.Random(seed)
    total_candidate_sets = math.comb(len(all_layers), layer_count)
    exhaustive = total_candidate_sets <= RANDOM_LAYER_ENUMERATION_LIMIT
    candidates: list[tuple[int, ...]] = []
    feasible_candidate_sets: int | None = None
    if exhaustive:
        feasible = [
            candidate
            for candidate in itertools.combinations(all_layers, layer_count)
            if is_matched(candidate)
        ]
        feasible_candidate_sets = len(feasible)
        rng.shuffle(feasible)
        candidates = feasible[:count]
    else:
        seen: set[tuple[int, ...]] = set()
        attempts = 0
        while len(candidates) < count and attempts < 200_000:
            attempts += 1
            candidate = tuple(sorted(rng.sample(all_layers, layer_count)))
            if candidate in seen:
                continue
            seen.add(candidate)
            if is_matched(candidate):
                candidates.append(candidate)

    if not candidates:
        raise RuntimeError("Could not construct any matched layer allocation")
    if len(candidates) < count and not exhaustive:
        raise RuntimeError(
            f"Could only construct {len(candidates)}/{count} distinct matched layer "
            "allocations without exhausting the search space"
        )
    return {
        "requested_count": count,
        "actual_count": len(candidates),
        "exhaustive": exhaustive and len(candidates) < count,
        "total_candidate_sets": total_candidate_sets,
        "feasible_candidate_sets": feasible_candidate_sets,
        "shortfall_reason": (
            "fewer distinct whole-layer allocations are mathematically feasible"
            if len(candidates) < count
            else None
        ),
        "sets": [list(candidate) for candidate in candidates],
    }


def make_unique_random_layer_allocations(
    module_rows: list[dict],
    target_w8_names: set[str],
    selected_layers: list[int],
    count: int = RANDOM_ALLOCATIONS,
    seed: int = 20261001,
) -> list[list[int]]:
    """Compatibility wrapper returning the locked layer sets."""
    return make_random_layer_allocation_plan(
        module_rows, target_w8_names, selected_layers, count=count, seed=seed
    )["sets"]


def _exact_budget_subset(
    module_rows: list[dict],
    budget_params: int,
    priorities: dict[str, tuple[float, ...]],
    seed: int,
) -> set[str]:
    """Find the lexicographically best subset at an exact parameter budget."""
    if budget_params < 0:
        raise ValueError("budget_params cannot be negative")
    if budget_params == 0:
        return set()
    rows = sorted(module_rows, key=lambda row: row["name"])
    if not rows:
        raise RuntimeError("No modules are available for a positive W8 budget")
    if len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Module names must be unique")
    divisor = math.gcd(
        budget_params,
        *[int(row["n_params"]) for row in rows],
    )
    normalized_budget = budget_params // divisor
    rng = random.Random(seed)
    tie_values = [rng.random() for _ in rows]
    width = max((len(value) for value in priorities.values()), default=1)
    zero_objective = (0.0,) * width
    states: dict[int, tuple[tuple[float, ...], float, int]] = {
        0: (zero_objective, 0.0, 0)
    }
    for index, row in enumerate(rows):
        weight = int(row["n_params"]) // divisor
        if weight <= 0 or weight > normalized_budget:
            continue
        value = priorities.get(row["name"], zero_objective)
        if len(value) != width:
            raise ValueError("Every priority tuple must have the same width")
        for subtotal, (objective, tie, mask) in list(states.items()):
            candidate_weight = subtotal + weight
            if candidate_weight > normalized_budget:
                continue
            candidate = (
                tuple(left + right for left, right in zip(objective, value)),
                tie + tie_values[index],
                mask | (1 << index),
            )
            incumbent = states.get(candidate_weight)
            if incumbent is None or (candidate[0], candidate[1], -candidate[2]) > (
                incumbent[0],
                incumbent[1],
                -incumbent[2],
            ):
                states[candidate_weight] = candidate
    if normalized_budget not in states:
        raise RuntimeError(
            f"No exact module allocation exists at the {budget_params}-parameter W8 budget"
        )
    mask = states[normalized_budget][2]
    return {
        row["name"] for index, row in enumerate(rows) if mask & (1 << index)
    }


def make_unique_random_module_allocations(
    module_rows: list[dict],
    target_w8_names: set[str],
    count: int = RANDOM_ALLOCATIONS,
    seed: int = 20262001,
) -> list[list[str]]:
    """Pre-register distinct module allocations at the exact SG W8 budget."""
    target_params = sum(
        int(row["n_params"])
        for row in module_rows
        if row["name"] in target_w8_names
    )
    rng = random.Random(seed)
    allocations: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    attempts = 0
    while len(allocations) < count and attempts < 10_000:
        attempts += 1
        priorities = {row["name"]: (rng.random(),) for row in module_rows}
        chosen = _exact_budget_subset(
            module_rows, target_params, priorities, seed + attempts
        )
        key = tuple(sorted(chosen))
        if not key or key in seen:
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
    priorities = {
        row["name"]: (
            float(int(row["n_params"])) if row["short"] in preferred_shorts else 0.0,
        )
        for row in module_rows
    }
    selected = _exact_budget_subset(module_rows, target_params, priorities, seed)
    chosen_preferred = {
        row["name"]
        for row in module_rows
        if row["name"] in selected and row["short"] in preferred_shorts
    }
    chosen_filler = selected - chosen_preferred
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
    """Allocate W8 by score density and repair to the exact SG budget."""
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
    greedy_selected = set(selected)
    priorities = {
        row["name"]: (
            float(int(row["n_params"])) if row["name"] in greedy_selected else 0.0,
            float(scores.get(row["name"], 0.0)),
        )
        for row in module_rows
    }
    selected = _exact_budget_subset(
        module_rows, target_params, priorities, seed=20264001
    )
    used = sum(
        int(row["n_params"]) for row in module_rows if row["name"] in selected
    )
    return {
        "selected_module_names": sorted(selected),
        "greedy_score_density_module_names": sorted(greedy_selected),
        "exact_budget_added_module_names": sorted(selected - greedy_selected),
        "exact_budget_removed_module_names": sorted(greedy_selected - selected),
        "allocation_algorithm": "greedy_score_density_with_exact_budget_repair",
        "target_w8_params": target_params,
        "actual_w8_params": used,
        "gap_params": target_params - used,
    }


def validate_random_allocation_manifest(manifest: dict) -> dict[str, int]:
    """Validate legacy and feasibility-aware random-allocation manifests."""
    requested = int(
        manifest.get(
            "requested_count_per_family",
            manifest.get("count_per_family", -1),
        )
    )
    if requested != RANDOM_ALLOCATIONS:
        raise ValueError("random-allocation requested count does not match the protocol")
    counts: dict[str, int] = {}
    for family, key in (("layer", "layer_sets"), ("module", "module_sets")):
        rows = manifest.get(key, [])
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"random-{family} allocation list is empty")
        canonical = [tuple(row) for row in rows]
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"random-{family} allocations are not unique")
        if manifest.get(f"{key}_sha256") != json_sha256(rows):
            raise ValueError(f"random-{family} allocation hash is invalid")
        if any(list(row) != sorted(set(row)) for row in rows):
            raise ValueError(f"random-{family} allocation members are not canonical")
        counts[family] = len(rows)
    if counts["module"] != requested:
        raise ValueError("random-module allocation count is incomplete")
    if counts["layer"] > requested:
        raise ValueError("random-layer allocation count exceeds the requested count")
    if counts["layer"] < requested:
        feasibility = manifest.get("layer_feasibility", {})
        if (
            feasibility.get("exhaustive") is not True
            or int(feasibility.get("requested_count", -1)) != requested
            or int(feasibility.get("actual_count", -1)) != counts["layer"]
            or int(feasibility.get("feasible_candidate_sets", -1))
            != counts["layer"]
        ):
            raise ValueError(
                "random-layer shortfall lacks an exhaustive feasibility certificate"
            )
    return counts


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

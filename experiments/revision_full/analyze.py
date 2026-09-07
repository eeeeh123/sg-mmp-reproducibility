"""Analyze complete-test revision runs; refuses incomplete 1,319-row results."""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from experiments.fix_gsm8k_500.direct_eval import mcnemar_exact_p
from experiments.revision_full.protocol import (
    BROAD_TASKS,
    CALIB_SEEDS,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_FORMAT_BATCH_SIZE,
    EXTRA_TASKS,
    GSM8K_TEST_SIZE,
    MODEL_SPECS,
    MAX_NEW_TOKENS,
    OUT,
    RANDOM_ALLOCATIONS,
    RANDOM_CALIB_SEED,
    RESULTS_DIR,
    PROTOCOL_VERSION,
    STATE_METADATA_DIR,
    method_id,
    validate_random_allocation_manifest,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sample_path(model_key: str, method: str) -> Path:
    return RESULTS_DIR / "samples" / f"{model_key}__{method}__gsm8k{GSM8K_TEST_SIZE}.jsonl"


def correctness(model_key: str, method: str) -> dict[int, int]:
    path = sample_path(model_key, method)
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    from experiments.revision_full.run import dataset_provenance, model_provenance

    data_hash = dataset_provenance()["manifest_sha256"]
    model_revision = model_provenance(model_key)["resolved_revision"]
    by_id = {int(row["doc_id"]): int(row["correct"]) for row in rows}
    if len(rows) != GSM8K_TEST_SIZE or len(by_id) != len(rows) or set(by_id) != set(
        range(GSM8K_TEST_SIZE)
    ) or (
        not method.startswith("external_")
        and any(
            row.get("protocol_version") != PROTOCOL_VERSION
            or row.get("dataset_manifest_sha256") != data_hash
            or row.get("model_revision") != model_revision
            or int(row.get("eval_batch_size_per_gpu", -1))
            != DEFAULT_EVAL_BATCH_SIZE
            or int(row.get("max_new_tokens", -1)) != MAX_NEW_TOKENS
            for row in rows
        )
    ):
        raise RuntimeError(
            f"Incomplete or duplicated full-test result: {path} has {len(by_id)} unique ids"
        )
    return by_id


def format_correctness(model_key: str, method: str) -> dict[int, int]:
    path = (
        RESULTS_DIR
        / "format_control"
        / f"{model_key}__{method}__gsm8k_mcq{GSM8K_TEST_SIZE}.jsonl"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_jsonl(path)
    from experiments.revision_full.run import dataset_provenance, model_provenance

    data_hash = dataset_provenance()["manifest_sha256"]
    model_revision = model_provenance(model_key)["resolved_revision"]
    by_id = {int(row["doc_id"]): int(row["correct"]) for row in rows}
    if len(rows) != GSM8K_TEST_SIZE or len(by_id) != len(rows) or set(by_id) != set(
        range(GSM8K_TEST_SIZE)
    ) or any(
        row.get("protocol_version") != PROTOCOL_VERSION
        or row.get("dataset_manifest_sha256") != data_hash
        or row.get("model_revision") != model_revision
        or int(row.get("format_batch_size_per_gpu", -1))
        != DEFAULT_FORMAT_BATCH_SIZE
        for row in rows
    ):
        raise RuntimeError(
            f"Incomplete format-control result: {path} has {len(by_id)} unique ids"
        )
    return by_id


def paired(a: dict[int, int], b: dict[int, int]) -> dict:
    ids = list(range(GSM8K_TEST_SIZE))
    av = [a[index] for index in ids]
    bv = [b[index] for index in ids]
    fixed = sum(1 for x, y in zip(av, bv) if not x and y)
    lost = sum(1 for x, y in zip(av, bv) if x and not y)
    differences = np.asarray(bv, dtype=np.float64) - np.asarray(av, dtype=np.float64)
    rng = np.random.default_rng(20260831)
    draws = []
    for start in range(0, 10000, 250):
        batch = min(250, 10000 - start)
        indices = rng.integers(0, len(differences), size=(batch, len(differences)))
        draws.append(100 * differences[indices].mean(axis=1))
    bootstrap = np.sort(np.concatenate(draws))
    lo = float(bootstrap[int(0.025 * len(bootstrap))])
    hi = float(bootstrap[int(0.975 * len(bootstrap))])
    boot_mean = float(bootstrap.mean())
    return {
        "n": len(ids),
        "a_accuracy": 100 * sum(av) / len(av),
        "b_accuracy": 100 * sum(bv) / len(bv),
        "delta": 100 * (sum(bv) - sum(av)) / len(av),
        "a_wrong_b_correct": fixed,
        "a_correct_b_wrong": lost,
        "mcnemar_p_exact": mcnemar_exact_p(fixed, lost),
        "paired_bootstrap_mean": boot_mean,
        "paired_bootstrap_ci95": [lo, hi],
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def bootstrap_mean(values: list[float], iters: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    draws = []
    for start in range(0, iters, 250):
        batch = min(250, iters - start)
        indices = rng.integers(0, len(array), size=(batch, len(array)))
        draws.append(array[indices].mean(axis=1))
    draws = np.sort(np.concatenate(draws))
    return {
        "mean": statistics.mean(values),
        "bootstrap_mean": float(draws.mean()),
        "ci95": [
            float(draws[int(0.025 * iters)]),
            float(draws[int(0.975 * iters)]),
        ],
    }


def hierarchical_seed_example_bootstrap(
    pairs: list[tuple[dict[int, int], dict[int, int]]],
    iters: int = 10000,
    seed: int = 20260831,
) -> dict:
    """Two-stage bootstrap over calibration runs and paired test examples."""
    rng = np.random.default_rng(seed)
    seed_count = len(pairs)
    ids = list(range(GSM8K_TEST_SIZE))
    differences = np.asarray(
        [[b[index] - a[index] for index in ids] for a, b in pairs],
        dtype=np.float64,
    )
    observed_by_seed = (100 * differences.mean(axis=1)).tolist()
    draws = []
    for start in range(0, iters, 100):
        batch = min(100, iters - start)
        sampled_seed_ids = rng.integers(0, seed_count, size=(batch, seed_count))
        sampled_example_ids = rng.integers(
            0, GSM8K_TEST_SIZE, size=(batch, seed_count, GSM8K_TEST_SIZE)
        )
        sampled = differences[
            sampled_seed_ids[:, :, None], sampled_example_ids
        ]
        draws.append(100 * sampled.mean(axis=(1, 2)))
    draws = np.sort(np.concatenate(draws))
    # The same test items recur across calibration seeds.  For a model-level
    # null test, average each item's effect across seeds and flip signs at the
    # item (cluster) level.  This avoids treating the seeds as independent
    # claims or the seed-by-item cells as independent observations.
    item_effects = differences.mean(axis=0)
    observed = abs(float(item_effects.mean()))
    extreme = 0
    for start in range(0, iters, 250):
        batch = min(250, iters - start)
        signs = rng.choice((-1.0, 1.0), size=(batch, GSM8K_TEST_SIZE))
        permuted = np.abs((signs * item_effects).mean(axis=1))
        extreme += int((permuted >= observed).sum())
    sign_flip_p = (extreme + 1) / (iters + 1)
    return {
        "n_seeds": seed_count,
        "n_examples_per_seed": GSM8K_TEST_SIZE,
        "mean_delta": statistics.mean(observed_by_seed),
        "seed_sd": statistics.stdev(observed_by_seed)
        if len(observed_by_seed) > 1
        else 0.0,
        "seed_min": min(observed_by_seed),
        "seed_max": max(observed_by_seed),
        "seed_deltas": observed_by_seed,
        "ci95": [
            float(draws[int(0.025 * iters)]),
            float(draws[int(0.975 * iters)]),
        ],
        "paired_item_cluster_sign_flip_p": sign_flip_p,
        "resampling": "calibration seeds, then paired examples within seed",
    }


def paired_interaction(
    first_a: dict[int, int],
    first_b: dict[int, int],
    second_a: dict[int, int],
    second_b: dict[int, int],
    seed: int,
    iters: int = 10000,
) -> dict:
    """Paired difference-in-differences with bootstrap CI and sign-flip p."""
    values = [
        (first_b[index] - first_a[index])
        - (second_b[index] - second_a[index])
        for index in range(GSM8K_TEST_SIZE)
    ]
    bootstrap = bootstrap_mean([100 * value for value in values], iters, seed)
    observed = abs(statistics.mean(values))
    rng = random.Random(seed + 1)
    extreme = 0
    for _ in range(iters):
        permuted = abs(statistics.mean(value * rng.choice((-1, 1)) for value in values))
        extreme += permuted >= observed
    return {
        "delta_points": 100 * statistics.mean(values),
        "paired_bootstrap_ci95": bootstrap["ci95"],
        "sign_flip_p": (extreme + 1) / (iters + 1),
    }


def relative_error_increase(fp16_accuracy: float, quantized_accuracy: float) -> float | None:
    baseline_error = 100.0 - fp16_accuracy
    if baseline_error <= 0:
        return None
    return ((100.0 - quantized_accuracy) - baseline_error) / baseline_error


PANEL_METRICS = {
    "svamp": ("exact_match,flexible-extract", "exact_match,none"),
    "asdiv_gen": ("exact_match,flexible-extract", "exact_match,none"),
    "hendrycks_math500": (
        "exact_match,flexible-extract",
        "exact_match,none",
    ),
    "truthfulqa_gen": ("bleu_acc,none", "rouge1_acc,none"),
}


def metric_percentage(metrics: dict, candidates: tuple[str, ...]) -> tuple[str, float]:
    for name in candidates:
        if name in metrics:
            value = metrics[name]
            if isinstance(value, (list, tuple)):
                value = value[0]
            value = float(value)
            return name, 100.0 * value if abs(value) <= 1.0 else value
    raise RuntimeError(f"None of the locked metrics {candidates} exists in {sorted(metrics)}")


def task_point(
    model_key: str,
    method: str,
    task: str,
    *,
    broad: bool,
) -> tuple[str, float]:
    directory = "broad" if broad else "extra"
    record = json.loads(
        (RESULTS_DIR / directory / f"{model_key}__{method}.json").read_text(
            encoding="utf-8"
        )
    )
    from experiments.revision_full.run import dataset_provenance, model_provenance

    if (
        record.get("protocol_version") != PROTOCOL_VERSION
        or record.get("model_snapshot") != model_provenance(model_key)
        or record.get("dataset_snapshot") != dataset_provenance()
        or int(record.get("batch_size_per_gpu", -1))
        != DEFAULT_EVAL_BATCH_SIZE
        or int(record.get("max_new_tokens", -1)) != MAX_NEW_TOKENS
    ):
        raise RuntimeError(f"Stale task-panel provenance: {model_key}/{method}/{task}")
    if broad:
        return "accuracy", float(record["scores"][task])
    return metric_percentage(record["results"][task]["metrics"], PANEL_METRICS[task])


def module_placement_control_rows() -> list[dict]:
    rows = []
    control_variants = [
        "qkv_only",
        "o_only",
        "ffn_only",
        "qkv_priority_matched",
        "o_priority_matched",
        "ffn_priority_matched",
        "hessian_diag_matched",
    ]
    for model_key, spec in MODEL_SPECS.items():
        sg_method = method_id("sg_mmp", RANDOM_CALIB_SEED)
        try:
            sg = correctness(model_key, sg_method)
        except FileNotFoundError:
            continue
        for variant in control_variants:
            method = method_id(variant, RANDOM_CALIB_SEED)
            metadata_path = (
                STATE_METADATA_DIR
                / model_key
                / f"calib_{RANDOM_CALIB_SEED}"
                / f"{variant}.json"
            )
            try:
                control = correctness(model_key, method)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            rows.append(
                {
                    "model_key": model_key,
                    "model": spec["display_name"],
                    "variant": variant,
                    "calibration_seed": RANDOM_CALIB_SEED,
                    "parameter_weighted_average_bits": metadata[
                        "parameter_weighted_average_bits"
                    ],
                    "allocation_details": metadata.get("allocation_details"),
                    "control_vs_sg": paired(control, sg),
                }
            )
    if rows:
        corrected = holm_adjust(
            [row["control_vs_sg"]["mcnemar_p_exact"] for row in rows]
        )
        for row, value in zip(rows, corrected):
            row["control_vs_sg"]["mcnemar_p_holm_family"] = value
    return rows


def analyze() -> dict:
    from experiments.revision_full.run import dataset_provenance, model_provenance

    dataset_identity = dataset_provenance()
    model_revisions = {
        model_key: model_provenance(model_key)["resolved_revision"]
        for model_key in MODEL_SPECS
    }
    comparisons = []
    seed_summary = []
    for model_key, spec in MODEL_SPECS.items():
        fp16 = correctness(model_key, "fp16")
        model_rows = []
        model_pairs = {
            "sg_vs_gptq_w4": [],
            "sg_vs_gptq_w5": [],
            "sg_vs_gptq_w6": [],
        }
        for seed in CALIB_SEEDS:
            w4_method = method_id("gptq_w4", seed)
            sg_method = method_id("sg_mmp", seed)
            w5_method = method_id("gptq_w5", seed)
            w6_method = method_id("gptq_w6", seed)
            w4 = correctness(model_key, w4_method)
            sg = correctness(model_key, sg_method)
            w5 = correctness(model_key, w5_method)
            w6 = correctness(model_key, w6_method)
            sg_vs_w4 = paired(w4, sg)
            sg_vs_w5 = paired(w5, sg)
            sg_vs_w6 = paired(w6, sg)
            w5_vs_w4 = paired(w4, w5)
            w6_vs_w4 = paired(w4, w6)
            model_pairs["sg_vs_gptq_w4"].append((w4, sg))
            model_pairs["sg_vs_gptq_w5"].append((w5, sg))
            model_pairs["sg_vs_gptq_w6"].append((w6, sg))
            fp16_acc = 100 * sum(fp16.values()) / GSM8K_TEST_SIZE
            denominator = fp16_acc - sg_vs_w4["a_accuracy"]
            recovery = sg_vs_w4["delta"] / denominator if denominator > 0 else None
            row = {
                "model_key": model_key,
                "model": spec["display_name"],
                "calibration_seed": seed,
                "fp16_accuracy": fp16_acc,
                "sg_vs_gptq_w4": sg_vs_w4,
                "sg_vs_gptq_w5": sg_vs_w5,
                "sg_vs_gptq_w6": sg_vs_w6,
                "gptq_w5_vs_gptq_w4": w5_vs_w4,
                "gptq_w6_vs_gptq_w4": w6_vs_w4,
                "gptq_w4_relative_error_increase": relative_error_increase(
                    fp16_acc, sg_vs_w4["a_accuracy"]
                ),
                "normalized_recovery": recovery,
            }
            comparisons.append(row)
            model_rows.append(row)
        seed_summary.append(
            {
                "model_key": model_key,
                "model": spec["display_name"],
                "calibration_seeds": list(CALIB_SEEDS),
                "comparisons": {
                    name: hierarchical_seed_example_bootstrap(
                        pairs, seed=20260831 + offset
                    )
                    for offset, (name, pairs) in enumerate(model_pairs.items())
                },
            }
        )

    for comparison_name in [
        "sg_vs_gptq_w4",
        "sg_vs_gptq_w5",
        "sg_vs_gptq_w6",
    ]:
        p_values = [row[comparison_name]["mcnemar_p_exact"] for row in comparisons]
        adjusted = holm_adjust(p_values)
        for row, corrected in zip(comparisons, adjusted):
            row[comparison_name]["mcnemar_p_holm_within_family"] = corrected

    format_controls = []
    for model_key, spec in MODEL_SPECS.items():
        methods = {
            "fp16": "fp16",
            "gptq_w4": method_id("gptq_w4", RANDOM_CALIB_SEED),
            "sg_mmp": method_id("sg_mmp", RANDOM_CALIB_SEED),
        }
        try:
            generative = {name: correctness(model_key, method) for name, method in methods.items()}
            multiple_choice = {
                name: format_correctness(model_key, method) for name, method in methods.items()
            }
        except FileNotFoundError:
            continue
        gen_acc = {
            name: 100 * sum(values.values()) / GSM8K_TEST_SIZE
            for name, values in generative.items()
        }
        mc_acc = {
            name: 100 * sum(values.values()) / GSM8K_TEST_SIZE
            for name, values in multiple_choice.items()
        }
        w4_format_interaction = paired_interaction(
            generative["gptq_w4"],
            generative["fp16"],
            multiple_choice["gptq_w4"],
            multiple_choice["fp16"],
            seed=20265001 + len(format_controls),
        )
        sg_recovery_format_interaction = paired_interaction(
            generative["gptq_w4"],
            generative["sg_mmp"],
            multiple_choice["gptq_w4"],
            multiple_choice["sg_mmp"],
            seed=20266001 + len(format_controls),
        )
        format_controls.append(
            {
                "model_key": model_key,
                "model": spec["display_name"],
                "calibration_seed": RANDOM_CALIB_SEED,
                "generative_accuracy": gen_acc,
                "multiple_choice_accuracy": mc_acc,
                "fp16_to_gptq_drop_generative": gen_acc["fp16"] - gen_acc["gptq_w4"],
                "fp16_to_gptq_drop_multiple_choice": mc_acc["fp16"] - mc_acc["gptq_w4"],
                "drop_difference_generation_minus_mcq": (
                    gen_acc["fp16"]
                    - gen_acc["gptq_w4"]
                    - mc_acc["fp16"]
                    + mc_acc["gptq_w4"]
                ),
                "w4_format_interaction": w4_format_interaction,
                "sg_recovery_format_interaction": sg_recovery_format_interaction,
            }
        )

    format_tests = [
        row[test_name]["sign_flip_p"]
        for row in format_controls
        for test_name in ["w4_format_interaction", "sg_recovery_format_interaction"]
    ]
    if format_tests:
        corrected = iter(holm_adjust(format_tests))
        for row in format_controls:
            for test_name in ["w4_format_interaction", "sg_recovery_format_interaction"]:
                row[test_name]["sign_flip_p_holm"] = next(corrected)

    random_controls = []
    for model_key, spec in MODEL_SPECS.items():
        if spec["role"] != "primary":
            continue
        sg_method = method_id("sg_mmp", RANDOM_CALIB_SEED)
        try:
            sg = correctness(model_key, sg_method)
        except FileNotFoundError:
            continue
        sg_accuracy = 100 * sum(sg.values()) / GSM8K_TEST_SIZE
        selection = json.loads(
            (OUT / "selections" / f"{model_key}.json").read_text(encoding="utf-8")
        )
        random_counts = validate_random_allocation_manifest(
            selection.get("random_allocation_manifest", {})
        )
        for prefix, allocation_kind in [
            ("random", "layer"),
            ("random_modules", "module"),
        ]:
            required_count = random_counts[allocation_kind]
            random_accuracies = []
            missing = []
            for allocation_id in range(required_count):
                random_method = method_id(
                    f"{prefix}_{allocation_id}", RANDOM_CALIB_SEED
                )
                try:
                    random_result = correctness(model_key, random_method)
                except FileNotFoundError:
                    missing.append(allocation_id)
                    continue
                random_accuracies.append(
                    100 * sum(random_result.values()) / GSM8K_TEST_SIZE
                )
            if random_accuracies:
                percentile = 100 * sum(
                    value <= sg_accuracy for value in random_accuracies
                ) / len(random_accuracies)
                empirical_p = (
                    1 + sum(value >= sg_accuracy for value in random_accuracies)
                ) / (len(random_accuracies) + 1)
                summary = {
                    "model_key": model_key,
                    "model": spec["display_name"],
                    "allocation_kind": allocation_kind,
                    "calibration_seed": RANDOM_CALIB_SEED,
                    "requested_random_allocations": RANDOM_ALLOCATIONS,
                    "required_random_allocations": required_count,
                    "completed_random_allocations": len(random_accuracies),
                    "missing_allocation_ids": missing,
                    "sg_accuracy": sg_accuracy,
                    "random_mean": statistics.mean(random_accuracies),
                    "random_std": statistics.stdev(random_accuracies)
                    if len(random_accuracies) > 1
                    else 0.0,
                    "random_min": min(random_accuracies),
                    "random_max": max(random_accuracies),
                    "sg_percentile": percentile,
                    "empirical_one_sided_p": empirical_p,
                    "evidence_complete": len(random_accuracies) == required_count,
                }
            else:
                summary = {
                    "model_key": model_key,
                    "model": spec["display_name"],
                    "allocation_kind": allocation_kind,
                    "calibration_seed": RANDOM_CALIB_SEED,
                    "requested_random_allocations": RANDOM_ALLOCATIONS,
                    "required_random_allocations": required_count,
                    "completed_random_allocations": 0,
                    "missing_allocation_ids": missing,
                    "evidence_complete": False,
                }
            random_controls.append(summary)

    complete_random_controls = [
        row
        for row in random_controls
        if row.get("evidence_complete") is True
        and row.get("empirical_one_sided_p") is not None
    ]
    if complete_random_controls:
        corrected = holm_adjust(
            [row["empirical_one_sided_p"] for row in complete_random_controls]
        )
        for row, value in zip(complete_random_controls, corrected):
            row["empirical_one_sided_p_holm_family"] = value

    module_controls = module_placement_control_rows()

    task_controls = []
    for model_key, spec in MODEL_SPECS.items():
        methods = {
            "fp16": "fp16",
            "gptq_w4": method_id("gptq_w4", RANDOM_CALIB_SEED),
            "sg_mmp": method_id("sg_mmp", RANDOM_CALIB_SEED),
        }
        task_points = []
        canonical = {
            name: 100 * sum(correctness(model_key, method).values()) / GSM8K_TEST_SIZE
            for name, method in methods.items()
        }
        task_points.append(("gsm8k", "math", "generation", "exact_match", canonical))
        format_values = {
            name: 100
            * sum(format_correctness(model_key, method).values())
            / GSM8K_TEST_SIZE
            for name, method in methods.items()
        }
        task_points.append(
            ("gsm8k_mcq", "math", "multiple_choice", "accuracy", format_values)
        )
        for task in BROAD_TASKS:
            values = {}
            metric = None
            for name, method in methods.items():
                current_metric, value = task_point(
                    model_key, method, task, broad=True
                )
                if metric is not None and current_metric != metric:
                    raise RuntimeError(f"Metric drift for {model_key}/{task}")
                metric = current_metric
                values[name] = value
            category = "math" if task == "mmlu_high_school_mathematics" else "non_math"
            task_points.append((task, category, "multiple_choice", metric, values))
        for task in EXTRA_TASKS:
            values = {}
            metric = None
            for name, method in methods.items():
                current_metric, value = task_point(
                    model_key, method, task, broad=False
                )
                if metric is not None and current_metric != metric:
                    raise RuntimeError(f"Metric drift for {model_key}/{task}")
                metric = current_metric
                values[name] = value
            category = "math" if task != "truthfulqa_gen" else "non_math"
            task_points.append((task, category, "generation", metric, values))

        for task, category, output_format, metric, values in task_points:
            denominator = values["fp16"] - values["gptq_w4"]
            task_controls.append(
                {
                    "model_key": model_key,
                    "model": spec["display_name"],
                    "task": task,
                    "category": category,
                    "output_format": output_format,
                    "metric": metric,
                    "calibration_seed": RANDOM_CALIB_SEED,
                    "fp16": values["fp16"],
                    "gptq_w4": values["gptq_w4"],
                    "sg_mmp": values["sg_mmp"],
                    "gptq_w4_absolute_degradation": values["fp16"]
                    - values["gptq_w4"],
                    "gptq_w4_relative_error_increase": relative_error_increase(
                        values["fp16"], values["gptq_w4"]
                    ),
                    "sg_minus_gptq_w4": values["sg_mmp"] - values["gptq_w4"],
                    "normalized_recovery": (
                        (values["sg_mmp"] - values["gptq_w4"]) / denominator
                        if denominator > 0
                        else None
                    ),
                }
            )
    external_baselines = []
    registry = OUT / "external_baselines"
    external_metadata_paths = [
        path for path in sorted(registry.glob("*.json")) if "__config" not in path.name
    ]
    if external_metadata_paths:
        from experiments.revision_full.external_baselines import validate

        validate()
    external_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in external_metadata_paths
    ]
    grouped_external: dict[tuple[str, str], list[dict]] = {}
    for metadata in external_records:
        grouped_external.setdefault(
            (metadata["model_key"], metadata["method"]), []
        ).append(metadata)
    for (model_key, external_method), records in sorted(grouped_external.items()):
        seeded = {
            int(record["calibration_seed"]): record
            for record in records
            if record.get("calibration_seed") is not None
        }
        if external_method == "tacq":
            if set(seeded) != set(CALIB_SEEDS):
                # Partial registrations remain visible to readiness, but do not
                # create a selectively reported scientific comparison.
                continue
            diagnostics = []
            pairs = []
            for calib_seed in CALIB_SEEDS:
                record = seeded[calib_seed]
                external = correctness(model_key, record["method_id"])
                sg = correctness(model_key, method_id("sg_mmp", calib_seed))
                diagnostics.append(
                    {
                        "calibration_seed": calib_seed,
                        "sg_minus_tacq": paired(external, sg),
                        "inference_role": "diagnostic; not an independent primary hypothesis",
                    }
                )
                pairs.append((external, sg))
            primary = hierarchical_seed_example_bootstrap(
                pairs,
                seed=20269600 + sorted(MODEL_SPECS).index(model_key),
            )
            external_baselines.append(
                {
                    "model_key": model_key,
                    "model": records[0]["model"],
                    "method": external_method,
                    "method_label": "TaCQ shared-backend adaptation",
                    "calibration_seeds": list(CALIB_SEEDS),
                    "parameter_weighted_average_bits": [
                        seeded[seed]["parameter_weighted_average_bits"]
                        for seed in CALIB_SEEDS
                    ],
                    "sg_parameter_weighted_average_bits": [
                        seeded[seed]["sg_parameter_weighted_average_bits"]
                        for seed in CALIB_SEEDS
                    ],
                    "registration_evidence": [
                        {
                            "calibration_seed": seed,
                            "samples_sha256": seeded[seed]["samples_sha256"],
                            "config_sha256": seeded[seed]["config_sha256"],
                            "source_commit": seeded[seed]["source_commit"],
                        }
                        for seed in CALIB_SEEDS
                    ],
                    "seed_diagnostics": diagnostics,
                    "model_level_sg_minus_tacq": primary,
                    "primary_hypothesis": True,
                }
            )
        else:
            # Non-preregistered or single-run external methods are diagnostic
            # only and are never mixed into the two-hypothesis TaCQ family.
            for record in records:
                seed = record.get("calibration_seed") or RANDOM_CALIB_SEED
                external_baselines.append(
                    {
                        **record,
                        "external_vs_sg": paired(
                            correctness(model_key, record["method_id"]),
                            correctness(model_key, method_id("sg_mmp", seed)),
                        ),
                        "primary_hypothesis": False,
                    }
                )
    primary_tacq = [
        row for row in external_baselines if row.get("primary_hypothesis") is True
    ]
    if primary_tacq:
        if len(primary_tacq) != 2:
            raise RuntimeError(
                "Primary TaCQ inference requires exactly two complete model-level comparisons"
            )
        corrected = holm_adjust(
            [
                row["model_level_sg_minus_tacq"]["paired_item_cluster_sign_flip_p"]
                for row in primary_tacq
            ]
        )
        for row, value in zip(primary_tacq, corrected):
            row["model_level_sg_minus_tacq"]["paired_item_cluster_sign_flip_p_holm_two_models"] = value

    result = {
        "protocol_version": PROTOCOL_VERSION,
        "dataset_manifest_sha256": dataset_identity["manifest_sha256"],
        "model_revisions": model_revisions,
        "execution": {
            "eval_batch_size_per_gpu": DEFAULT_EVAL_BATCH_SIZE,
            "format_batch_size_per_gpu": DEFAULT_FORMAT_BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "test_protocol": "complete official GSM8K test set",
        "n": GSM8K_TEST_SIZE,
        "comparisons": comparisons,
        "run_level_summary": seed_summary,
        "same_item_format_controls": format_controls,
        "random_same_budget_controls": random_controls,
        "module_placement_controls": module_controls,
        "cross_task_format_controls": task_controls,
        "external_baselines": external_baselines,
        "run_level_inference": "Two-stage bootstrap over three calibration seeds and paired examples.",
    }
    output_json = OUT / "analysis_full.json"
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Full GSM8K revision analysis",
        "",
        f"Every row uses all {GSM8K_TEST_SIZE} official test examples. Layer selection used GSM8K train only.",
        "",
        "| Model | Calib seed | W4 | W5 | W6 | SG-MMP | SG-W4 | 95% paired CI | McNemar Holm p | W4 rel. error increase | Recovery |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in comparisons:
        comp = row["sg_vs_gptq_w4"]
        recovery = row["normalized_recovery"]
        recovery_text = "NA" if recovery is None else f"{100 * recovery:.1f}%"
        relative_error = row["gptq_w4_relative_error_increase"]
        relative_error_text = (
            "NA" if relative_error is None else f"{100 * relative_error:.1f}%"
        )
        lines.append(
            f"| {row['model']} | {row['calibration_seed']} | {comp['a_accuracy']:.2f} | "
            f"{row['sg_vs_gptq_w5']['a_accuracy']:.2f} | "
            f"{row['sg_vs_gptq_w6']['a_accuracy']:.2f} | "
            f"{comp['b_accuracy']:.2f} | {comp['delta']:+.2f} | "
            f"[{comp['paired_bootstrap_ci95'][0]:+.2f}, {comp['paired_bootstrap_ci95'][1]:+.2f}] | "
            f"{comp['mcnemar_p_holm_within_family']:.4g} | "
            f"{relative_error_text} | {recovery_text} |"
        )
    lines.extend(
        [
            "",
            "## Run-level inference",
            "",
            "Two-stage bootstrap resamples three calibration seeds and paired test examples within each seed.",
            "",
            "| Model | Comparison | Per-seed deltas | Mean | SD | Min | Max | Hierarchical 95% CI |",
            "|---|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in seed_summary:
        for name, summary in row["comparisons"].items():
            lines.append(
                f"| {row['model']} | {name} | "
                f"{', '.join(f'{value:+.2f}' for value in summary['seed_deltas'])} | "
                f"{summary['mean_delta']:+.2f} | {summary['seed_sd']:.2f} | "
                f"{summary['seed_min']:+.2f} | {summary['seed_max']:+.2f} | "
                f"[{summary['ci95'][0]:+.2f}, {summary['ci95'][1]:+.2f}] |"
            )
    lines.extend(
        [
            "",
            "## Same-budget random allocation audit",
            "",
            "| Model | Allocation | Completed | SG accuracy | Random mean | SG percentile | Holm empirical p | Evidence complete |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in random_controls:
        if row["completed_random_allocations"]:
            lines.append(
                f"| {row['model']} | {row['allocation_kind']} | {row['completed_random_allocations']}/{row['required_random_allocations']} | "
                f"{row['sg_accuracy']:.2f} | {row['random_mean']:.2f} | {row['sg_percentile']:.1f} | "
                f"{row['empirical_one_sided_p_holm_family']:.4f} | {'yes' if row['evidence_complete'] else 'no'} |"
            )
        else:
            lines.append(
                f"| {row['model']} | {row['allocation_kind']} | 0/{row['required_random_allocations']} | NA | NA | NA | NA | no |"
            )
    lines.append("")
    lines.extend(
        [
            "## External matched-budget baselines",
            "",
            "Primary TaCQ inference aggregates three calibration seeds within each model. Per-seed paired tests are diagnostics; Holm correction covers exactly the two model-level hypotheses.",
            "",
            "| Model | Method | Avg bits by seed | SG minus external | Hierarchical 95% CI | Holm p (2 models) |",
            "|---|---|---|---:|---|---:|",
        ]
    )
    for row in external_baselines:
        if row.get("primary_hypothesis"):
            comp = row["model_level_sg_minus_tacq"]
            bits = ", ".join(
                f"c{seed}:{value:.3f}"
                for seed, value in zip(
                    row["calibration_seeds"],
                    row["parameter_weighted_average_bits"],
                )
            )
            lines.append(
                f"| {row['model']} | {row['method_label']} | {bits} | "
                f"{comp['mean_delta']:+.2f} | "
                f"[{comp['ci95'][0]:+.2f}, {comp['ci95'][1]:+.2f}] | "
                f"{comp['paired_item_cluster_sign_flip_p_holm_two_models']:.4g} |"
            )
        else:
            comp = row["external_vs_sg"]
            lines.append(
                f"| {row['model']} | {row['method']} (diagnostic) | "
                f"{row['parameter_weighted_average_bits']:.3f} | {comp['delta']:+.2f} | "
                f"[{comp['paired_bootstrap_ci95'][0]:+.2f}, {comp['paired_bootstrap_ci95'][1]:+.2f}] | NA |"
            )
    if not external_baselines:
        lines.append("| Not registered | NA | NA | NA | NA | NA |")
    lines.append("")
    lines.extend(
        [
            "## Module placement controls",
            "",
            "Pure-family points are reported at their actual bit budgets. Role-priority and Hessian controls are matched to SG-MMP.",
            "",
            "| Model | Variant | Avg bits | SG minus control | 95% paired CI | Holm p |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for row in module_controls:
        comp = row["control_vs_sg"]
        lines.append(
            f"| {row['model']} | {row['variant']} | {row['parameter_weighted_average_bits']:.3f} | "
            f"{comp['delta']:+.2f} | "
            f"[{comp['paired_bootstrap_ci95'][0]:+.2f}, {comp['paired_bootstrap_ci95'][1]:+.2f}] | "
            f"{comp['mcnemar_p_holm_family']:.4g} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Same-item format control",
            "",
            "| Model | Generation drop FP16->W4 | MCQ drop FP16->W4 | Interaction | 95% CI | Holm p |",
            "|---|---:|---:|---:|---|---:|",
        ]
    )
    for row in format_controls:
        lines.append(
            f"| {row['model']} | {row['fp16_to_gptq_drop_generative']:.2f} | "
            f"{row['fp16_to_gptq_drop_multiple_choice']:.2f} | "
            f"{row['w4_format_interaction']['delta_points']:+.2f} | "
            f"[{row['w4_format_interaction']['paired_bootstrap_ci95'][0]:+.2f}, "
            f"{row['w4_format_interaction']['paired_bootstrap_ci95'][1]:+.2f}] | "
            f"{row['w4_format_interaction']['sign_flip_p_holm']:.4g} |"
        )
    if not format_controls:
        lines.append("| Not yet complete | NA | NA | NA | NA | NA |")
    lines.append("")
    lines.extend(
        [
            "## Cross-task and format-normalized results",
            "",
            "Relative error increase and normalized recovery are descriptive for transfer tasks; paired inference is reserved for the locked same-item GSM8K comparisons.",
            "",
            "| Model | Task | Category | Format | Metric | FP16 | W4 | SG-MMP | W4 rel. error increase | Recovery |",
            "|---|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in task_controls:
        relative_error = row["gptq_w4_relative_error_increase"]
        recovery = row["normalized_recovery"]
        lines.append(
            f"| {row['model']} | {row['task']} | {row['category']} | "
            f"{row['output_format']} | {row['metric']} | {row['fp16']:.2f} | "
            f"{row['gptq_w4']:.2f} | {row['sg_mmp']:.2f} | "
            f"{'NA' if relative_error is None else f'{100 * relative_error:.1f}%'} | "
            f"{'NA' if recovery is None else f'{100 * recovery:.1f}%'} |"
        )
    lines.append("")
    (OUT / "analysis_full.md").write_text("\n".join(lines), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(analyze(), ensure_ascii=False, indent=2))

"""Deterministic accuracy and process-block inference for deployment-gguf-v1."""

from __future__ import annotations

import json
import math
import random
import statistics
from pathlib import Path

from experiments.deployment_gguf.protocol import (
    BENCH_DIR,
    FORMAL_MAX_BLOCKS,
    FORMAL_MIN_BLOCKS,
    MANIFEST_DIR,
    PILOT_BLOCKS,
    PROTOCOL_VERSION,
    PHASE_MODELS,
    QUALITY_DIR,
    atomic_write_json,
    sha256_file,
)


BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 2026090807


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _bootstrap_mean(values: list[float], seed: int) -> dict:
    if not values:
        raise ValueError("Cannot bootstrap an empty sample")
    rng = random.Random(seed)
    draws = [
        statistics.mean(rng.choice(values) for _ in values)
        for _ in range(BOOTSTRAP_REPLICATES)
    ]
    return {
        "mean": statistics.mean(values),
        "ci95": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
    }


def paired_ratio(values_a: dict[int, float], values_b: dict[int, float]) -> dict:
    blocks = sorted(set(values_a) & set(values_b))
    if not blocks:
        raise ValueError("No paired process blocks")
    if any(
        not math.isfinite(values_a[block])
        or not math.isfinite(values_b[block])
        or values_a[block] <= 0
        or values_b[block] <= 0
        for block in blocks
    ):
        raise ValueError("Paired ratio inputs must be finite and strictly positive")
    log_ratios = [math.log(values_a[block] / values_b[block]) for block in blocks]
    rng = random.Random(BOOTSTRAP_SEED + sum(blocks))
    draws = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled = [rng.choice(log_ratios) for _ in log_ratios]
        draws.append(math.exp(statistics.mean(sampled)))
    point = math.exp(statistics.mean(log_ratios))
    lower, upper = _percentile(draws, 0.025), _percentile(draws, 0.975)
    return {
        "paired_blocks": blocks,
        "n_process_blocks": len(blocks),
        "geometric_mean_ratio": point,
        "ci95": [lower, upper],
        "ci_half_width": (upper - lower) / 2,
        "two_percent_precision_reached": (
            len(blocks) >= FORMAL_MIN_BLOCKS and (upper - lower) / 2 <= 0.02
        ),
        "maximum_blocks": FORMAL_MAX_BLOCKS,
    }


def _exact_mcnemar(discordant_ab: int, discordant_ba: int) -> float:
    n = discordant_ab + discordant_ba
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(discordant_ab, discordant_ba) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_accuracy(path_a: Path, path_b: Path) -> dict:
    rows_a = {
        int(row["doc_id"]): row
        for row in _read_jsonl(path_a)
    }
    rows_b = {
        int(row["doc_id"]): row
        for row in _read_jsonl(path_b)
    }
    if set(rows_a) != set(rows_b) or not rows_a:
        raise RuntimeError("Quality files are not paired on the same complete items")
    item_deltas = [
        100 * (
            int(rows_a[index]["flexible_correct"])
            - int(rows_b[index]["flexible_correct"])
        )
        for index in sorted(rows_a)
    ]
    a_wrong_b_correct = sum(
        not rows_a[index]["flexible_correct"] and rows_b[index]["flexible_correct"]
        for index in rows_a
    )
    a_correct_b_wrong = sum(
        rows_a[index]["flexible_correct"] and not rows_b[index]["flexible_correct"]
        for index in rows_a
    )
    estimate = _bootstrap_mean(item_deltas, BOOTSTRAP_SEED)
    return {
        "n": len(rows_a),
        "a_accuracy": 100
        * statistics.mean(int(row["flexible_correct"]) for row in rows_a.values()),
        "b_accuracy": 100
        * statistics.mean(int(row["flexible_correct"]) for row in rows_b.values()),
        "delta_percentage_points": estimate["mean"],
        "paired_bootstrap_ci95": estimate["ci95"],
        "a_wrong_b_correct": a_wrong_b_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "mcnemar_exact_p": _exact_mcnemar(a_wrong_b_correct, a_correct_b_wrong),
    }


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _block_files(kind: str, phase: str, model_key: str, method: str) -> list[Path]:
    return sorted((BENCH_DIR / "blocks" / kind / phase / model_key / method).glob("block_*.json"))


def _micro_metrics(phase: str, model_key: str, method: str) -> dict[str, dict[int, float]]:
    metrics: dict[str, dict[int, float]] = {}
    for path in _block_files("micro", phase, model_key, method):
        record = json.loads(path.read_text(encoding="utf-8"))
        block = int(record["process_block"])
        for group_name in ("prompt_processing", "text_generation"):
            for row in record[group_name]["llama_bench_json"]:
                key = f"{group_name}__p{row['n_prompt']}__g{row['n_gen']}__d{row['n_depth']}__tokens_per_second"
                metrics.setdefault(key, {})[block] = statistics.median(
                    float(value) for value in row["samples_ts"]
                )
    return metrics


def _service_metrics(phase: str, model_key: str, method: str) -> dict[str, dict[int, float]]:
    metrics: dict[str, dict[int, float]] = {}
    for path in _block_files("service", phase, model_key, method):
        record = json.loads(path.read_text(encoding="utf-8"))
        block = int(record["process_block"])
        for concurrency in (1, 4):
            groups = [
                group for group in record["groups"] if int(group["concurrency"]) == concurrency
            ]
            metrics.setdefault(f"c{concurrency}__aggregate_tokens_per_second", {})[block] = statistics.median(
                float(group["aggregate_tokens_per_second"]) for group in groups
            )
            for field in ("ttft_seconds", "latency_seconds", "inter_token_latency_seconds"):
                values = [
                    float(request[field])
                    for group in groups
                    for request in group["requests"]
                    if request[field] is not None
                ]
                metrics.setdefault(f"c{concurrency}__{field}", {})[block] = statistics.median(values)
        resources = record["resources"]
        for field in ("loaded_model_gpu_delta_mib", "peak_gpu_delta_mib", "loaded_host_rss_mib", "startup_seconds"):
            if resources.get(field) is not None:
                metrics.setdefault(field, {})[block] = float(resources[field])
    return metrics


def analyze_model(model_key: str, phase: str) -> dict:
    if phase not in {"engineering", "value", "formal"}:
        raise ValueError(phase)
    if model_key not in PHASE_MODELS[phase]:
        raise ValueError(f"{model_key} is not registered for {phase} phase")
    methods = ("fp16", "q4", "q5", "sg")
    block_files = {
        f"{kind}/{method}": _block_files(kind, phase, model_key, method)
        for kind in ("micro", "service")
        for method in methods
    }
    block_sets = {
        key: {int(path.stem.removeprefix("block_")) for path in paths}
        for key, paths in block_files.items()
    }
    expected_minimum = FORMAL_MIN_BLOCKS if phase == "formal" else PILOT_BLOCKS
    reference_blocks = next(iter(block_sets.values()))
    if any(blocks != reference_blocks for blocks in block_sets.values()):
        raise RuntimeError("Micro/service process blocks are not paired across methods")
    if reference_blocks != set(range(len(reference_blocks))):
        raise RuntimeError("Process block IDs must be contiguous from zero")
    if len(reference_blocks) < expected_minimum:
        raise RuntimeError(
            f"{phase} analysis requires at least {expected_minimum} paired process blocks"
        )
    if phase != "formal" and len(reference_blocks) != PILOT_BLOCKS:
        raise RuntimeError("Pilot analysis is frozen to exactly five process blocks")
    if phase == "formal" and len(reference_blocks) > FORMAL_MAX_BLOCKS:
        raise RuntimeError("Formal analysis exceeds the frozen 30-block maximum")
    artifact_records = {}
    quality_summaries = {}
    micro = {}
    service = {}
    for method in methods:
        artifact_path = MANIFEST_DIR / "artifacts" / model_key / f"{method}.json"
        if artifact_path.is_file():
            artifact_records[method] = json.loads(artifact_path.read_text(encoding="utf-8"))
        summary_path = QUALITY_DIR / "summaries" / f"{model_key}__{method}.json"
        if summary_path.is_file():
            quality_summaries[method] = json.loads(summary_path.read_text(encoding="utf-8"))
        micro[method] = _micro_metrics(phase, model_key, method)
        service[method] = _service_metrics(phase, model_key, method)
    if set(artifact_records) != set(methods):
        raise RuntimeError("Analysis requires all four audited artifact manifests")
    if phase != "engineering" and (
        set(quality_summaries) != set(methods)
        or any(row.get("complete") is not True for row in quality_summaries.values())
    ):
        raise RuntimeError("Test-bearing analysis requires four complete quality summaries")
    if phase != "engineering":
        for method, summary in quality_summaries.items():
            sample = QUALITY_DIR / "samples" / f"{model_key}__{method}__gsm8k1319.jsonl"
            if (
                not sample.is_file()
                or summary.get("sample_file_sha256") != sha256_file(sample)
            ):
                raise RuntimeError(f"Stale quality summary for {model_key}/{method}")
    input_paths = [path for paths in block_files.values() for path in paths]
    input_paths.extend(
        MANIFEST_DIR / "artifacts" / model_key / f"{method}.json"
        for method in methods
    )
    if phase != "engineering":
        input_paths.extend(
            QUALITY_DIR / "summaries" / f"{model_key}__{method}.json"
            for method in methods
        )
        input_paths.extend(
            QUALITY_DIR / "samples" / f"{model_key}__{method}__gsm8k1319.jsonl"
            for method in methods
        )
    if any(not path.is_file() for path in input_paths):
        raise RuntimeError("Analysis input registration contains a missing file")

    accuracy_effects = {}
    sg_samples = QUALITY_DIR / "samples" / f"{model_key}__sg__gsm8k1319.jsonl"
    if sg_samples.is_file():
        for comparator in ("q4", "q5"):
            other = QUALITY_DIR / "samples" / f"{model_key}__{comparator}__gsm8k1319.jsonl"
            if other.is_file():
                accuracy_effects[f"sg_minus_{comparator}"] = paired_accuracy(sg_samples, other)

    performance_ratios = {"micro": {}, "service": {}}
    for kind, source in (("micro", micro), ("service", service)):
        for comparator in ("q4", "q5"):
            common_metrics = sorted(set(source["sg"]) & set(source[comparator]))
            performance_ratios[kind][f"sg_over_{comparator}"] = {
                metric: paired_ratio(source["sg"][metric], source[comparator][metric])
                for metric in common_metrics
                if set(source["sg"][metric]) & set(source[comparator][metric])
            }

    primary_service_metrics = (
        "c1__aggregate_tokens_per_second",
        "c4__aggregate_tokens_per_second",
    )
    primary_precision_records = {
        f"{comparison}/{metric}": performance_ratios["service"][comparison][metric]
        for comparison in ("sg_over_q4", "sg_over_q5")
        for metric in primary_service_metrics
        if metric in performance_ratios["service"][comparison]
    }
    precision_target_reached = (
        len(primary_precision_records) == 4
        and all(
            record["two_percent_precision_reached"]
            for record in primary_precision_records.values()
        )
    )
    formal_precision = {
        "primary_endpoints": list(primary_precision_records),
        "required_primary_endpoints": [
            f"{comparison}/{metric}"
            for comparison in ("sg_over_q4", "sg_over_q5")
            for metric in primary_service_metrics
        ],
        "two_percent_ci_half_width_target_reached": precision_target_reached,
        "current_blocks": len(reference_blocks),
        "maximum_blocks": FORMAL_MAX_BLOCKS,
        "additional_blocks_required": (
            phase == "formal"
            and not precision_target_reached
            and len(reference_blocks) < FORMAL_MAX_BLOCKS
        ),
        "maximum_exhausted_without_target": (
            phase == "formal"
            and not precision_target_reached
            and len(reference_blocks) == FORMAL_MAX_BLOCKS
        ),
    }

    record = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "run_phase": phase,
        "artifact_accounting": {
            method: {
                key: value
                for key, value in item.items()
                if key
                in {
                    "artifact_sha256",
                    "artifact_bytes",
                    "packed_eligible_bits_per_weight",
                    "whole_stored_payload_bits_per_element",
                    "original_gptq_logical_bits_per_weight",
                }
            }
            for method, item in artifact_records.items()
        },
        "quality": quality_summaries,
        "paired_accuracy_effects": accuracy_effects,
        "performance_ratios": performance_ratios,
        "formal_precision": formal_precision,
        "inference_unit": "independent process block",
        "within_process_repetitions_are_not_independent": True,
        "process_blocks": sorted(reference_blocks),
        "input_file_sha256": {
            str(path): sha256_file(path)
            for path in input_paths
        },
        "complete": True,
    }
    output = BENCH_DIR.parent / "analysis" / phase / f"{model_key}.json"
    atomic_write_json(output, record)
    _write_markdown(output.with_suffix(".md"), record)
    return record


def _write_markdown(path: Path, record: dict) -> None:
    lines = [
        f"# deployment-gguf-v1: {record['model_key']}",
        "",
        "This extension measures allocation-policy transfer on one pinned GGUF/CUDA stack. It does not rename or replace the original GPTQ results.",
        "",
        "## Artifact accounting",
        "",
        "| method | exact bytes | eligible packed bpw | whole stored payload bits/element | original logical bpw |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, row in record["artifact_accounting"].items():
        lines.append(
            f"| {method} | {row['artifact_bytes']} | {row['packed_eligible_bits_per_weight']:.4f} | "
            f"{row['whole_stored_payload_bits_per_element']:.4f} | {row['original_gptq_logical_bits_per_weight']:.4f} |"
        )
    lines.extend(["", "## Quality", "", "| method | flexible accuracy | strict accuracy | delimiter coverage |", "|---|---:|---:|---:|"])
    for method, row in record["quality"].items():
        lines.append(
            f"| {method} | {row['flexible_accuracy']:.2f} | {row['strict_accuracy']:.2f} | {100 * row['delimiter_coverage']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Performance ratios and confidence intervals are stored in the adjacent JSON. Ratios are paired by independent process block; within-process repetitions are summarized, not counted as independent observations.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

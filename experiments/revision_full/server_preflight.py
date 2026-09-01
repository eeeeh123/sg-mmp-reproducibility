"""Fail-fast hardware, environment, model, data, disk, and protocol checks."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.revision_full.protocol import (
    GROUP_SIZE,
    MAX_CONCURRENT_RAM_BUILDERS,
    MIN_AVAILABLE_RAM_GIB,
    MODEL_SPECS,
    OUT,
    ROOT,
    STATE_DIR,
)
from experiments.revision_full.download_models import MODEL_SOURCES
from experiments.revision_full.download_core_datasets import (
    MANIFEST_PATH as DATASET_MANIFEST_PATH,
    PANEL_TASKS,
    snapshot_sha256 as dataset_snapshot_sha256,
    verify_cache_files,
)
from experiments.revision_full.readiness import preflight_errors


EXPECTED_VERSIONS = {
    "torch": "2.11.0+cu128",
    "transformers": "5.8.0",
    "datasets": "4.8.5",
    "lm_eval": "0.4.11",
    "pyarrow": "24.0.0",
    "numpy": "2.4.3",
    "scipy": "1.17.1",
    "accelerate": "1.13.0",
    "safetensors": "0.7.0",
    "sentencepiece": "0.2.1",
    "matplotlib": "3.10.8",
    "peft": "0.19.1",
}
MIN_GPU_GIB = 16
RECOMMENDED_GPU_GIB = 23
MIN_PERSISTENT_FREE_GIB = 30
RECOMMENDED_PERSISTENT_FREE_GIB = 60


def storage_thresholds(
    state_peak_estimates: list[float], concurrent_models: int
) -> dict:
    concurrent_peaks = sorted(state_peak_estimates, reverse=True)[:concurrent_models]
    estimated = sum(concurrent_peaks) or 12.0
    minimum_state = max(16, math.ceil(1.25 * estimated + 2))
    recommended_state = max(24, math.ceil(1.5 * estimated + 5))
    return {
        "concurrent_peaks": concurrent_peaks,
        "estimated_state_peak_gib": estimated,
        "minimum_state_free_gib": minimum_state,
        "recommended_state_free_gib": recommended_state,
        "minimum_shared_free_gib": MIN_PERSISTENT_FREE_GIB + minimum_state,
        "recommended_shared_free_gib": RECOMMENDED_PERSISTENT_FREE_GIB
        + recommended_state,
    }


def ram_thresholds(
    concurrent_models: int,
    max_concurrent_ram_builders: int,
    configured_min_available_gib: float = MIN_AVAILABLE_RAM_GIB,
) -> dict:
    """Capacity gates for full-concurrency and low-RAM staggered execution."""
    if max_concurrent_ram_builders not in {1, concurrent_models}:
        raise ValueError(
            "max_concurrent_ram_builders must be 1 or equal concurrent_models"
        )
    if max_concurrent_ram_builders == 1:
        return {
            "mode": "serialized_ram_builders_with_parallel_gpu_evaluation",
            "minimum_total_gib": max(30, 22 + 4 * concurrent_models),
            "recommended_total_gib": max(48, 32 + 8 * concurrent_models),
            "minimum_available_gib": max(
                configured_min_available_gib,
                24,
                16 + 4 * concurrent_models,
            ),
            "recommended_available_gib": max(32, 24 + 4 * concurrent_models),
        }
    return {
        "mode": "concurrent_ram_builders",
        "minimum_total_gib": max(60, 32 * concurrent_models),
        "recommended_total_gib": max(90, 48 * concurrent_models),
        "minimum_available_gib": max(48, 24 * concurrent_models),
        "recommended_available_gib": max(72, 36 * concurrent_models),
    }


def estimate_stream_state_peak_gib(model_dir: Path, weight_bytes: int) -> float:
    """Estimate one W4/W5/W6/W8 bank plus one materialized int8-code state."""
    suffixes = tuple(
        f"{name}.weight"
        for name in (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        )
    )
    try:
        from safetensors import safe_open

        shapes = []
        for path in model_dir.glob("*.safetensors"):
            with safe_open(path, framework="pt", device="cpu") as stream:
                for key in stream.keys():
                    if key.endswith(suffixes):
                        shapes.append(stream.get_slice(key).get_shape())
        if shapes:
            bank_bytes = 0
            state_bytes = 0
            for out_features, in_features in shapes:
                parameters = out_features * in_features
                groups = math.ceil(in_features / GROUP_SIZE)
                bank_bytes += (
                    4 * parameters
                    + 24 * out_features * groups
                    + 4 * out_features
                )
                state_bytes += parameters + 8 * out_features * groups
            return (bank_bytes + state_bytes) / 1024**3
    except (ImportError, OSError, RuntimeError, ValueError):
        pass
    return 2.5 * weight_bytes / 1024**3


def version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "unknown"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def system_memory_gib() -> tuple[float | None, float | None]:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None, None
    values = {}
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        key = line.split(":", 1)[0]
        if key in {"MemTotal", "MemAvailable"}:
            values[key] = int(line.split()[1]) / 1024**2
    return values.get("MemTotal"), values.get("MemAvailable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-gpus", type=int, default=1)
    parser.add_argument("--concurrent-models", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.concurrent_models <= len(MODEL_SPECS):
        raise SystemExit("--concurrent-models must be between 1 and 4")
    if args.expected_gpus < args.concurrent_models:
        raise SystemExit("--expected-gpus must be at least --concurrent-models")
    if MAX_CONCURRENT_RAM_BUILDERS not in {1, args.concurrent_models}:
        raise SystemExit(
            "REVISION_FULL_MAX_CONCURRENT_RAM_BUILDERS must be 1 or equal "
            "--concurrent-models"
        )
    errors: list[str] = []
    warnings: list[str] = []
    if sys.version_info[:2] != (3, 12):
        errors.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} is active; "
            "the tested server environment requires Python 3.12"
        )
    packages = {}
    for module_name, expected in EXPECTED_VERSIONS.items():
        try:
            actual = version_of(module_name)
            packages[module_name] = actual
            if actual != expected:
                errors.append(f"{module_name}={actual}, required lock is {expected}")
        except Exception as exc:
            errors.append(f"cannot import {module_name}: {exc}")

    gpu_rows = []
    try:
        import torch

        if not torch.cuda.is_available():
            errors.append("torch.cuda.is_available() is false")
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            memory_gib = props.total_memory / 1024**3
            gpu_rows.append(
                {"index": index, "name": props.name, "memory_gib": round(memory_gib, 2)}
            )
            if memory_gib < MIN_GPU_GIB:
                errors.append(f"GPU {index} has only {memory_gib:.1f} GiB")
            elif memory_gib < RECOMMENDED_GPU_GIB:
                warnings.append(f"GPU {index} is below the recommended 24-GB class")
        if torch.cuda.device_count() < args.expected_gpus:
            errors.append(
                f"only {torch.cuda.device_count()} CUDA devices are visible; "
                f"the launch plan expects {args.expected_gpus}"
            )
    except Exception as exc:
        errors.append(f"CUDA inspection failed: {exc}")

    model_rows = []
    state_peak_estimates = []
    snapshot_manifest_path = OUT / "model_snapshot_manifest.json"
    snapshot_models = {}
    if not snapshot_manifest_path.exists():
        errors.append(
            "missing model snapshot provenance; use experiments/revision_full/download_models.py"
        )
    else:
        try:
            snapshot_models = json.loads(
                snapshot_manifest_path.read_text(encoding="utf-8")
            ).get("models", {})
        except (OSError, ValueError) as exc:
            errors.append(f"invalid model snapshot manifest: {exc}")
    for model_key, spec in MODEL_SPECS.items():
        model_dir = Path(spec["path"])
        weights = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        size = sum(path.stat().st_size for path in weights)
        stream_state_peak_gib = estimate_stream_state_peak_gib(model_dir, size)
        model_rows.append(
            {
                "model": model_key,
                "path": str(model_dir),
                "weight_gib": round(size / 1024**3, 3),
                "stream_state_peak_gib": round(stream_state_peak_gib, 2),
            }
        )
        state_peak_estimates.append(stream_state_peak_gib)
        if not (model_dir / "config.json").exists() or size < 100 * 1024**2:
            errors.append(f"model is missing or incomplete: {model_key} at {model_dir}")
        snapshot = snapshot_models.get(model_key, {})
        expected_source = MODEL_SOURCES[model_key]
        revision = str(snapshot.get("resolved_revision", ""))
        if snapshot.get("repo_id") != expected_source["repo_id"] or len(revision) != 40:
            errors.append(f"missing pinned upstream revision for {model_key}")
        file_records = snapshot.get("weight_file_records", [])
        if not file_records:
            errors.append(
                f"model manifest lacks file hashes for {model_key}; rerun download_models.py"
            )
        else:
            for record in file_records:
                path = model_dir / str(record.get("name", ""))
                if not path.exists():
                    errors.append(f"missing frozen model file: {path}")
                    continue
                if path.stat().st_size != int(record.get("bytes", -1)):
                    errors.append(f"model file size changed: {path}")
                    continue
                if sha256(path) != record.get("sha256"):
                    errors.append(f"model file hash changed: {path}")

    datasets = {}

    dataset_manifest = {}
    if not DATASET_MANIFEST_PATH.exists():
        errors.append(
            "missing frozen core/panel dataset manifest; run download_core_datasets.py"
        )
    else:
        try:
            dataset_manifest = json.loads(
                DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
            )
            if dataset_manifest.get("schema_version") != 2:
                errors.append("dataset snapshot manifest schema is not v2")
            if dataset_manifest.get("snapshot_sha256") != dataset_snapshot_sha256(
                dataset_manifest
            ):
                errors.append("dataset snapshot manifest identity hash is invalid")
            task_inventory = dataset_manifest.get("panels", {}).get("tasks", {})
            if set(task_inventory) != set(PANEL_TASKS):
                errors.append("frozen panel task inventory is incomplete")
            if any(
                int(task_inventory.get(task, {}).get("evaluation_docs", 0)) <= 0
                for task in PANEL_TASKS
            ):
                errors.append("one or more frozen panel tasks have no evaluation documents")
            for dataset_key, record in dataset_manifest.get("core", {}).items():
                datasets[dataset_key] = [
                    item.get("path") for item in record.get("cache_files", [])
                ]
            errors.extend(verify_cache_files(dataset_manifest))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"invalid dataset snapshot manifest: {exc}")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get(
        "HF_DATASETS_OFFLINE"
    ) != "1":
        errors.append(
            "formal execution must export HF_HUB_OFFLINE=1 and HF_DATASETS_OFFLINE=1 "
            "after dataset/model staging"
        )

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    persistent_disk = shutil.disk_usage(ROOT)
    state_disk = shutil.disk_usage(STATE_DIR)
    persistent_free_gib = persistent_disk.free / 1024**3
    state_free_gib = state_disk.free / 1024**3
    storage = storage_thresholds(state_peak_estimates, args.concurrent_models)
    concurrent_peaks = storage["concurrent_peaks"]
    estimated_state_peak_gib = storage["estimated_state_peak_gib"]
    min_state_free_gib = storage["minimum_state_free_gib"]
    recommended_state_free_gib = storage["recommended_state_free_gib"]
    same_filesystem = os.stat(ROOT).st_dev == os.stat(STATE_DIR).st_dev
    if same_filesystem:
        minimum = storage["minimum_shared_free_gib"]
        recommended = storage["recommended_shared_free_gib"]
        if persistent_free_gib < minimum:
            errors.append(
                f"only {persistent_free_gib:.1f} GiB free on the shared results/state filesystem; "
                f"streaming execution requires at least {minimum} GiB"
            )
        elif persistent_free_gib < recommended:
            warnings.append(
                f"{persistent_free_gib:.1f} GiB free; about {recommended} GiB is recommended "
                "for streaming states plus persistent results"
            )
    else:
        if persistent_free_gib < MIN_PERSISTENT_FREE_GIB:
            errors.append(
                f"only {persistent_free_gib:.1f} GiB free for persistent results; "
                f"at least {MIN_PERSISTENT_FREE_GIB} GiB is required"
            )
        elif persistent_free_gib < RECOMMENDED_PERSISTENT_FREE_GIB:
            warnings.append(
                f"{persistent_free_gib:.1f} GiB free for persistent results; "
                f"{RECOMMENDED_PERSISTENT_FREE_GIB} GiB is recommended"
            )
        if state_free_gib < min_state_free_gib:
            errors.append(
                f"only {state_free_gib:.1f} GiB free in {STATE_DIR}; "
                f"the largest streaming state needs at least {min_state_free_gib} GiB"
            )
        elif state_free_gib < recommended_state_free_gib:
            warnings.append(
                f"{state_free_gib:.1f} GiB free in {STATE_DIR}; "
                f"{recommended_state_free_gib} GiB is recommended"
            )

    ram_gib, available_ram_gib = system_memory_gib()
    ram = ram_thresholds(
        args.concurrent_models, MAX_CONCURRENT_RAM_BUILDERS
    )
    minimum_ram_gib = ram["minimum_total_gib"]
    recommended_ram_gib = ram["recommended_total_gib"]
    if ram_gib is not None and ram_gib < minimum_ram_gib:
        errors.append(
            f"only {ram_gib:.1f} GiB system RAM; at least {minimum_ram_gib} GiB is required"
        )
    elif ram_gib is not None and ram_gib < recommended_ram_gib:
        warnings.append(
            f"{ram_gib:.1f} GiB RAM; {recommended_ram_gib} GiB is recommended"
        )
    if (
        available_ram_gib is not None
        and available_ram_gib < ram["minimum_available_gib"]
    ):
        errors.append(
            f"only {available_ram_gib:.1f} GiB system RAM is currently available; "
            f"at least {ram['minimum_available_gib']} GiB is required before launch"
        )
    elif (
        available_ram_gib is not None
        and available_ram_gib < ram["recommended_available_gib"]
    ):
        warnings.append(
            f"{available_ram_gib:.1f} GiB RAM is currently available; "
            f"{ram['recommended_available_gib']} GiB is recommended"
        )
    if MAX_CONCURRENT_RAM_BUILDERS == 1:
        if MIN_AVAILABLE_RAM_GIB < ram["minimum_available_gib"]:
            errors.append(
                "REVISION_FULL_MIN_AVAILABLE_RAM_GIB is below the preflight safety floor"
            )
        warnings.append(
            "low-RAM mode is active: activation-heavy screen/precision builders are "
            "serialized, while GPU evaluation remains parallel"
        )

    errors.extend(preflight_errors())
    report = {
        "ready": not errors,
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "ram_gib": None if ram_gib is None else round(ram_gib, 1),
            "available_ram_gib": (
                None if available_ram_gib is None else round(available_ram_gib, 1)
            ),
            "persistent_free_disk_gib": round(persistent_free_gib, 1),
            "state_free_disk_gib": round(state_free_gib, 1),
            "state_dir": str(STATE_DIR),
            "state_filesystem_is_persistent_filesystem": same_filesystem,
            "estimated_concurrent_stream_state_peak_gib": round(
                estimated_state_peak_gib, 2
            ),
            "expected_gpus": args.expected_gpus,
            "concurrent_models": args.concurrent_models,
            "ram_execution_mode": ram["mode"],
            "max_concurrent_ram_builders": MAX_CONCURRENT_RAM_BUILDERS,
            "concurrent_model_state_peaks_gib": [
                round(value, 2) for value in concurrent_peaks
            ],
            "minimum_state_free_gib": min_state_free_gib,
            "recommended_state_free_gib": recommended_state_free_gib,
            "minimum_shared_free_gib": storage["minimum_shared_free_gib"],
            "recommended_shared_free_gib": storage[
                "recommended_shared_free_gib"
            ],
            "minimum_ram_gib": minimum_ram_gib,
            "recommended_ram_gib": recommended_ram_gib,
            "minimum_available_ram_gib": ram["minimum_available_gib"],
            "recommended_available_ram_gib": ram["recommended_available_gib"],
        },
        "gpus": gpu_rows,
        "packages": packages,
        "models": model_rows,
        "model_snapshot_manifest": str(snapshot_manifest_path),
        "dataset_snapshot_manifest": str(DATASET_MANIFEST_PATH),
        "datasets": datasets,
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

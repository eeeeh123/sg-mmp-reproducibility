"""Fail-fast hardware, environment, model, data, disk, and protocol checks."""

from __future__ import annotations

import importlib
import json
import math
import os
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.fix_gsm8k_500.direct_eval import find_arrow
from experiments.revision_full.protocol import (
    GROUP_SIZE,
    MODEL_SPECS,
    OUT,
    ROOT,
    STATE_DIR,
)
from experiments.revision_full.download_models import MODEL_SOURCES
from experiments.revision_full.readiness import preflight_errors
from ptq.data import _latest_arrow


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
}
MIN_GPU_GIB = 16
RECOMMENDED_GPU_GIB = 23
MIN_PERSISTENT_FREE_GIB = 30
RECOMMENDED_PERSISTENT_FREE_GIB = 60
MIN_RAM_GIB = 60
RECOMMENDED_RAM_GIB = 90


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


def system_ram_gib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024**2
    return None


def main() -> None:
    errors: list[str] = []
    warnings: list[str] = []
    packages = {}
    for module_name, expected in EXPECTED_VERSIONS.items():
        try:
            actual = version_of(module_name)
            packages[module_name] = actual
            if actual != expected:
                warnings.append(f"{module_name}={actual}, tested lock is {expected}")
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

    datasets = {}
    for split in ("train", "test"):
        try:
            datasets[f"gsm8k_{split}"] = str(find_arrow(split))
        except FileNotFoundError as exc:
            errors.append(str(exc))
    wikitext = _latest_arrow("wikitext-train.arrow")
    if wikitext is None:
        errors.append("WikiText train Arrow cache is missing")
    else:
        datasets["wikitext_train"] = str(wikitext)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    persistent_disk = shutil.disk_usage(ROOT)
    state_disk = shutil.disk_usage(STATE_DIR)
    persistent_free_gib = persistent_disk.free / 1024**3
    state_free_gib = state_disk.free / 1024**3
    estimated_state_peak_gib = max(state_peak_estimates, default=12.0)
    min_state_free_gib = max(16, math.ceil(1.25 * estimated_state_peak_gib + 2))
    recommended_state_free_gib = max(
        24, math.ceil(1.5 * estimated_state_peak_gib + 5)
    )
    same_filesystem = os.stat(ROOT).st_dev == os.stat(STATE_DIR).st_dev
    if same_filesystem:
        minimum = MIN_PERSISTENT_FREE_GIB + min_state_free_gib
        recommended = RECOMMENDED_PERSISTENT_FREE_GIB + recommended_state_free_gib
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

    ram_gib = system_ram_gib()
    if ram_gib is not None and ram_gib < MIN_RAM_GIB:
        errors.append(
            f"only {ram_gib:.1f} GiB system RAM; at least {MIN_RAM_GIB} GiB is required"
        )
    elif ram_gib is not None and ram_gib < RECOMMENDED_RAM_GIB:
        warnings.append(
            f"{ram_gib:.1f} GiB RAM; {RECOMMENDED_RAM_GIB} GiB is recommended"
        )

    errors.extend(preflight_errors())
    report = {
        "ready": not errors,
        "system": {
            "platform": platform.platform(),
            "python": sys.version,
            "cpu_count": os.cpu_count(),
            "ram_gib": None if ram_gib is None else round(ram_gib, 1),
            "persistent_free_disk_gib": round(persistent_free_gib, 1),
            "state_free_disk_gib": round(state_free_gib, 1),
            "state_dir": str(STATE_DIR),
            "state_filesystem_is_persistent_filesystem": same_filesystem,
            "estimated_largest_stream_state_peak_gib": round(
                estimated_state_peak_gib, 2
            ),
            "minimum_state_free_gib": min_state_free_gib,
        },
        "gpus": gpu_rows,
        "packages": packages,
        "models": model_rows,
        "model_snapshot_manifest": str(snapshot_manifest_path),
        "datasets": datasets,
        "warnings": warnings,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

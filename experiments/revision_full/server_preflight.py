"""Fail-fast hardware, environment, model, data, disk, and protocol checks."""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")

from experiments.fix_gsm8k_500.direct_eval import find_arrow
from experiments.revision_full.protocol import MODEL_SPECS, OUT, ROOT
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
MIN_FREE_DISK_GIB = 400
RECOMMENDED_FREE_DISK_GIB = 500
MIN_RAM_GIB = 60
RECOMMENDED_RAM_GIB = 90


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
        model_rows.append(
            {"model": model_key, "path": str(model_dir), "weight_gib": round(size / 1024**3, 3)}
        )
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

    disk = shutil.disk_usage(ROOT)
    free_disk_gib = disk.free / 1024**3
    if free_disk_gib < MIN_FREE_DISK_GIB:
        errors.append(
            f"only {free_disk_gib:.1f} GiB free; at least {MIN_FREE_DISK_GIB} GiB is required"
        )
    elif free_disk_gib < RECOMMENDED_FREE_DISK_GIB:
        warnings.append(
            f"{free_disk_gib:.1f} GiB free; {RECOMMENDED_FREE_DISK_GIB} GiB is recommended for retained states"
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
            "free_disk_gib": round(free_disk_gib, 1),
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

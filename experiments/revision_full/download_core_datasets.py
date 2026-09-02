"""Pre-stage and fingerprint every dataset needed by revision-full-v4."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import load_dataset

from experiments.revision_full.storage_layout import require_managed_storage


OUT = ROOT / "experiments" / "revision_full" / "outputs"
MANIFEST_PATH = OUT / "dataset_snapshot_manifest.json"
CORE_REQUESTS = (
    ("openai/gsm8k", "main", ("train", "test")),
    ("Salesforce/wikitext", "wikitext-2-raw-v1", ("train",)),
)
PANEL_TASKS = (
    "arc_challenge",
    "hellaswag",
    "mmlu",
    "mmlu_high_school_mathematics",
    "svamp",
    "asdiv_gen",
    "hendrycks_math500",
    "truthfulqa_gen",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_identity(manifest: dict) -> dict:
    """Return the path/time-independent dataset identity used by result locks."""

    def cache_identity(records: list[dict]) -> list[dict]:
        return sorted(
            [
                {
                    "bytes": int(record["bytes"]),
                    "sha256": str(record["sha256"]),
                }
                for record in records
            ],
            key=lambda row: (row["sha256"], row["bytes"]),
        )

    return {
        "schema_version": int(manifest.get("schema_version", -1)),
        "core": {
            name: {
                "splits": record.get("splits", {}),
                "cache_files": cache_identity(record.get("cache_files", [])),
            }
            for name, record in sorted(manifest.get("core", {}).items())
        },
        "panels": {
            "tasks": manifest.get("panels", {}).get("tasks", {}),
            "datasets": manifest.get("panels", {}).get("datasets", {}),
            "cache_files": cache_identity(
                manifest.get("panels", {}).get("cache_files", [])
            ),
        },
    }


def snapshot_sha256(manifest: dict) -> str:
    payload = json.dumps(
        snapshot_identity(manifest), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def cache_file_records(datasets_by_name: dict[str, object]) -> list[dict]:
    paths: set[Path] = set()
    for dataset in datasets_by_name.values():
        for record in getattr(dataset, "cache_files", []) or []:
            filename = record.get("filename")
            if filename:
                paths.add(Path(filename).resolve())
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(paths, key=str)
    ]


def dataset_dict_record(dataset) -> dict:
    splits = dict(dataset) if hasattr(dataset, "items") else {"data": dataset}
    return {
        "splits": {
            name: {
                "rows": len(split),
                "fingerprint": getattr(split, "_fingerprint", None),
                "columns": list(getattr(split, "column_names", []) or []),
            }
            for name, split in splits.items()
        },
        "cache_files": cache_file_records(splits),
    }


def task_objects(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from task_objects(child)
    elif hasattr(value, "dataset"):
        yield value


def evaluation_doc_count(task) -> int:
    if task.has_test_docs():
        return len(task.test_docs())
    if task.has_validation_docs():
        return len(task.validation_docs())
    if task.has_training_docs():
        return len(task.training_docs())
    raise RuntimeError(f"Task {getattr(task, 'config', task)!r} has no evaluation split")


def panel_inventory() -> dict:
    from lm_eval.tasks import TaskManager, get_task_dict

    manager = TaskManager(
        include_path=str(ROOT / "experiments" / "fix_svamp_ood"),
        verbosity="ERROR",
    )
    inventory = {}
    all_datasets = {}
    for requested in PANEL_TASKS:
        # Group keys (for example ``mmlu``) are ConfigurableGroup objects rather
        # than the requested string, so resolve and walk each request separately.
        task_map = get_task_dict([requested], task_manager=manager)
        leaves = list(task_objects(task_map))
        if not leaves:
            raise RuntimeError(f"lm-eval task {requested!r} has no executable leaves")
        leaf_rows = []
        for ordinal, task in enumerate(leaves):
            key = str(getattr(task.config, "task", f"{requested}:{ordinal}"))
            count = evaluation_doc_count(task)
            if count <= 0:
                raise RuntimeError(f"lm-eval task {key!r} has no evaluation documents")
            leaf_rows.append({"task": key, "evaluation_docs": count})
            all_datasets[f"{requested}:{key}"] = task.dataset
        inventory[requested] = {
            "leaves": leaf_rows,
            "evaluation_docs": sum(row["evaluation_docs"] for row in leaf_rows),
        }

    dataset_records = {}
    cache_paths: dict[str, dict] = {}
    for key, dataset in all_datasets.items():
        record = dataset_dict_record(dataset)
        dataset_records[key] = {"splits": record["splits"]}
        for item in record["cache_files"]:
            cache_paths[item["path"]] = item
    return {
        "tasks": inventory,
        "datasets": dataset_records,
        "cache_files": [cache_paths[key] for key in sorted(cache_paths)],
    }


def verify_cache_files(manifest: dict) -> list[str]:
    errors = []
    records = []
    for dataset in manifest.get("core", {}).values():
        records.extend(dataset.get("cache_files", []))
    records.extend(manifest.get("panels", {}).get("cache_files", []))
    for record in records:
        path = Path(record.get("path", ""))
        if not path.exists():
            errors.append(f"missing frozen dataset cache file: {path}")
            continue
        if path.stat().st_size != int(record.get("bytes", -1)):
            errors.append(f"dataset cache size changed: {path}")
            continue
        if sha256(path) != record.get("sha256"):
            errors.append(f"dataset cache hash changed: {path}")
    return errors


def main() -> None:
    require_managed_storage(ROOT)
    core = {}
    for dataset_name, config, splits in CORE_REQUESTS:
        for split in splits:
            loaded = load_dataset(dataset_name, config, split=split)
            key = f"{dataset_name}/{config}/{split}"
            core[key] = dataset_dict_record(loaded)
            print(f"cached {key}: {len(loaded)} rows", flush=True)

    manifest = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "offline_execution_required": True,
        "core": core,
        "panels": panel_inventory(),
        "environment": {
            "hf_home": os.environ.get("HF_HOME"),
            "hf_datasets_cache": os.environ.get("HF_DATASETS_CACHE"),
        },
    }
    manifest["snapshot_sha256"] = snapshot_sha256(manifest)
    OUT.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, MANIFEST_PATH)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

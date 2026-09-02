"""Download and pin the four revision models directly on the server.

Model weights are intentionally not stored in Git. The first successful run
resolves each requested Hugging Face revision to an immutable commit SHA and
writes ``experiments/revision_full/outputs/model_snapshot_manifest.json``. Interrupted downloads resume
against that same SHA on the next run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, get_token, snapshot_download

from experiments.revision_full.storage_layout import require_managed_storage


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = (
    ROOT / "experiments" / "revision_full" / "outputs" / "model_snapshot_manifest.json"
)
MODEL_SOURCES = {
    "qwen05": {
        "repo_id": "Qwen/Qwen2.5-0.5B",
        "local_dir": "Qwen2.5-0.5B",
        "gated": False,
    },
    "qwen15": {
        "repo_id": "Qwen/Qwen2.5-1.5B",
        "local_dir": "Qwen2.5-1.5B",
        "gated": False,
    },
    "smollm": {
        "repo_id": "HuggingFaceTB/SmolLM2-1.7B",
        "local_dir": "SmolLM-1.7B",
        "gated": False,
    },
    "gemma2": {
        "repo_id": "google/gemma-2-2b-it",
        "local_dir": "gemma-2-2b-it",
        "gated": True,
    },
}
ALLOW_PATTERNS = [
    "*.json",
    "*.jinja",
    "*.model",
    "*.txt",
    "*.safetensors",
]
IGNORE_PATTERNS = [
    "*.bin",
    "*.gguf",
    "*.h5",
    "*.msgpack",
    "onnx/*",
    "original/*",
]


def stable_model_record(record: dict) -> dict:
    """Drop download-time fields while retaining every model identity field."""
    required = [
        "repo_id",
        "resolved_revision",
        "local_directory",
        "weight_bytes",
        "weight_file_records",
    ]
    missing = [key for key in required if key not in record]
    if missing:
        raise RuntimeError(f"Model manifest record lacks {missing}")
    return {key: record[key] for key in required}


def read_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"schema_version": 1, "models": {}}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def write_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST_PATH.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, MANIFEST_PATH)


def parse_revisions(values: list[str]) -> dict[str, str]:
    revisions = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Revision must use MODEL=REVISION, got {value!r}")
        model_key, revision = value.split("=", 1)
        if model_key not in MODEL_SOURCES or not revision:
            raise ValueError(f"Invalid revision override {value!r}")
        revisions[model_key] = revision
    return revisions


def verify_local_model(path: Path) -> dict:
    weights = sorted(path.glob("*.safetensors"))
    weight_bytes = sum(item.stat().st_size for item in weights)
    if not (path / "config.json").exists():
        raise RuntimeError(f"Missing config.json in {path}")
    if not weights or weight_bytes < 100 * 1024**2:
        raise RuntimeError(f"Missing or incomplete safetensors weights in {path}")
    if not any((path / name).exists() for name in ["tokenizer.json", "tokenizer.model"]):
        raise RuntimeError(f"Missing tokenizer assets in {path}")
    files = []
    for item in weights:
        digest = hashlib.sha256()
        with item.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        files.append(
            {
                "name": item.name,
                "bytes": item.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return {
        "weight_files": [item.name for item in weights],
        "weight_file_records": files,
        "weight_bytes": weight_bytes,
    }


def download_one(
    model_key: str,
    requested_revision: str | None,
    endpoint: str,
    token: str | None,
    manifest: dict,
) -> None:
    source = MODEL_SOURCES[model_key]
    previous = manifest["models"].get(model_key, {})
    revision = (
        requested_revision
        or previous.get("resolved_revision")
        or previous.get("requested_revision")
        or "main"
    )
    if source["gated"] and not token:
        raise RuntimeError(
            "Gemma requires access approval and authentication. Accept the license at "
            "https://huggingface.co/google/gemma-2-2b-it and run `hf auth login` "
            "or export HF_TOKEN before retrying."
        )

    api = HfApi(endpoint=endpoint, token=token)
    info = api.model_info(source["repo_id"], revision=revision, token=token)
    resolved_revision = str(info.sha)
    target = ROOT / "models" / source["local_dir"]
    print(
        f"[download] {model_key}: {source['repo_id']}@{resolved_revision} -> {target}",
        flush=True,
    )
    snapshot_download(
        repo_id=source["repo_id"],
        revision=resolved_revision,
        local_dir=target,
        token=token,
        endpoint=endpoint,
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        max_workers=4,
    )
    local = verify_local_model(target)
    manifest["models"][model_key] = {
        "repo_id": source["repo_id"],
        "requested_revision": revision,
        "resolved_revision": resolved_revision,
        "local_directory": str(target.relative_to(ROOT)).replace("\\", "/"),
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **local,
    }
    write_manifest(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SOURCES,
        default=list(MODEL_SOURCES),
    )
    parser.add_argument(
        "--revision",
        action="append",
        default=[],
        metavar="MODEL=REVISION",
        help="Optional branch, tag, or commit. Resolved commit SHAs are recorded.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
    )
    args = parser.parse_args()

    require_managed_storage(ROOT)
    revisions = parse_revisions(args.revision)
    token = get_token() or os.environ.get("HF_TOKEN")
    manifest = read_manifest()
    manifest["endpoint"] = args.endpoint
    for model_key in args.models:
        download_one(
            model_key,
            revisions.get(model_key),
            args.endpoint,
            token,
            manifest,
        )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

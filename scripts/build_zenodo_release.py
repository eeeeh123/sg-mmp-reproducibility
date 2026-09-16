"""Build and validate the multi-file Zenodo v2 release delivery."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from build_zenodo_zip import ROOT, release_version, sha256, verify_manifest


FORBIDDEN_WEIGHT_SUFFIXES = {".bin", ".gguf", ".pt", ".pth", ".safetensors"}
REVISION_REQUIRED = {
    "experiments/revision_full/outputs/analysis_full.json",
    "experiments/revision_full/outputs/analysis_full.md",
    "experiments/revision_full/outputs/protocol_lock.json",
    "experiments/revision_full/outputs/model_snapshot_manifest.json",
    "experiments/revision_full/outputs/dataset_snapshot_manifest.json",
    "experiments/revision_full/outputs/tacq/frozen_manifest.json",
}
TACQ_REQUIRED = {
    "experiments/revision_full/outputs/analysis_full.json",
    "experiments/revision_full/outputs/tacq/frozen_manifest.json",
    "logs/readiness_tacq_final_0907.log",
    "logs/readiness_resubmission_final_0907.log",
}
DEPLOYMENT_REQUIRED = {
    "README_FINAL.md",
    "unified_10_block_review.json",
    "unified_10_block_review_zh.md",
    "status/readiness__formal.json",
    "status/readiness__value.json",
}


def git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def require_release_checkout(version: str) -> tuple[str, str]:
    dirty = git_text("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise RuntimeError("Tracked worktree is dirty; commit release files first.")
    commit = git_text("rev-parse", "HEAD")
    tag = git_text("describe", "--exact-match", "--tags", "HEAD")
    expected = f"v{version}"
    if tag != expected:
        raise RuntimeError(f"HEAD must have exact tag {expected}; found {tag!r}.")
    return commit, tag


def safe_tar_members(path: Path, required: set[str]) -> dict[str, int]:
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = {member.name.rstrip("/") for member in members}
        unsafe = []
        forbidden = []
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or member.issym() or member.islnk():
                unsafe.append(member.name)
            if member.isfile() and pure.suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES:
                forbidden.append(member.name)
        missing = sorted(required - names)
        if unsafe:
            raise RuntimeError(f"Unsafe tar members: {unsafe[:5]}")
        if forbidden:
            raise RuntimeError(f"Weight/state files must not be archived: {forbidden[:5]}")
        if missing:
            raise RuntimeError(f"Required tar members are missing: {missing}")
        sample_count = sum(
            member.isfile() and "/results/samples/" in member.name
            for member in members
        )
        return {
            "members": len(members),
            "files": sum(member.isfile() for member in members),
            "sample_files": sample_count,
        }


def validate_deployment(root: Path) -> dict[str, int]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    relatives = {path.relative_to(root).as_posix() for path in files}
    missing = sorted(DEPLOYMENT_REQUIRED - relatives)
    if missing:
        raise RuntimeError(f"Deployment files are missing: {missing}")
    forbidden = [
        path for path in files if path.suffix.lower() in FORBIDDEN_WEIGHT_SUFFIXES
    ]
    if forbidden:
        raise RuntimeError(f"Deployment archive contains weights/states: {forbidden[:5]}")

    quality_files = sorted((root / "quality" / "samples").glob("*.jsonl"))
    if len(quality_files) != 16:
        raise RuntimeError(f"Expected 16 deployment quality files, found {len(quality_files)}")
    quality_rows = sum(
        sum(1 for line in path.open("r", encoding="utf-8") if line.strip())
        for path in quality_files
    )
    if quality_rows != 21_104:
        raise RuntimeError(f"Expected 21,104 deployment quality rows, found {quality_rows}")

    block_files = list((root / "benchmarks" / "blocks").rglob("*.json"))
    final_block_files = [
        path
        for path in block_files
        if "formal" in path.relative_to(root).parts
        or "value" in path.relative_to(root).parts
    ]
    if len(final_block_files) != 320:
        raise RuntimeError(
            f"Expected 320 final deployment block records, found {len(final_block_files)}"
        )

    unified = json.loads((root / "unified_10_block_review.json").read_text(encoding="utf-8"))
    if not unified:
        raise RuntimeError("Unified deployment review is empty")
    return {
        "files": len(files),
        "quality_files": len(quality_files),
        "quality_rows": quality_rows,
        "all_block_records_including_engineering_pilot": len(block_files),
        "final_block_records": len(final_block_files),
    }


def deterministic_zip_tree(root: Path, output: Path, top_level: str) -> None:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
            relative = path.relative_to(root).as_posix()
            info = zipfile.ZipInfo(f"{top_level}/{relative}", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            with path.open("rb") as handle, archive.open(info, "w") as target:
                shutil.copyfileobj(handle, target, length=1024 * 1024)


def deterministic_zip_files(paths: list[Path], output: Path, top_level: str) -> None:
    with zipfile.ZipFile(
        output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(paths, key=lambda item: item.name):
            info = zipfile.ZipInfo(f"{top_level}/{path.name}", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())


def file_record(path: Path, role: str) -> dict[str, object]:
    return {
        "name": path.name,
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def ensure_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision-archive", type=Path, required=True)
    parser.add_argument("--tacq-archive", type=Path, required=True)
    parser.add_argument("--deployment-root", type=Path, required=True)
    parser.add_argument("--audit", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    version = release_version(ROOT)
    verify_manifest(ROOT)
    commit, tag = require_release_checkout(version)

    revision_archive = args.revision_archive.resolve()
    tacq_archive = args.tacq_archive.resolve()
    deployment_root = args.deployment_root.resolve()
    audits = [path.resolve() for path in args.audit]
    for path in [revision_archive, tacq_archive, *audits]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if len(audits) != 2:
        raise RuntimeError("Exactly two independent audit reports are required")

    revision_validation = safe_tar_members(revision_archive, REVISION_REQUIRED)
    if revision_validation["sample_files"] < 250:
        raise RuntimeError("Revision archive is missing expected sample families")
    tacq_validation = safe_tar_members(tacq_archive, TACQ_REQUIRED)
    deployment_validation = validate_deployment(deployment_root)

    output_dir = args.output_dir.resolve()
    ensure_output_directory(output_dir)

    source_output = output_dir / f"sg-mmp-reproducibility-v{version}-source.zip"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_zenodo_zip.py"),
            "--output",
            str(source_output),
        ],
        cwd=ROOT,
        check=True,
    )

    revision_output = output_dir / f"sg-mmp-revision-full-v4-results-v{version}.tar.gz"
    shutil.copy2(revision_archive, revision_output)
    tacq_output = output_dir / f"sg-mmp-tacq-execution-evidence-v{version}.tar.gz"
    shutil.copy2(tacq_archive, tacq_output)

    deployment_output = output_dir / f"sg-mmp-deployment-gguf-results-v{version}.zip"
    deterministic_zip_tree(
        deployment_root,
        deployment_output,
        f"sg-mmp-deployment-gguf-results-v{version}",
    )

    audits_output = output_dir / f"sg-mmp-integrity-audits-v{version}.zip"
    deterministic_zip_files(audits, audits_output, f"sg-mmp-integrity-audits-v{version}")

    readme_output = output_dir / f"README_ZENODO_v{version}.md"
    shutil.copy2(ROOT / "docs" / "zenodo_release.md", readme_output)

    roles = {
        source_output: "immutable source snapshot",
        revision_output: "complete revision-full-v4 result evidence",
        tacq_output: "run-era TaCQ source, registrations, samples, and readiness logs",
        deployment_output: "complete packed GGUF/CUDA deployment evidence",
        audits_output: "independent integrity audit reports",
        readme_output: "archive scope, exclusions, and upload instructions",
    }
    records = [file_record(path, role) for path, role in roles.items()]
    manifest_output = output_dir / f"MANIFEST_v{version}.json"
    manifest = {
        "schema": "sg-mmp-zenodo-delivery-v1",
        "release_version": version,
        "source_commit": commit,
        "source_tag": tag,
        "repository": "https://github.com/eeeeh123/sg-mmp-reproducibility",
        "zenodo_concept_doi": "10.5281/zenodo.21096006",
        "revision_validation": revision_validation,
        "tacq_validation": tacq_validation,
        "deployment_validation": deployment_validation,
        "files": records,
        "excluded": [
            "pretrained model checkpoints",
            "reconstructible PyTorch quantized states",
            "GGUF weight files",
            "dataset caches",
            "private credentials",
            "unpublished manuscript",
        ],
    }
    manifest_output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    checksum_paths = [*roles, manifest_output]
    checksum_output = output_dir / f"SHA256SUMS_v{version}.txt"
    checksum_output.write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in checksum_paths),
        encoding="utf-8",
        newline="\n",
    )

    for line in checksum_output.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        actual = sha256(output_dir / name)
        if actual != expected:
            raise RuntimeError(f"Post-build checksum mismatch: {name}")

    print(json.dumps({
        "release_version": version,
        "source_commit": commit,
        "output_dir": str(output_dir),
        "files": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [*checksum_paths, checksum_output]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()

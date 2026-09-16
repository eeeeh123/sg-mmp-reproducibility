"""Build the single manual-upload archive for a Zenodo software record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from write_manifest import is_released_file


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    if not manifest.exists():
        raise FileNotFoundError("SHA256SUMS is missing. Run scripts/write_manifest.py first.")
    mismatches = []
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        listed.add(relative)
        candidate = root / relative
        if not candidate.exists() or sha256(candidate) != expected:
            mismatches.append(relative)
    current = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if is_released_file(path, root, manifest)
    }
    missing = sorted(current - listed)
    extra = sorted(listed - current)
    if mismatches or missing or extra:
        raise RuntimeError(
            "SHA256SUMS is stale. Regenerate it before archiving. "
            f"Mismatches: {mismatches}; missing: {missing}; extra: {extra}"
        )


def release_version(root: Path) -> str:
    manifest = root / "configs" / "reproduction_manifest.json"
    version = str(json.loads(manifest.read_text(encoding="utf-8"))["release_version"])
    zenodo_version = str(
        json.loads((root / "zenodo.json").read_text(encoding="utf-8"))["version"]
    )
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*([^\s]+)\s*$", citation, flags=re.MULTILINE)
    citation_version = match.group(1) if match else None
    if version != zenodo_version or version != citation_version:
        raise RuntimeError(
            "Release version mismatch: "
            f"manifest={version}, zenodo={zenodo_version}, citation={citation_version}"
        )
    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    version = release_version(ROOT)
    output = args.output or ROOT / f"sg-mmp-reproducibility-v{version}-source.zip"
    output = output.resolve()
    verify_manifest(ROOT)

    files = sorted(
        path
        for path in ROOT.rglob("*")
        if is_released_file(path, ROOT, ROOT / "SHA256SUMS") or path == ROOT / "SHA256SUMS"
    )
    top_level = f"sg-mmp-reproducibility-v{version}"
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, arcname=f"{top_level}/{path.relative_to(ROOT).as_posix()}")

    print(f"Wrote {len(files)} files to {output}")
    print(f"SHA256 {sha256(output)}")


if __name__ == "__main__":
    main()

"""Build the single manual-upload archive for a Zenodo software record."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        candidate = root / relative
        if not candidate.exists() or sha256(candidate) != expected:
            mismatches.append(relative)
    if mismatches:
        raise RuntimeError(
            "SHA256SUMS is stale. Regenerate it before archiving. Mismatches: "
            + ", ".join(mismatches)
        )


def release_version(root: Path) -> str:
    manifest = root / "configs" / "reproduction_manifest.json"
    return str(json.loads(manifest.read_text(encoding="utf-8"))["release_version"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    version = release_version(ROOT)
    output = args.output or ROOT / f"sg-mmp-reproducibility-v{version}.zip"
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

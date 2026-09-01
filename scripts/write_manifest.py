"""Write SHA-256 checksums for a release snapshot."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def is_released_file(path: Path, root: Path, output: Path) -> bool:
    if not path.is_file() or path.resolve() == output:
        return False
    parts = path.relative_to(root).parts
    if not parts:
        return False
    if parts[0] in {
        ".git",
        "models",
        "results",
        "figures",
        "server_plans",
        ".venv",
        ".pytest_cache",
        ".release-audit",
    }:
        return False
    if any(
        part in {"__pycache__", "samples", "logs", "cache", "outputs"}
        for part in parts
    ):
        return False
    if path.name in {"server_all.sh", "server_all.log"}:
        return False
    return path.suffix.lower() not in {
        ".pt",
        ".pth",
        ".pyc",
        ".safetensors",
        ".tmp",
        ".zip",
        ".tar",
        ".gz",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("SHA256SUMS"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    paths = sorted(
        path
        for path in root.rglob("*")
        if is_released_file(path, root, output)
    )
    lines = [f"{digest(path)}  {path.relative_to(root).as_posix()}" for path in paths]
    # Checksum manifests are consumed across Linux and Windows. Write bytes so
    # Windows newline translation cannot make Git report every changed row as
    # trailing whitespace or alter the release archive across platforms.
    output.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    print(f"Wrote {len(lines)} checksums to {output}")


if __name__ == "__main__":
    main()

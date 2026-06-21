"""Launch TinyLlama validation jobs in the background with a clean env."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "experiments" / "fix_tinyllama" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    path_val = env.get("Path") or env.get("PATH") or ""
    for key in list(env):
        if key.lower() == "path":
            env.pop(key, None)
    env["Path"] = path_val
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    ns, args = parser.parse_known_args()
    if not args:
        raise SystemExit("No experiments/fix_tinyllama/run.py arguments provided")
    out_path = LOG_DIR / f"{ns.name}.out.log"
    err_path = LOG_DIR / f"{ns.name}.err.log"
    out = open(out_path, "w", encoding="utf-8")
    err = open(err_path, "w", encoding="utf-8")
    cmd = [sys.executable, "experiments/fix_tinyllama/run.py", *args]
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=out,
        stderr=err,
        env=clean_env(),
        creationflags=flags,
    )
    print(proc.pid)
    print(out_path)
    print(err_path)


if __name__ == "__main__":
    main()

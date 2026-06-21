"""Launch SmolLM config_b compact regeneration with a clean Windows env."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "experiments" / "fix_gsm8k_500" / "results_direct" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


def clean_env() -> dict[str, str]:
    env = dict(os.environ)
    path_val = env.get("Path") or env.get("PATH") or ""
    for key in list(env):
        if key.lower() == "path":
            env.pop(key, None)
    env["Path"] = path_val
    env["HF_DATASETS_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    return env


def main() -> None:
    step = sys.argv[1] if len(sys.argv) > 1 else "quantize"
    if step not in {"quantize", "eval", "all"}:
        raise SystemExit("Usage: python launch_smollm_config_b.py [quantize|eval|all]")

    out_path = LOG_DIR / f"smollm_config_b_compact_{step}.out.log"
    err_path = LOG_DIR / f"smollm_config_b_compact_{step}.err.log"
    out = open(out_path, "w", encoding="utf-8")
    err = open(err_path, "w", encoding="utf-8")
    cmd = [sys.executable, "experiments/exp13_smollm_config_b/run.py", step]
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

"""Regenerate GPTQ compact states without reading legacy oversized files."""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") in (None, "", "expandable_segments:True"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

sys.path.insert(0, ".")

from ptq.data import get_calib_dataset
from ptq.eval import cleanup_gpu
from ptq.quant.gptq import quantize_model_gptq


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "fix_gsm8k_500" / "results_direct"
STATUS = OUT / "requantize_status.json"
MODEL_SPECS = {
    "smollm": {
        "name": "SmolLM-1.7B",
        "path": ROOT / "models" / "SmolLM-1.7B",
        "dst": ROOT / "results" / "SmolLM-1.7B_gptq_compact.pt",
    },
}


def status(stage: str, **extra) -> None:
    record = {"time": datetime.now().isoformat(timespec="seconds"), "stage": stage, **extra}
    print("[status]", json.dumps(record, ensure_ascii=False), flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def quantize(model_key: str, n_calib: int, max_length: int, group_size: int) -> None:
    spec = MODEL_SPECS[model_key]
    dst = spec["dst"]
    if dst.exists():
        status("skip_existing", model=model_key, dst=str(dst), gb=round(dst.stat().st_size / 1024**3, 2))
        return

    t0 = time.time()
    status("load_model_start", model=model_key, path=str(spec["path"]))
    cleanup_gpu()
    status("load_model_cpu_start", model=model_key)
    model = AutoModelForCausalLM.from_pretrained(
        str(spec["path"]),
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    status("load_model_cpu_done", model=model_key)
    status("move_model_cuda_start", model=model_key)
    model.to("cuda:0")
    status("move_model_cuda_done", model=model_key)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(spec["path"]), trust_remote_code=True, local_files_only=True)
    status("load_model_done", model=model_key)

    status("calib_start", model=model_key, n_calib=n_calib, max_length=max_length)
    calib = get_calib_dataset(tokenizer, n_samples=n_calib, max_length=max_length)
    status("calib_done", model=model_key, shape=list(calib.shape))

    status("quantize_start", model=model_key, group_size=group_size)
    state = quantize_model_gptq(model, calib, bits=4, group_size=group_size)
    status("quantize_done", model=model_key, layers=len(state))

    tmp = dst.with_suffix(".pt.tmp")
    status("save_start", model=model_key, dst=str(dst))
    torch.save(state, tmp)
    try:
        os.replace(tmp, dst)
    except PermissionError:
        shutil.copy2(tmp, dst)
    status("save_done", model=model_key, dst=str(dst), gb=round(dst.stat().st_size / 1024**3, 2), elapsed_sec=round(time.time() - t0, 1))

    del state, calib, tokenizer, model
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_gpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_SPECS), default="smollm")
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=128)
    args = parser.parse_args()
    quantize(args.model, args.n_calib, args.max_length, args.group_size)


if __name__ == "__main__":
    main()

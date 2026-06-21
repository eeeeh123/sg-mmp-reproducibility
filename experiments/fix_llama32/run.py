"""Llama-3.2-1B-Instruct new-family validation wrapper.

This wrapper reuses the TinyLlama validation implementation but swaps the
model-specific constants. Keeping the execution path shared avoids subtle
differences in layer screening, GPTQ, SG-MMP, and GSM8K direct evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")

import experiments.fix_tinyllama.run as base
from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "fix_llama32" / "results"
LOG_DIR = ROOT / "experiments" / "fix_llama32" / "logs"
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_NAME = "Llama-3.2-1B-Instruct"
MODEL_KEY = "llama32"
MODEL_PATH = ROOT / "models" / MODEL_NAME
GPTQ_STATE = ROOT / "results" / f"{MODEL_NAME}_gptq_compact.pt"
SG_STATE = ROOT / "results" / f"{MODEL_NAME}_sg_mmp_compact.pt"
SCREEN_FILE = OUT / "layer_screen_train300.jsonl"
SELECTED_FILE = OUT / "selected_layers.json"
TRAIN_INDEX_FILE = OUT / "gsm8k_train_screen_indices.json"
SCREEN_SEED = 20260618
DEFAULT_SCREEN_N = 300
DEFAULT_TOP_K = 4


def configure_base() -> None:
    base.OUT = OUT
    base.LOG_DIR = LOG_DIR
    base.MODEL_ID = MODEL_ID
    base.MODEL_NAME = MODEL_NAME
    base.MODEL_KEY = MODEL_KEY
    base.MODEL_PATH = MODEL_PATH
    base.GPTQ_STATE = GPTQ_STATE
    base.SG_STATE = SG_STATE
    base.SCREEN_FILE = SCREEN_FILE
    base.SELECTED_FILE = SELECTED_FILE
    base.TRAIN_INDEX_FILE = TRAIN_INDEX_FILE
    base.SCREEN_SEED = SCREEN_SEED
    base.DEFAULT_SCREEN_N = DEFAULT_SCREEN_N
    base.DEFAULT_TOP_K = DEFAULT_TOP_K
    for path in [OUT, LOG_DIR, MODEL_PATH.parent, GPTQ_STATE.parent]:
        path.mkdir(parents=True, exist_ok=True)


def has_hf_token() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        return True
    for path in [Path.home() / ".cache" / "huggingface" / "token", Path.home() / ".huggingface" / "token"]:
        if path.exists():
            return True
    return False


def download_model() -> None:
    if not has_hf_token():
        raise RuntimeError(
            "Llama-3.2-1B-Instruct is a gated Meta repository and no Hugging Face token "
            "is configured. Request access on Hugging Face, then run `huggingface-cli login` "
            "or set HF_TOKEN in the local shell. Do not paste the token into chat."
        )
    endpoint = os.environ.get("PTQ_HF_DOWNLOAD_ENDPOINT") or os.environ.get("HF_ENDPOINT") or "https://hf-mirror.com"
    os.environ["HF_ENDPOINT"] = endpoint
    base.status("download_start", repo_id=MODEL_ID, local_dir=str(MODEL_PATH), endpoint=endpoint)
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_PATH),
        local_dir_use_symlinks=False,
        resume_download=True,
        token=True,
        allow_patterns=[
            "*.json",
            "*.model",
            "*.txt",
            "*.safetensors",
            "tokenizer.*",
            "generation_config.json",
            "special_tokens_map.json",
        ],
    )
    base.status("download_done", local_dir=str(MODEL_PATH))


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cmd",
        choices=["download", "inspect", "screen", "select", "quantize-gptq", "quantize-sg"],
    )
    parser.add_argument("--screen-n", type=int, default=DEFAULT_SCREEN_N)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.cmd == "download":
        download_model()
    elif args.cmd == "inspect":
        base.inspect_model()
    elif args.cmd == "screen":
        base.screen_layers(args.screen_n, args.batch_size, args.max_new_tokens, force=args.force)
    elif args.cmd == "select":
        selected = base.select_layers(args.top_k)
        print(selected)
    elif args.cmd == "quantize-gptq":
        base.quantize_gptq(args.calib_samples, args.calib_length, force=args.force)
    elif args.cmd == "quantize-sg":
        base.quantize_sg(args.calib_samples, args.calib_length, args.top_k, force=args.force)


if __name__ == "__main__":
    main()

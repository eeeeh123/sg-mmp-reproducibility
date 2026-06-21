"""Gemma-2-2B-it new-family validation wrapper.

This wrapper reuses the TinyLlama validation implementation but swaps the
model-specific constants. The shared execution path keeps sensitivity screening,
GPTQ-W4, SG-MMP, and direct GSM8K evaluation comparable with the other models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, ".")

import experiments.fix_tinyllama.run as base


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "fix_gemma2" / "results"
LOG_DIR = ROOT / "experiments" / "fix_gemma2" / "logs"
MODEL_ID = "google/gemma-2-2b-it"
MODEL_NAME = "gemma-2-2b-it"
MODEL_KEY = "gemma2"
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


def ensure_local_model() -> None:
    required = [
        "config.json",
        "generation_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    missing = [name for name in required if not (MODEL_PATH / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing Gemma local model files under {MODEL_PATH}: {missing}")
    base.status("local_model_ready", path=str(MODEL_PATH), files=len(required))


def main() -> None:
    configure_base()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cmd",
        choices=["check-files", "inspect", "screen", "select", "quantize-gptq", "quantize-sg"],
    )
    parser.add_argument("--screen-n", type=int, default=DEFAULT_SCREEN_N)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-length", type=int, default=2048)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_local_model()
    if args.cmd == "check-files":
        return
    if args.cmd == "inspect":
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

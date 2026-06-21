"""TinyLlama-1.1B new-family validation for the SG-MMP paper.

This experiment is intentionally self-contained and conservative:

1. Download TinyLlama-1.1B-Chat-v1.0 into ``models/``.
2. Select sensitive layers on a GSM8K train subset, not on the final test set.
3. Quantize GPTQ-W4 and SG-MMP compact states.
4. Evaluate through ``experiments/fix_gsm8k_500/direct_eval.py``.

The script does not perform the final GSM8K-500 evaluation itself; use
``direct_eval.py`` after the states are produced so the output format remains
identical to the existing three-model validation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") in (None, "", "expandable_segments:True"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.fix_gsm8k_500.direct_eval import (
    build_fewshot,
    build_model_prompts,
    extract_prediction,
    get_dataset,
    gold_answer,
    is_correct,
)
from ptq.data import get_calib_dataset
from ptq.eval import cleanup_gpu
from ptq.quant.gptq import quantize_model_gptq
from ptq.quant.mixed_precision import (
    ATTN_PROJ,
    FFN_PROJ,
    parse_layer_num,
    quantize_model_mixed_precision,
)
from ptq.quant.rtn import dequantize_tensor_rtn, quantize_tensor_rtn


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "fix_tinyllama" / "results"
LOG_DIR = ROOT / "experiments" / "fix_tinyllama" / "logs"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MODEL_NAME = "TinyLlama-1.1B-intermediate-step-1431k-3T"
MODEL_KEY = "tinyllama"
MODEL_PATH = ROOT / "models" / MODEL_NAME
GPTQ_STATE = ROOT / "results" / f"{MODEL_NAME}_gptq_compact.pt"
SG_STATE = ROOT / "results" / f"{MODEL_NAME}_sg_mmp_compact.pt"
SCREEN_FILE = OUT / "layer_screen_base_train300.jsonl"
SELECTED_FILE = OUT / "selected_layers_base.json"
TRAIN_INDEX_FILE = OUT / "gsm8k_train_screen_indices_base.json"

for path in [OUT, LOG_DIR, MODEL_PATH.parent, GPTQ_STATE.parent]:
    path.mkdir(parents=True, exist_ok=True)

SCREEN_SEED = 20260617
DEFAULT_SCREEN_N = 300
DEFAULT_TOP_K = 4


def status(stage: str, **extra) -> None:
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        **extra,
    }
    print("[status]", json.dumps(record, ensure_ascii=False), flush=True)
    (OUT / "status.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_state_atomic(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    try:
        os.replace(tmp, path)
    except PermissionError:
        # Windows can briefly keep a memory-mapped file handle alive. Falling
        # back to a regular copy is less elegant but avoids losing the state.
        import shutil

        shutil.copyfile(tmp, path)
    status("state_saved", path=str(path), bytes=path.stat().st_size, entries=len(state))


def download_model() -> None:
    # The mirror endpoint works for some cached dataset paths in this project,
    # but huggingface_hub 1.x may reject mirror metadata for snapshot downloads.
    # Use the official endpoint by default for the one-time model download.
    os.environ["HF_ENDPOINT"] = os.environ.get("PTQ_HF_DOWNLOAD_ENDPOINT", "https://huggingface.co")
    status("download_start", repo_id=MODEL_ID, local_dir=str(MODEL_PATH))
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_PATH),
        local_dir_use_symlinks=False,
        resume_download=True,
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
    status("download_done", local_dir=str(MODEL_PATH))


def load_model_tokenizer():
    cleanup_gpu()
    status("load_model_cpu_start", path=str(MODEL_PATH))
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_PATH),
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    status("load_model_cpu_done")
    status("move_model_cuda_start")
    model.to("cuda:0")
    model.eval()
    status("move_model_cuda_done", cuda_memory_mb=round(torch.cuda.memory_allocated() / 1024**2, 1))
    tok = AutoTokenizer.from_pretrained(str(MODEL_PATH), trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return model, tok


def inspect_model() -> None:
    model, tok = load_model_tokenizer()
    linear = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, nn.Linear) and "lm_head" not in name]
    short_counts = {}
    layers = set()
    for name, _ in linear:
        short_counts[name.split(".")[-1]] = short_counts.get(name.split(".")[-1], 0) + 1
        ln = parse_layer_num(name)
        if ln >= 0:
            layers.add(ln)
    payload = {
        "model": MODEL_NAME,
        "vocab_size": len(tok),
        "n_linear": len(linear),
        "n_transformer_layers": len(layers),
        "linear_short_counts": short_counts,
        "first_linear_modules": [name for name, _ in linear[:16]],
    }
    (OUT / "inspect.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    del model, tok
    cleanup_gpu()


def train_screen_indices(n: int) -> list[int]:
    if TRAIN_INDEX_FILE.exists():
        data = json.loads(TRAIN_INDEX_FILE.read_text(encoding="utf-8"))
        if data.get("n") == n:
            return data["indices"]
    train, _ = get_dataset()
    candidates = list(range(5, len(train)))
    rng = random.Random(SCREEN_SEED)
    rng.shuffle(candidates)
    selected = sorted(candidates[:n])
    TRAIN_INDEX_FILE.write_text(
        json.dumps(
            {
                "dataset": "openai/gsm8k",
                "split": "train",
                "n": n,
                "seed": SCREEN_SEED,
                "selection": "sorted(shuffled(range(5, len(train)))[:n])",
                "indices": selected,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return selected


@torch.no_grad()
def eval_loaded_model(
    model,
    tok,
    examples: list[dict],
    batch_size: int,
    max_new_tokens: int,
    phase: str,
    layer: int | None = None,
) -> dict:
    train, _ = get_dataset()
    prefix = build_fewshot(train, k=5)
    rows = []
    t0 = time.time()
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        prompts = build_model_prompts(MODEL_KEY, tok, train, prefix, [ex["question"] for ex in batch])
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
        input_len = enc["input_ids"].shape[1]
        outputs = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        decoded = tok.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        for ex, generation in zip(batch, decoded):
            gold = gold_answer(ex["answer"])
            pred = extract_prediction(generation)
            rows.append({"gold": gold, "prediction": pred, "correct": is_correct(pred, gold)})
        done = len(rows)
        acc = 100 * sum(r["correct"] for r in rows) / done
        status("screen_eval_progress", phase=phase, layer=layer, done=done, total=len(examples), accuracy=round(acc, 2))
        del enc, outputs
        torch.cuda.empty_cache()
    correct = sum(r["correct"] for r in rows)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": round(100 * correct / len(rows), 2),
        "elapsed_s": round(time.time() - t0, 1),
    }


def transformer_layers(model) -> list[tuple[int, list[tuple[str, nn.Linear]]]]:
    per_layer: dict[int, list[tuple[str, nn.Linear]]] = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear) or "lm_head" in name:
            continue
        ln = parse_layer_num(name)
        if ln >= 0:
            per_layer.setdefault(ln, []).append((name, mod))
    return [(ln, per_layer[ln]) for ln in sorted(per_layer)]


def existing_screen_records() -> dict:
    records = {}
    for row in read_jsonl(SCREEN_FILE):
        records[(row["type"], row.get("layer"))] = row
    return records


def screen_layers(n: int, batch_size: int, max_new_tokens: int, force: bool = False) -> None:
    if force and SCREEN_FILE.exists():
        SCREEN_FILE.unlink()
    train, _ = get_dataset()
    indices = train_screen_indices(n)
    examples = [train[i] for i in indices]
    records = existing_screen_records()

    model, tok = load_model_tokenizer()
    layers = transformer_layers(model)
    status("screen_start", n=n, batch_size=batch_size, max_new_tokens=max_new_tokens, layers=len(layers))

    if ("baseline", None) not in records:
        status("screen_baseline_start")
        baseline = eval_loaded_model(
            model,
            tok,
            examples,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            phase="baseline",
        )
        append_jsonl(SCREEN_FILE, {"type": "baseline", **baseline})
        status("screen_baseline_done", **baseline)
    else:
        baseline = records[("baseline", None)]
        status("screen_baseline_cached", accuracy=baseline["accuracy"])

    for idx, (layer_idx, specs) in enumerate(layers, start=1):
        if ("layer", layer_idx) in records:
            status("screen_layer_cached", layer=layer_idx, accuracy=records[("layer", layer_idx)]["accuracy"])
            continue
        status("screen_layer_start", layer=layer_idx, ordinal=idx, total=len(layers))
        saved = [(name, mod, mod.weight.data.detach().cpu().clone()) for name, mod in specs]
        try:
            for name, mod, _ in saved:
                w_q, scale, zero = quantize_tensor_rtn(mod.weight.data, bits=4, group_size=128)
                w_deq = dequantize_tensor_rtn(w_q, scale, zero, group_size=128)
                mod.weight.data.copy_(w_deq.to(mod.weight.dtype))
                del w_q, scale, zero, w_deq
            torch.cuda.empty_cache()
            result = eval_loaded_model(
                model,
                tok,
                examples,
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                phase="layer",
                layer=layer_idx,
            )
            drop = round(float(baseline["accuracy"]) - float(result["accuracy"]), 2)
            row = {"type": "layer", "layer": layer_idx, "bits": 4, **result, "drop_vs_fp16": drop}
            append_jsonl(SCREEN_FILE, row)
            status("screen_layer_done", layer=layer_idx, accuracy=result["accuracy"], drop=drop)
        finally:
            for _, mod, weight in saved:
                mod.weight.data.copy_(weight.to(mod.weight.device, dtype=mod.weight.dtype))
            del saved
            gc.collect()
            torch.cuda.empty_cache()

    del model, tok
    cleanup_gpu()
    select_layers(DEFAULT_TOP_K)


def select_layers(top_k: int) -> list[int]:
    rows = read_jsonl(SCREEN_FILE)
    baseline = next((r for r in rows if r["type"] == "baseline"), None)
    layer_rows = [r for r in rows if r["type"] == "layer" and r.get("accuracy") is not None]
    if baseline is None or not layer_rows:
        raise RuntimeError(f"Screen file incomplete: {SCREEN_FILE}")
    ranked = sorted(layer_rows, key=lambda r: (r["drop_vs_fp16"], -r["layer"]), reverse=True)
    selected = [int(r["layer"]) for r in ranked[:top_k]]
    payload = {
        "model": MODEL_NAME,
        "selection_data": "GSM8K train subset",
        "screen_file": str(SCREEN_FILE.relative_to(ROOT)),
        "baseline_accuracy": baseline["accuracy"],
        "top_k": top_k,
        "selected_layers": selected,
        "ranking": ranked,
    }
    SELECTED_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    status("layers_selected", top_k=top_k, selected_layers=selected)
    return selected


def quantize_gptq(n_samples: int, max_length: int, force: bool = False) -> None:
    if GPTQ_STATE.exists() and not force:
        status("gptq_state_exists", path=str(GPTQ_STATE), bytes=GPTQ_STATE.stat().st_size)
        return
    model, tok = load_model_tokenizer()
    status("calib_start", n_samples=n_samples, max_length=max_length, dataset="wikitext")
    calib = get_calib_dataset(tok, n_samples=n_samples, max_length=max_length, dataset_name="wikitext")
    status("calib_done", shape=list(calib.shape))
    status("gptq_quantize_start")
    state = quantize_model_gptq(model, calib, bits=4, group_size=128)
    status("gptq_quantize_done", entries=len(state))
    save_state_atomic(state, GPTQ_STATE)
    del state, calib, model, tok
    cleanup_gpu()


def sg_policy(selected_layers: set[int]):
    def policy(global_idx: int, layer_name: str, layer_short: str) -> str:
        ln = parse_layer_num(layer_name)
        if ln in selected_layers:
            return "w8"
        if layer_short in ATTN_PROJ:
            return "w8"
        if layer_short in FFN_PROJ:
            return "w4"
        return "skip"

    return policy


def quantize_sg(n_samples: int, max_length: int, top_k: int, force: bool = False) -> None:
    if SG_STATE.exists() and not force:
        status("sg_state_exists", path=str(SG_STATE), bytes=SG_STATE.stat().st_size)
        return
    selected = select_layers(top_k)
    model, tok = load_model_tokenizer()
    status("calib_start", n_samples=n_samples, max_length=max_length, dataset="wikitext")
    calib = get_calib_dataset(tok, n_samples=n_samples, max_length=max_length, dataset_name="wikitext")
    status("calib_done", shape=list(calib.shape))
    status("sg_quantize_start", selected_layers=selected)
    state = quantize_model_mixed_precision(
        model,
        calib,
        bits_w4=4,
        group_size=128,
        layer_policy=sg_policy(set(selected)),
    )
    status("sg_quantize_done", entries=len(state))
    save_state_atomic(state, SG_STATE)
    del state, calib, model, tok
    cleanup_gpu()


def main() -> None:
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
        inspect_model()
    elif args.cmd == "screen":
        screen_layers(args.screen_n, args.batch_size, args.max_new_tokens, force=args.force)
    elif args.cmd == "select":
        selected = select_layers(args.top_k)
        print(selected)
    elif args.cmd == "quantize-gptq":
        quantize_gptq(args.calib_samples, args.calib_length, force=args.force)
    elif args.cmd == "quantize-sg":
        quantize_sg(args.calib_samples, args.calib_length, args.top_k, force=args.force)


if __name__ == "__main__":
    main()

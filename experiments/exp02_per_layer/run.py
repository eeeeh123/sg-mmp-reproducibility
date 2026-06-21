#!/usr/bin/env python
"""exp02_per_layer: 逐层 4-bit vs 8-bit 敏感度。

对 Qwen2.5-0.5B 的每个 transformer 层单独量化、评测 GSM8K、恢复，
定位哪些层对量化最敏感。

用法:
  python experiments/exp02_per_layer/run.py
  python experiments/exp02_per_layer/run.py --model Qwen2.5-1.5B --bits 4
"""
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import gc
import json
import time
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq.quant.rtn import quantize_tensor_rtn, dequantize_tensor_rtn
from ptq.eval import run_eval_on_model, cleanup_gpu


def get_transformer_layers(model):
    """返回 [(layer_idx, layer_module, linear_specs)] 列表。

    linear_specs: [(full_name, module)] 该 transformer 层内所有 Linear 模块。
    """
    layers = []
    for name, mod in model.named_modules():
        parts = name.split(".")
        if "layers" in parts:
            idx = parts.index("layers")
            if idx + 1 >= len(parts):
                continue  # skip "model.layers" ModuleList itself
            layer_idx = int(parts[idx + 1])
            while len(layers) <= layer_idx:
                layers.append([])
            if isinstance(mod, nn.Linear) and "lm_head" not in name:
                layers[layer_idx].append((name, mod))
    return [(i, model.model.layers[i], specs) for i, specs in enumerate(layers) if specs]


def load_results(output_file: str) -> dict:
    """读取已有结果 {(layer_idx, bits): score}."""
    existing = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == "run":
                    existing[(r["layer"], r["bits"])] = r["gsm8k_score"]
    return existing


def save_result(output_file: str, layer_idx: int, bits: int, score, elapsed: float):
    """追加一行结果。"""
    record = {"type": "run", "layer": layer_idx, "bits": bits,
              "gsm8k_score": score, "elapsed_s": round(elapsed, 1)}
    with open(output_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def save_baseline(output_file: str, score, elapsed: float):
    """保存 baseline。"""
    record = {"type": "baseline", "gsm8k_score": score, "elapsed_s": round(elapsed, 1)}
    with open(output_file, "a") as f:
        f.write(json.dumps(record) + "\n")


def main():
    parser = argparse.ArgumentParser(description="exp02_per_layer")
    parser.add_argument("--model", type=str, default="Qwen2.5-0.5B")
    parser.add_argument("--bits", type=str, default="4,8")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    bits_list = [int(b) for b in args.bits.split(",")]
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "per_layer.jsonl")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(_project_root, "models", args.model)

    existing = load_results(output_file)

    # Load model once
    print(f"=== exp02_per_layer: {args.model} ===")
    cleanup_gpu()
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    layers = get_transformer_layers(model)
    n_layers = len(layers)
    total_runs = (1 if not args.skip_baseline else 0) + n_layers * len(bits_list)
    print(f"  Model has {n_layers} transformer layers, {n_layers * 7} Linear modules")
    print(f"  Bits: {bits_list}, Total runs: {total_runs} (including baseline)")
    run_count = 0

    # Baseline
    if not args.skip_baseline:
        existing_baseline = None
        if os.path.exists(output_file):
            with open(output_file) as f:
                for line in f:
                    r = json.loads(line)
                    if r["type"] == "baseline":
                        existing_baseline = r["gsm8k_score"]
        if existing_baseline is not None:
            print(f"\nBaseline (fp16): {existing_baseline:.2f}% (cached)")
        else:
            run_count += 1
            print(f"\n[{run_count}/{total_runs}] Baseline (fp16)...")
            t_start = time.time()
            scores = run_eval_on_model(model, tokenizer, ["gsm8k"],
                                       batch_size=args.batch_size, limit=300)
            elapsed = time.time() - t_start
            baseline_score = scores.get("gsm8k")
            print(f"  gsm8k: {baseline_score:.2f}% ({elapsed:.0f}s)")
            save_baseline(output_file, baseline_score, elapsed)

    # Per-layer quantization
    for bits in bits_list:
        for layer_idx, layer_mod, linear_specs in layers:
            run_count += 1
            key = (layer_idx, bits)
            if key in existing:
                print(f"[{run_count}/{total_runs}] Layer {layer_idx}/{n_layers-1} {bits}-bit: {existing[key]:.2f}% (cached)")
                continue

            print(f"\n[{run_count}/{total_runs}] Layer {layer_idx}/{n_layers-1} {bits}-bit...")
            t_start = time.time()

            # Save original weights
            saved_weights = {}
            for full_name, mod in linear_specs:
                saved_weights[full_name] = mod.weight.data.clone()

            try:
                # Quantize all Linear modules in this layer
                for full_name, mod in linear_specs:
                    w_q, scale, zero = quantize_tensor_rtn(
                        mod.weight.data, bits=bits, group_size=128)
                    w_deq = dequantize_tensor_rtn(w_q, scale, zero, group_size=128)
                    mod.weight.data.copy_(w_deq.to(mod.weight.dtype))

                # Eval
                scores = run_eval_on_model(model, tokenizer, ["gsm8k"],
                                           batch_size=args.batch_size, limit=300)
                score = scores.get("gsm8k")
                elapsed = time.time() - t_start
                print(f"  gsm8k: {score:.2f}% ({elapsed:.0f}s)")
                save_result(output_file, layer_idx, bits, score, elapsed)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
                save_result(output_file, layer_idx, bits, None, time.time() - t_start)

            finally:
                # Restore original weights
                for full_name, mod in linear_specs:
                    mod.weight.data.copy_(saved_weights[full_name])
                del saved_weights
                gc.collect()
                torch.cuda.empty_cache()

    # Summary
    print(f"\n=== Summary ===")
    print(f"Results: {output_file}")

    # Print per-bit summary
    all_results = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == "run" and r["gsm8k_score"] is not None:
                    all_results[(r["layer"], r["bits"])] = r["gsm8k_score"]

    for bits in bits_list:
        scores = [all_results[k] for k in sorted(all_results) if k[1] == bits]
        if scores:
            print(f"  {bits}-bit: min={min(scores):.2f}%, max={max(scores):.2f}%, "
                  f"mean={sum(scores)/len(scores):.2f}%")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

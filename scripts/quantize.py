#!/usr/bin/env python
"""统一量化入口 — 薄 CLI。

用法:
  python scripts/quantize.py --model Qwen2.5-0.5B --method rtn
  python scripts/quantize.py --model Qwen2.5-1.5B --method gptq,awq
  python scripts/quantize.py --all
"""
import argparse
import torch
import gc
import os
from ptq.config import MODELS, QUANT_CONFIGS
from ptq.data import get_calib_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_quantize(model_name: str, method: str, model_dir: str = "models",
                 results_dir: str = "results", bits: int = 4, group_size: int = 128):
    """对指定模型运行量化，保存 state 文件到 results_dir。"""
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(model_dir, model_name)

    print(f"\n{'='*50}")
    print(f"Quantizing {model_name} with {method}")
    print(f"{'='*50}")

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if method == "rtn":
        from ptq.quant.rtn import quantize_model_rtn
        state = quantize_model_rtn(model, bits=bits, group_size=group_size)
        out = os.path.join(results_dir, f"{model_name}_rtn.pt")
        torch.save(state, out)
        print(f"Saved: {out}")

    elif method == "gptq":
        from ptq.quant.gptq import quantize_model_gptq
        calib = get_calib_dataset(tokenizer, n_samples=128, max_length=2048)
        state = quantize_model_gptq(model, calib, bits=bits, group_size=group_size)
        out = os.path.join(results_dir, f"{model_name}_gptq.pt")
        torch.save(state, out)
        print(f"Saved: {out}")

    elif method == "awq":
        from ptq.quant.awq import quantize_model_awq
        calib = get_calib_dataset(tokenizer, n_samples=32, max_length=2048)
        state, scales = quantize_model_awq(model, calib, bits=bits, group_size=group_size)
        torch.save(state, os.path.join(results_dir, f"{model_name}_awq.pt"))
        torch.save(scales, os.path.join(results_dir, f"{model_name}_awq_scales.pt"))
        print(f"Saved: {model_name}_awq.pt, {model_name}_awq_scales.pt")

    elif method == "smoothquant":
        from ptq.quant.smoothquant import compute_smooth_scales
        calib = get_calib_dataset(tokenizer, n_samples=32, max_length=2048)
        scales = compute_smooth_scales(model, calib)
        out = os.path.join(results_dir, f"{model_name}_smoothquant.pt")
        torch.save(scales, out)
        print(f"Saved: {out}")

    elif method == "mixed_precision":
        from ptq.quant.mixed_precision import quantize_model_mixed_precision
        calib = get_calib_dataset(tokenizer, n_samples=128, max_length=2048)
        state = quantize_model_mixed_precision(model, calib, bits_w4=4, group_size=128)
        out = os.path.join(results_dir, f"{model_name}_mixed_precision.pt")
        torch.save(state, out)
        print(f"Saved: {out}")

    else:
        raise ValueError(f"Unknown method: {method}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done: {model_name} [{method}]")


def main():
    parser = argparse.ArgumentParser(description="PTQ Benchmark — Quantization")
    parser.add_argument("--model", type=str, default=None, help="Model name or comma-separated")
    parser.add_argument("--method", type=str, default=None, help="Method or comma-separated")
    parser.add_argument("--all", action="store_true", help="Quantize all models with all methods")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    if args.all:
        models = [m["name"] for m in MODELS]
        methods = [q["method"] for q in QUANT_CONFIGS if q["method"] != "fp16"]
    else:
        if not args.model or not args.method:
            parser.error("Must specify --model/--method or --all")
        models = args.model.split(",")
        methods = args.method.split(",")

    for model_name in models:
        for method in methods:
            run_quantize(model_name, method,
                         model_dir=args.model_dir, results_dir=args.results_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""exp04_module_error: Attention vs FFN 模块量化误差分布。

对 Qwen2.5-0.5B 的每个 Linear 层做 4-bit RTN 量化，计算 MSE(原始输出, 量化输出)，
生成 layers × module_types 的 heatmap 数据。

用法:
  python experiments/exp04_module_error/run.py
  python experiments/exp04_module_error/run.py --model Qwen2.5-1.5B
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
import json
import gc
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq.quant.rtn import quantize_tensor_rtn, dequantize_tensor_rtn
from ptq.data import get_calib_dataset


MODULE_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"]


def collect_activations(model, calib_data, device):
    """Hook 所有 Linear 层，收集一次 forward 的输入激活。"""
    linear_layers = [(name, mod) for name, mod in model.named_modules()
                     if isinstance(mod, nn.Linear) and "lm_head" not in name]

    captured_inputs = {}

    def make_hook(layer_name):
        def hook(module, args, output):
            inp = args[0].detach().cpu()
            if layer_name in captured_inputs:
                captured_inputs[layer_name] = torch.cat([captured_inputs[layer_name], inp], dim=0)
            else:
                captured_inputs[layer_name] = inp
        return hook

    hooks = []
    for name, mod in linear_layers:
        hooks.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    n_act_samples = min(8, calib_data.shape[0])
    with torch.no_grad():
        for i in range(n_act_samples):
            batch = calib_data[i:i+1].to(device)
            try:
                model(batch)
            except Exception:
                pass
            del batch
            torch.cuda.empty_cache()
    print(f"  Collected activations from {n_act_samples} samples")

    for h in hooks:
        h.remove()

    return captured_inputs


def compute_module_mse(module, inp_tensor):
    """计算单个 Linear 层的量化 MSE 和 relative MSE。

    inp_tensor: (batch*tokens, in_features) on CPU
    """
    device = module.weight.device
    w_orig = module.weight.data.float()
    bias = module.bias.data.float() if module.bias is not None else None
    inp = inp_tensor.to(device)

    # 原始输出
    out_orig = nn.functional.linear(inp.float(), w_orig, bias)

    # 4-bit RTN 量化 + 去量化
    w_q, scale, zero = quantize_tensor_rtn(w_orig, bits=4, group_size=128)
    w_deq = dequantize_tensor_rtn(w_q, scale, zero, group_size=128).to(device)

    # 量化输出
    out_quant = nn.functional.linear(inp.float(), w_deq, bias)

    mse = nn.functional.mse_loss(out_quant, out_orig).item()
    var = out_orig.var().item()
    rel_mse = mse / var if var > 1e-12 else float("inf")

    return mse, rel_mse


def extract_layer_idx(name: str) -> int:
    """从模块名提取层号。如 'model.layers.5.self_attn.q_proj' → 5"""
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers":
            return int(parts[i + 1])
    return -1


def extract_module_type(name: str) -> str:
    """从模块名提取类型。如 '...self_attn.q_proj' → 'q_proj'"""
    return name.split(".")[-1]


def main():
    parser = argparse.ArgumentParser(description="exp04_module_error")
    parser.add_argument("--model", type=str, default="Qwen2.5-0.5B")
    parser.add_argument("--n_samples", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = os.path.join(_project_root, "models", args.model)

    print(f"=== exp04_module_error: {args.model} ===")

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("Loading calibration data...")
    calib_data = get_calib_dataset(tokenizer, n_samples=args.n_samples, max_length=2048)

    print("Collecting activations via hooks...")
    captured_inputs = collect_activations(model, calib_data, device)

    linear_layers = [(name, mod) for name, mod in model.named_modules()
                     if isinstance(mod, nn.Linear) and "lm_head" not in name]

    print(f"Computing MSE for {len(linear_layers)} Linear layers...")
    results = []
    for i, (name, mod) in enumerate(linear_layers):
        if name not in captured_inputs:
            print(f"  [{i+1}/{len(linear_layers)}] {name}: SKIP (no activations)")
            continue

        inp = captured_inputs[name]
        inp_flat = inp.view(-1, inp.shape[-1])

        mse, rel_mse = compute_module_mse(mod, inp_flat)
        layer_idx = extract_layer_idx(name)
        mod_type = extract_module_type(name)

        results.append({
            "name": name,
            "layer": layer_idx,
            "module_type": mod_type,
            "mse": round(mse, 8),
            "rel_mse": round(rel_mse, 6),
        })

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(linear_layers)}] {name}: MSE={mse:.6f}, rel={rel_mse:.6f}")
        del inp, inp_flat
        torch.cuda.empty_cache()

    print(f"\nDone. Processed {len(results)} layers.")

    # Save JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "module_mse.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # Build CSV heatmap: rows=layers, cols=module_types
    num_layers = max(r["layer"] for r in results) + 1
    csv_path = os.path.join(args.output_dir, "module_mse.csv")
    with open(csv_path, "w") as f:
        header = ["layer"] + MODULE_TYPES
        f.write(",".join(header) + "\n")
        for layer in range(num_layers):
            row = [str(layer)]
            for mt in MODULE_TYPES:
                vals = [r["mse"] for r in results if r["layer"] == layer and r["module_type"] == mt]
                row.append(f"{vals[0]:.6f}" if vals else "")
            f.write(",".join(row) + "\n")
    print(f"Saved: {csv_path}")

    # Summary by module type
    print("\n=== Summary by Module Type ===")
    print(f"{'type':>10s}  {'mean_MSE':>12s}  {'mean_rel':>10s}")
    for mt in MODULE_TYPES:
        vals = [r["mse"] for r in results if r["module_type"] == mt]
        rels = [r["rel_mse"] for r in results if r["module_type"] == mt]
        if vals:
            print(f"{mt:>10s}  {sum(vals)/len(vals):12.6f}  {sum(rels)/len(rels):10.6f}")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

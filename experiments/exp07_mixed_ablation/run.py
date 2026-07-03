"""模块混合精度消融实验：A/B/C 三配置 GSM8K 快速筛选。

配置 A: 模块级 — q/k/v→W8, o/gate/up/down→W4（已跑，复用结果）
配置 B: 敏感层引导 — layer 2/6/7/11 全 W8，其余按模块分
配置 C: 渐进式 — 浅层/深层全 W8，中层仅 v_proj+k_proj→W8

先跑 GSM8K（limit=300），选最优跑全量。
"""

import os
import sys
sys.path.insert(0, ".")

import argparse
import torch
import gc
import re
from ptq.data import get_calib_dataset
from ptq.quant.mixed_precision import (
    quantize_model_mixed_precision,
    apply_mixed_precision_to_model_gpu,
    ATTN_PROJ, FFN_PROJ,
)
from ptq.eval import run_eval, cleanup_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Layer policy 定义
# ============================================================

def _layer_num(name: str) -> int:
    m = re.search(r'layers\.(\d+)', name)
    return int(m.group(1)) if m else -1


def policy_a(layer_idx, layer_name, layer_short):
    """配置 A: 模块级 — q/k/v→W8, o/gate/up/down→W4"""
    if layer_short in FFN_PROJ:
        return "w4"
    elif layer_short in ATTN_PROJ:
        return "w8"
    return "skip"


def policy_b(layer_idx, layer_name, layer_short):
    """配置 B: layer 2/6/7/11 全 W8，其余按模块分"""
    ln = _layer_num(layer_name)
    if ln in {2, 6, 7, 11}:
        return "w8"  # 敏感层全部模块 W8
    if layer_short in FFN_PROJ:
        return "w4"
    elif layer_short in ATTN_PROJ:
        return "w8"
    return "skip"


def policy_c(layer_idx, layer_name, layer_short):
    """配置 C: 浅层(0-3)全W8, 深层(20-23)全W8, 中层(4-19)仅v+k→W8"""
    ln = _layer_num(layer_name)
    if ln <= 3 or ln >= 20:
        return "w8"  # 浅层/深层全部 W8
    # 中层 4-19: 仅 v_proj, k_proj → W8
    if layer_short in {"v_proj", "k_proj"}:
        return "w8"
    elif layer_short in FFN_PROJ or layer_short in {"q_proj", "o_proj"}:
        return "w4"
    return "skip"


CONFIGS = [
    ("config_a", None, "results/Qwen2.5-0.5B_config_a.pt"),   # 使用默认 policy（同 mixed_precision）
    ("config_b", policy_b, "results/Qwen2.5-0.5B_config_b.pt"),
    ("config_c", policy_c, "results/Qwen2.5-0.5B_config_c.pt"),
]


# ============================================================
# 量化 + GSM8K 评测
# ============================================================

def quantize_config(config_name, policy, state_path):
    """量化并保存 state 文件。"""
    if os.path.exists(state_path):
        print(f"[{config_name}] State already exists: {state_path}, skip quantization")
        return

    print(f"\n{'='*60}")
    print(f"[{config_name}] Quantizing...")
    print(f"{'='*60}")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    calib = get_calib_dataset(tokenizer, n_samples=128, max_length=2048)

    print(f"  Calib shape: {calib.shape}, policy={'default' if policy is None else 'custom'}")
    state = quantize_model_mixed_precision(model, calib, bits_w4=4, group_size=128,
                                           layer_policy=policy)
    torch.save(state, state_path)
    print(f"  Saved: {state_path} ({len(state)} layers)")

    del model, tokenizer, calib, state
    gc.collect()
    torch.cuda.empty_cache()


def eval_gsm8k(config_name, state_path):
    """加载量化模型，运行 GSM8K 评测。"""
    print(f"\n[{config_name}] GSM8K eval...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    quant_state = torch.load(state_path, map_location="cpu", weights_only=False)
    apply_mixed_precision_to_model_gpu(model, quant_state)

    # 用 run_eval 的底层 API，直接复用已加载的 model
    from ptq.eval import run_eval_on_model
    scores = run_eval_on_model(model, tokenizer, ["gsm8k"], batch_size=4,
                               max_gen_toks=256, limit=300)
    gsm8k = scores.get("gsm8k", None)
    print(f"  [{config_name}] GSM8K={gsm8k:.2f}" if gsm8k else f"  [{config_name}] GSM8K=FAILED")

    del model, tokenizer, quant_state
    cleanup_gpu()
    return gsm8k


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="config_a,config_b,config_c",
        help="Comma-separated subset of config_a, config_b, config_c.",
    )
    parser.add_argument(
        "--quantize-only",
        action="store_true",
        help="Write requested states without running the legacy GSM8K-300 screen.",
    )
    args = parser.parse_args()
    requested = {name.strip() for name in args.configs.split(",") if name.strip()}
    known = {name for name, _, _ in CONFIGS}
    unknown = requested - known
    if unknown:
        parser.error(f"Unknown configs: {sorted(unknown)}")

    results = {}

    for config_name, policy, state_path in CONFIGS:
        if config_name not in requested:
            continue
        quantize_config(config_name, policy, state_path)
        if args.quantize_only:
            continue
        gsm8k = eval_gsm8k(config_name, state_path)
        results[config_name] = gsm8k

    if args.quantize_only:
        print(f"Quantized requested configurations: {', '.join(sorted(requested))}")
        return

    # 对比
    print(f"\n{'='*60}")
    print("GSM8K 对比")
    print(f"{'='*60}")
    # 加入已跑的 mixed_precision (== config_a 用默认 policy)
    print(f"  config_a (default/模块级)  : GSM8K=21.33 (复用 mixed_precision)")
    print(f"  config_a (显式重跑)        : GSM8K={results.get('config_a', 'N/A')}")
    print(f"  config_b (敏感层引导)      : GSM8K={results.get('config_b', 'N/A')}")
    print(f"  config_c (渐进式)          : GSM8K={results.get('config_c', 'N/A')}")

    # 找最优
    valid = {k: v for k, v in results.items() if v is not None}
    if valid:
        best = max(valid, key=valid.get)
        print(f"\n最优配置: {best} (GSM8K={valid[best]:.2f})")
        print(f"下一步: 跑 {best} 全量 benchmark")
    else:
        print("\n所有配置均失败!")


if __name__ == "__main__":
    main()

"""模块级混合精度量化：注意力投影 INT8 + FFN 投影 GPTQ-W4。

q_proj/k_proj/v_proj → per-channel INT8 对称量化 (W8, 同 SmoothQuant 简化版)
o_proj/gate_proj/up_proj/down_proj → GPTQ W4 group_size=128
lm_head → 保持 FP16

支持通过 layer_policy 回调自定义每层量化策略。
"""

import re
import torch
import torch.nn as nn
from typing import Callable, Dict, Optional
from transformers import AutoModelForCausalLM

from ptq.quant.gptq import gptq_quantize_linear

ATTN_PROJ = {"q_proj", "k_proj", "v_proj"}
FFN_PROJ = {"o_proj", "gate_proj", "up_proj", "down_proj"}


def default_policy(layer_idx: int, layer_name: str, layer_short: str) -> str:
    """默认模块级混合精度：q/k/v → W8, o/gate/up/down → W4。"""
    if layer_short in FFN_PROJ:
        return "w4"
    elif layer_short in ATTN_PROJ:
        return "w8"
    else:
        return "skip"


def parse_layer_num(name: str) -> int:
    """从 layer name 提取层号，如 'model.layers.5.self_attn.q_proj' → 5。"""
    m = re.search(r'layers\.(\d+)', name)
    return int(m.group(1)) if m else -1


@torch.no_grad()
def _quantize_w8_perchannel(weight: torch.Tensor) -> Dict:
    """Per-channel INT8 对称量化，返回 w_q (int8) + q_scale。"""
    w = weight.data.float()
    amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    q_scale = amax / 127.0
    w_q = (w / q_scale).round().clamp(-128, 127).to(torch.int8)
    return {"w_q": w_q, "q_scale": q_scale, "method": "w8_perchannel"}


def _collect_tokens(layer_inputs: list, max_tokens: int = 2048) -> torch.Tensor:
    """从逐样本输入列表中收集最多 max_tokens 个 token，返回 (total_tokens, in_features)。"""
    chunks = []
    total = 0
    for t in layer_inputs:
        flat = t.view(-1, t.shape[-1])
        chunks.append(flat)
        total += flat.shape[0]
        if total >= max_tokens:
            break
    return torch.cat(chunks, dim=0)[:max_tokens]


@torch.no_grad()
def quantize_model_mixed_precision(
    model: AutoModelForCausalLM,
    calib_data: torch.Tensor,
    bits_w4: int = 4,
    group_size: int = 128,
    layer_policy: Optional[Callable[[int, str, str], str]] = None,
) -> Dict[str, Dict]:
    """混合精度量化，复用 GPTQ 前向收集 Hessian 的流水线。

    layer_policy(layer_idx, layer_name, layer_short) → "w4" | "w8" | "skip"
    若为 None，使用默认模块级策略（q/k/v→W8, o/gate/up/down→W4）。
    """
    import gc
    if layer_policy is None:
        layer_policy = default_policy

    device = next(model.parameters()).device
    model.eval()
    n_samples = calib_data.shape[0]

    linear_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, nn.Linear) and "lm_head" not in name]

    BATCH_LAYERS = 8
    MAX_CALIB_TOKENS = 4096
    quant_state = {}

    for batch_start in range(0, len(linear_layers), BATCH_LAYERS):
        batch_end = min(batch_start + BATCH_LAYERS, len(linear_layers))
        current_batch = linear_layers[batch_start:batch_end]
        batch_names = {name for name, _ in current_batch}
        batch_inputs = {name: [] for name in batch_names}
        batch_token_counts = {name: 0 for name in batch_names}

        def make_hook(layer_name):
            def hook(module, args, output):
                if batch_token_counts[layer_name] >= MAX_CALIB_TOKENS:
                    return
                x = args[0].detach().cpu()
                batch_inputs[layer_name].append(x)
                batch_token_counts[layer_name] += x.shape[0] * x.shape[1]
            return hook

        hooks = []
        for layer_name, layer_module in linear_layers:
            if layer_name in batch_names:
                hooks.append(layer_module.register_forward_hook(make_hook(layer_name)))

        torch.cuda.empty_cache()
        for i in range(0, n_samples):
            batch = calib_data[i:i+1].to(device)
            model.model(batch)
            del batch
            torch.cuda.empty_cache()

        for h in hooks:
            h.remove()

        for layer_idx, (layer_name, layer_module) in enumerate(current_batch):
            global_idx = batch_start + layer_idx
            layer_short = layer_name.split(".")[-1]
            action = layer_policy(global_idx, layer_name, layer_short)

            if action == "w4":
                print(f"  GPTQ-W4 layer {global_idx+1}/{len(linear_layers)}: {layer_name}", flush=True)
                layer_inputs = batch_inputs[layer_name]
                inp_gpu = _collect_tokens(layer_inputs, max_tokens=2048).float().to(device)
                qi = gptq_quantize_linear(layer_module, inp_gpu, bits=bits_w4, group_size=group_size)
                qi["method"] = "gptq_w4"
                quant_state[layer_name] = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in qi.items()}
                del inp_gpu, qi

            elif action == "w8":
                print(f"  INT8    layer {global_idx+1}/{len(linear_layers)}: {layer_name}", flush=True)
                qi = _quantize_w8_perchannel(layer_module.weight)
                quant_state[layer_name] = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in qi.items()}
                del qi

            else:
                print(f"  SKIP    layer {global_idx+1}/{len(linear_layers)}: {layer_name}", flush=True)

            if layer_name in batch_inputs:
                del batch_inputs[layer_name]
            gc.collect()
            torch.cuda.empty_cache()

    return quant_state


@torch.no_grad()
def apply_mixed_precision_to_model_gpu(
    model: AutoModelForCausalLM,
    quant_state: Dict[str, Dict],
) -> AutoModelForCausalLM:
    """GPU 上原地应用混合精度量化权重。"""
    import gc
    device = next(model.parameters()).device
    replaced = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in quant_state:
            continue

        qi = quant_state.pop(name)
        method = qi.get("method", "gptq_w4")

        if method == "gptq_w4":
            w_q = qi["w_q"].to(device)
            scale = qi["scale"].to(device)
            zero = qi["zero"].to(device)
            group_size = qi["group_size"]
            out_feat, in_feat = w_q.shape
            n_groups = (in_feat + group_size - 1) // group_size
            w_deq = torch.empty(out_feat, in_feat, device=device, dtype=torch.float16)
            for g in range(n_groups):
                g_start = g * group_size
                g_end = min(g_start + group_size, in_feat)
                s = scale[:, g:g+1].float()
                z = zero[:, g:g+1].float()
                w_deq[:, g_start:g_end] = (w_q[:, g_start:g_end].float() - z) * s
            module.weight = nn.Parameter(w_deq, requires_grad=False)
            del qi, w_q, scale, zero, w_deq

        elif method == "w8_perchannel":
            w_q = qi["w_q"].to(device)
            q_scale = qi["q_scale"].to(device)
            w_deq = (w_q.float() * q_scale.float()).to(torch.float16)
            module.weight = nn.Parameter(w_deq, requires_grad=False)
            del qi, w_q, q_scale, w_deq

        replaced += 1
        if replaced % 40 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()
    return model

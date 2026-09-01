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

from ptq.quant.gptq import (
    collect_linear_inputs,
    dequantize_gptq,
    gptq_quantize_linear,
    gptq_quantize_linear_multi,
)

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
    linear_layers, layer_inputs = collect_linear_inputs(
        model, calib_data, max_tokens=4096
    )
    quant_state = {}
    for layer_idx, (layer_name, layer_module) in enumerate(linear_layers):
        layer_short = layer_name.split(".")[-1]
        action = layer_policy(layer_idx, layer_name, layer_short)
        if action == "w4":
            print(f"  GPTQ-W4 layer {layer_idx+1}/{len(linear_layers)}: {layer_name}", flush=True)
            inp_gpu = layer_inputs[layer_name].to(device=device, dtype=torch.float32)
            qi = gptq_quantize_linear(
                layer_module, inp_gpu, bits=bits_w4, group_size=group_size
            )
            quant_state[layer_name] = {
                key: value.cpu() if hasattr(value, "cpu") else value
                for key, value in qi.items()
            }
            del inp_gpu, qi
        elif action == "w8":
            print(f"  INT8    layer {layer_idx+1}/{len(linear_layers)}: {layer_name}", flush=True)
            qi = _quantize_w8_perchannel(layer_module.weight)
            qi.update(
                {
                    "bits": 8,
                    "in_features": layer_module.in_features,
                    "out_features": layer_module.out_features,
                }
            )
            quant_state[layer_name] = {
                key: value.cpu() if hasattr(value, "cpu") else value
                for key, value in qi.items()
            }
            del qi
        else:
            print(f"  SKIP    layer {layer_idx+1}/{len(linear_layers)}: {layer_name}", flush=True)
        del layer_inputs[layer_name]
        gc.collect()
        torch.cuda.empty_cache()

    return quant_state


@torch.no_grad()
def quantize_model_precision_bank(
    model: AutoModelForCausalLM,
    calib_data: torch.Tensor,
    bits_w4: int = 4,
    group_size: int = 128,
    uniform_bits: tuple[int, ...] = (5, 6),
    max_calib_tokens: int = 4096,
) -> Dict[str, Dict[str, Dict]]:
    """Build reusable W4/W5/W6/W8 states from one activation capture.

    The returned bank lets allocation experiments reuse identical per-module
    quantization results. This removes calibration and quantizer variation from
    comparisons between SG-MMP, random allocations, and module controls.
    """
    import gc

    device = next(model.parameters()).device
    model.eval()
    linear_layers, layer_inputs = collect_linear_inputs(
        model, calib_data, max_tokens=max_calib_tokens
    )
    bank: Dict[str, Dict[str, Dict]] = {}
    all_gptq_bits = (bits_w4, *uniform_bits)
    for ordinal, (layer_name, layer_module) in enumerate(linear_layers, start=1):
        print(f"  BANK layer {ordinal}/{len(linear_layers)}: {layer_name}", flush=True)
        inp_gpu = layer_inputs.pop(layer_name).to(device=device, dtype=torch.float32)
        quantized = gptq_quantize_linear_multi(
            layer_module,
            inp_gpu,
            bits_values=all_gptq_bits,
            group_size=group_size,
        )
        w4 = quantized[bits_w4]
        w4_dequantized = dequantize_gptq(
            w4["w_q"], w4["scale"], w4["zero"], group_size
        ).float()
        original = layer_module.weight.detach().float()
        input_second_moment = inp_gpu.square().mean(dim=0).unsqueeze(0)
        signal = (original.square() * input_second_moment).mean().clamp(min=1e-12)
        noise = ((original - w4_dequantized).square() * input_second_moment).mean()
        hessian_diag_nmse = float((noise / signal).item())
        w8 = _quantize_w8_perchannel(layer_module.weight)
        w8.update(
            {
                "bits": 8,
                "in_features": layer_module.in_features,
                "out_features": layer_module.out_features,
            }
        )
        choices = {
            f"w{bits}": {
                key: value.cpu() if hasattr(value, "cpu") else value
                for key, value in info.items()
            }
            for bits, info in quantized.items()
        }
        choices["w8"] = {
            key: value.cpu() if hasattr(value, "cpu") else value
            for key, value in w8.items()
        }
        choices["scores"] = {
            "hessian_diag_reconstruction_nmse": hessian_diag_nmse,
        }
        bank[layer_name] = choices
        del (
            inp_gpu,
            input_second_moment,
            noise,
            original,
            signal,
            quantized,
            w4,
            w4_dequantized,
            w8,
        )
        gc.collect()
        torch.cuda.empty_cache()

    return bank


def compose_precision_state(
    bank: Dict[str, Dict[str, Dict]],
    layer_policy: Callable[[int, str, str], str],
) -> Dict[str, Dict]:
    """Materialize one precision allocation from a reusable bank."""
    state: Dict[str, Dict] = {}
    for module_idx, (name, choices) in enumerate(bank.items()):
        action = layer_policy(module_idx, name, name.split(".")[-1])
        if action == "skip":
            continue
        if action not in choices:
            available = sorted(key for key in choices if key != "scores")
            raise KeyError(
                f"Precision bank has no {action} entry for {name}; available={available}"
            )
        # Application functions pop top-level entries. A shallow copy keeps the
        # reusable bank intact while avoiding duplicate tensor storage in RAM.
        state[name] = dict(choices[action])
    return state


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

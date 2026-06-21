"""Hadamard 旋转预处理 + GPTQ 量化。

对权重矩阵做正交旋转变换打散离群值，再跑标准 GPTQ。
推理时反旋回去，无额外计算开销。
"""

import torch
import torch.nn as nn
import gc
from typing import Dict
from transformers import AutoModelForCausalLM
from scipy.linalg import hadamard

from ptq.quant.gptq import gptq_quantize_linear


def _next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()


@torch.no_grad()
def _hadamard_rotate_weight(weight: torch.Tensor, normalize: bool = True):
    """对权重矩阵做 Hadamard 双端旋转。"""
    out_feat, in_feat = weight.shape
    pad_in = _next_pow2(in_feat)
    pad_out = _next_pow2(out_feat)

    # 过大的 Hadamard 矩阵（>4096）会爆内存，跳过旋转直接量化
    if pad_in > 4096 or pad_out > 4096:
        return None  # 调用方处理
    H_in = torch.tensor(hadamard(pad_in), dtype=torch.float32)
    H_out = torch.tensor(hadamard(pad_out), dtype=torch.float32)
    if normalize:
        H_in = H_in / (pad_in ** 0.5)
        H_out = H_out / (pad_out ** 0.5)

    W_padded = torch.zeros(pad_out, pad_in, dtype=torch.float32)
    W_padded[:out_feat, :in_feat] = weight.float().cpu()

    W_rot = H_out @ W_padded @ H_in.T
    return W_rot, H_in, H_out, in_feat, out_feat, pad_in, pad_out


@torch.no_grad()
def quantize_model_hadamard_gptq(
    model: AutoModelForCausalLM,
    calib_data: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Dict[str, Dict]:
    """Hadamard 旋转 + GPTQ 量化。

    一次性前向捕获所有层激活（节省 N_batch × N_sample 次重复前向），
    然后对每层权重做 Hadamard 旋转 → GPTQ 量化 → 保存。
    """
    device = next(model.parameters()).device
    model.eval()
    n_samples, seq_len = calib_data.shape

    # 列出所有要量化的 Linear 层
    linear_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, nn.Linear) and "lm_head" not in name]

    # === 一次性捕获所有层输入激活 ===
    all_inputs = {}
    def make_hook(layer_name):
        def hook(module, args, output):
            # 只存前 MAX_TOKENS 个 token，截断省内存
            act = args[0].detach().cpu()  # (1, seq, hidden)
            MAX_TOKENS = 256
            all_inputs[layer_name] = act[:, :MAX_TOKENS, :].view(-1, act.shape[-1])
        return hook

    hooks = [module.register_forward_hook(make_hook(name))
             for name, module in linear_layers]

    torch.cuda.empty_cache()
    print(f"  Capturing activations ({n_samples} forward passes)...")
    for i in range(n_samples):
        batch = calib_data[i:i+1].to(device)
        model.model(batch)
        del batch
        if (i + 1) % 16 == 0:
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()
    torch.cuda.empty_cache()

    print(f"  Captured {len(all_inputs)}/{len(linear_layers)} layer activations")

    # === 逐层 Hadamard 旋转 + GPTQ 量化 ===
    quant_state = {}
    for idx, (layer_name, layer_module) in enumerate(linear_layers):
        if layer_name not in all_inputs:
            print(f"  SKIP {layer_name}: no activation captured")
            continue

        inp = all_inputs[layer_name]  # (≤256, in_features)
        print(f"  [{idx+1}/{len(linear_layers)}] {layer_name} "
              f"(inp={list(inp.shape)}, W={list(layer_module.weight.shape)})", flush=True)

        # 1. Hadamard 旋转权重（过大的层跳过旋转，直接用 GPTQ）
        W_orig = layer_module.weight.data.clone()
        rot_result = _hadamard_rotate_weight(W_orig)

        if rot_result is None:
            # 层太大（pad > 4096），跳过旋转直接 GPTQ
            print(f"    skip rotation (pad > 4096), direct GPTQ")
            inp_gpu = inp.float()[:1024].to(device)
            qi = gptq_quantize_linear(layer_module, inp_gpu, bits=bits, group_size=group_size)
            quant_state[layer_name] = {
                k: v.cpu() if hasattr(v, 'cpu') else v for k, v in qi.items()
            }
            # 标记无旋转
            quant_state[layer_name]["H_in"] = None
            quant_state[layer_name]["H_out"] = None
            del inp, inp_gpu, W_orig, qi
        else:
            W_rot, H_in, H_out, orig_in, orig_out, pad_in, pad_out = rot_result

            # 2. 旋转输入（补零 + H_in.T）
            inp_padded = torch.zeros(inp.shape[0], pad_in, dtype=torch.float32)
            inp_padded[:, :orig_in] = inp.float()
            inp_rot = inp_padded @ H_in.T

            # 3. 标准 GPTQ
            W_rot_gpu = W_rot.to(device)
            rot_layer = nn.Linear(pad_in, pad_out, bias=False)
            rot_layer.weight.data = W_rot_gpu
            inp_rot_gpu = inp_rot.to(device)

            qi = gptq_quantize_linear(rot_layer, inp_rot_gpu, bits=bits, group_size=group_size)

            # 4. 保存
            quant_state[layer_name] = {
                k: v.cpu() if hasattr(v, 'cpu') else v for k, v in qi.items()
            }
            quant_state[layer_name]["H_in"] = H_in
            quant_state[layer_name]["H_out"] = H_out
            quant_state[layer_name]["orig_in_features"] = orig_in
            quant_state[layer_name]["orig_out_features"] = orig_out
            quant_state[layer_name]["pad_in"] = pad_in
            quant_state[layer_name]["pad_out"] = pad_out

            del inp, inp_padded, inp_rot, inp_rot_gpu, W_orig, W_rot, W_rot_gpu, H_in, H_out, qi, rot_layer
        gc.collect()
        torch.cuda.empty_cache()

    return quant_state


@torch.no_grad()
def apply_hadamard_gptq_to_model_gpu(model, quant_state: Dict[str, Dict]):
    """将 Hadamard+GPTQ 量化应用到 GPU 模型。

    去量化 → 反旋转 → 裁 padding → 替换权重。
    """
    device = next(model.parameters()).device
    replaced = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in quant_state:
            continue

        qi = quant_state.pop(name)
        w_q = qi["w_q"].to(device)
        scale = qi["scale"].to(device)
        zero = qi["zero"].to(device)
        group_size = qi["group_size"]

        # GPTQ 去量化
        out_feat, in_feat = w_q.shape
        n_groups = (in_feat + group_size - 1) // group_size
        w_rot_deq = torch.empty(out_feat, in_feat, device=device, dtype=torch.float32)
        for g in range(n_groups):
            g_start = g * group_size
            g_end = min(g_start + group_size, in_feat)
            s = scale[:, g:g+1].float()
            z = zero[:, g:g+1].float()
            w_rot_deq[:, g_start:g_end] = (w_q[:, g_start:g_end].float() - z) * s

        # 反旋转（如果该层做了旋转）
        if qi.get("H_in") is not None:
            H_in_dev = qi["H_in"].to(device)
            H_out_dev = qi["H_out"].to(device)
            orig_in = qi["orig_in_features"]
            orig_out = qi["orig_out_features"]
            w_deq = H_out_dev.T @ w_rot_deq @ H_in_dev
            w_deq = w_deq[:orig_out, :orig_in].to(torch.float16)
            del H_in_dev, H_out_dev
        else:
            # 无旋转，直接使用去量化权重（已是原始尺寸）
            w_deq = w_rot_deq.to(torch.float16)

        module.weight = nn.Parameter(w_deq, requires_grad=False)

        del qi, w_q, scale, zero, w_rot_deq, w_deq
        replaced += 1
        if replaced % 40 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()
    return model

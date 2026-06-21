"""AWQ 量化自定义实现。

参考: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration" (Lin et al., 2024)

核心算法：
1. 用校准数据收集每层激活幅度
2. 按通道计算最佳 per-channel scaling factor（通过 grid search）
3. 将 scaling 应用到权重上，再做 RTN 量化
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from transformers import AutoModelForCausalLM


def _compute_awq_scale(
    w: torch.Tensor,
    act_max: torch.Tensor,  # (in_features,)
    bits: int = 4,
    group_size: int = 128,
) -> torch.Tensor:
    """计算 AWQ 的 per-channel scale。

    简化版：对 top-10% 激活通道用固定 scale=2.0，其余保持 1.0。
    原始 AWQ 用 grid search，这里用固定因子加速。
    """
    in_feat = w.shape[1]
    best_scale = torch.ones(in_feat, device=w.device, dtype=torch.float32)

    k = max(1, in_feat // 10)
    _, topk_indices = act_max.topk(k)
    best_scale[topk_indices] = 2.0

    return best_scale


def _rtn_quant(w: torch.Tensor, bits: int, group_size: int):
    qmin, qmax = 0, 2**bits - 1
    out_feat, in_feat = w.shape
    n_groups = (in_feat + group_size - 1) // group_size
    padded = n_groups * group_size

    w_p = torch.zeros(out_feat, padded, device=w.device, dtype=w.dtype)
    w_p[:, :in_feat] = w
    w_reshaped = w_p.view(out_feat, n_groups, group_size)

    w_min = w_reshaped.amin(dim=-1, keepdim=True)
    w_max = w_reshaped.amax(dim=-1, keepdim=True)
    scale = (w_max - w_min).clamp(min=1e-8) / (qmax - qmin)
    zero = (qmin - w_min / scale).round().clamp(qmin, qmax)

    q = (w_reshaped / scale + zero).round().clamp(qmin, qmax)
    q = q.view(out_feat, padded)[:, :in_feat].contiguous()
    return q, scale.squeeze(-1), zero.squeeze(-1)


def _rtn_dequant(q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, group_size: int):
    out_feat, in_feat = q.shape
    n_groups = (in_feat + group_size - 1) // group_size
    padded = n_groups * group_size

    q_p = torch.zeros(out_feat, padded, device=q.device, dtype=q.dtype)
    q_p[:, :in_feat] = q
    q_r = q_p.view(out_feat, n_groups, group_size)

    dq = (q_r.float() - zero.unsqueeze(-1).float()) * scale.unsqueeze(-1).float()
    dq = dq.view(out_feat, padded)[:, :in_feat].contiguous()
    return dq


class AWQLinear(nn.Module):
    """AWQ 量化 Linear 层。去量化权重缓存。"""

    def __init__(self, quant_info: Dict, bias: Optional[nn.Parameter], awq_scale: torch.Tensor, dtype=torch.float16):
        super().__init__()
        self.in_features = quant_info["in_features"]
        self.out_features = quant_info["out_features"]
        self.bias = bias
        self.dtype = dtype

        w_deq = _rtn_dequant(
            quant_info["w_q"], quant_info["scale"], quant_info["zero"], quant_info["group_size"]
        ).to(dtype)
        # 除以 awq_scale（量化时乘了，推理时除回来）
        w_deq = w_deq / awq_scale.unsqueeze(0).to(dtype)
        self.register_buffer("weight_deq", w_deq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight_deq.to(x.dtype), self.bias)


@torch.no_grad()
def quantize_model_awq(
    model: AutoModelForCausalLM,
    calib_data: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Dict[str, Dict]:
    """逐层 AWQ 量化（分批收集激活，跳过 lm_head 省显存）。"""
    import gc
    device = next(model.parameters()).device
    model.eval()
    n_samples = calib_data.shape[0]
    calib_batch_size = 4

    linear_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, nn.Linear) and "lm_head" not in name]

    BATCH_LAYERS = 8

    quant_state = {}
    awq_scales = {}

    for batch_start in range(0, len(linear_layers), BATCH_LAYERS):
        batch_end = min(batch_start + BATCH_LAYERS, len(linear_layers))
        current_batch_layers = linear_layers[batch_start:batch_end]
        batch_names = {name for name, _ in current_batch_layers}

        batch_inputs = {name: [] for name in batch_names}

        def make_hook(layer_name):
            def hook(module, args, output):
                batch_inputs[layer_name].append(args[0].detach().cpu())
            return hook

        hooks = []
        for layer_name, layer_module in linear_layers:
            if layer_name in batch_names:
                hooks.append(layer_module.register_forward_hook(make_hook(layer_name)))

        torch.cuda.empty_cache()
        for i in range(0, n_samples, 1):  # batch=1 避免 SDPA OOM
            batch = calib_data[i:i+1].to(device)
            model.model(batch)
            del batch
            torch.cuda.empty_cache()

        for h in hooks:
            h.remove()

        for layer_idx, (layer_name, layer_module) in enumerate(current_batch_layers):
            global_idx = batch_start + layer_idx
            layer_inputs = batch_inputs[layer_name]
            acts = torch.cat(layer_inputs, dim=0).view(-1, layer_inputs[0].shape[-1])
            acts = acts[:2048].to(device)
            act_max = acts.abs().amax(dim=0)

            print(f"  AWQ layer {global_idx+1}/{len(linear_layers)}: {layer_name}", flush=True)

            awq_s = _compute_awq_scale(layer_module.weight.data, act_max, bits=bits, group_size=group_size)
            w_scaled = layer_module.weight.data.float() * awq_s.unsqueeze(0)
            w_q, scale, zero = _rtn_quant(w_scaled, bits, group_size)

            quant_state[layer_name] = {
                "w_q": w_q.cpu(),
                "scale": scale.cpu(),
                "zero": zero.cpu(),
                "group_size": group_size,
                "in_features": layer_module.in_features,
                "out_features": layer_module.out_features,
            }
            awq_scales[layer_name] = awq_s.cpu()

            del acts, w_q, scale, zero, awq_s, batch_inputs[layer_name]
            gc.collect()
            torch.cuda.empty_cache()

    return quant_state, awq_scales


def apply_awq_to_model(model: AutoModelForCausalLM, quant_state: Dict, awq_scales: Dict) -> AutoModelForCausalLM:
    """将 AWQ 量化应用到模型。"""
    for name, module in model.named_modules():
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1] if "." in name else name

        if isinstance(module, nn.Linear) and name in quant_state:
            parent = model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            awq_linear = AWQLinear(quant_state[name], module.bias, awq_scales[name])
            setattr(parent, child_name, awq_linear)

    return model


@torch.no_grad()
def apply_awq_to_model_gpu(model: AutoModelForCausalLM, quant_state: Dict, awq_scales: Dict) -> AutoModelForCausalLM:
    """将 AWQ 量化应用到已加载到 GPU 的模型（原地替换权重，零额外显存）。"""
    import gc
    device = next(model.parameters()).device
    replaced = 0

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in quant_state:
            continue

        qi = quant_state.pop(name)
        awq_scale = awq_scales.pop(name).to(device)

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
            s = scale[:, g : g + 1].float()
            z = zero[:, g : g + 1].float()
            w_deq[:, g_start:g_end] = (w_q[:, g_start:g_end].float() - z) * s

        w_deq = w_deq / awq_scale.unsqueeze(0).to(torch.float16)
        module.weight = nn.Parameter(w_deq, requires_grad=False)

        del qi, awq_scale, w_q, scale, zero, w_deq
        replaced += 1
        if replaced % 40 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()
    return model

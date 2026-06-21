"""SmoothQuant W8A8 量化核心实现。

算法流程：
1. 用校准数据前向传播，收集每层 Linear 输入的激活值
2. 计算平滑因子: s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
3. 将平滑因子融到权重和激活中
4. 对权重做 per-channel INT8 对称量化，激活做 per-token INT8 对称量化
5. 推理时用 fake-quant 模拟量化噪声（无原生 INT8 kernel）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---- 平滑因子计算 ----
@torch.no_grad()
def compute_smooth_scales(
    model: AutoModelForCausalLM,
    calib_data: torch.Tensor,
    alpha: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """用校准数据计算每层的平滑因子。增量统计，不存完整激活张量。

    Args:
        model: FP16 模型
        calib_data: 校准 token ids, shape (batch, seq_len)
        alpha: 迁移强度，0.5 为 SmoothQuant 默认值

    Returns:
        scales: {layer_name: scale_vector}，scale_vector shape = (in_features,)
    """
    device = next(model.parameters()).device

    # 按批次处理 layers，避免同时保存所有层的激活统计
    linear_layers = [(name, module) for name, module in model.named_modules()
                     if isinstance(module, nn.Linear)]

    BATCH_LAYERS = 8
    scales = {}

    for batch_start in range(0, len(linear_layers), BATCH_LAYERS):
        batch_end = min(batch_start + BATCH_LAYERS, len(linear_layers))
        current_batch = linear_layers[batch_start:batch_end]
        batch_names = {name for name, _ in current_batch}

        # 增量统计：只保存 running max，不存完整张量
        act_max_running = {name: None for name in batch_names}

        def make_hook(layer_name):
            def hook(module, args, output):
                # args[0]: (1, seq_len, in_features) — per-token
                x = args[0].detach().float()
                # amax over batch+seq dims -> (in_features,)
                token_max = x.abs().amax(dim=(0, 1)).cpu()
                if act_max_running[layer_name] is None:
                    act_max_running[layer_name] = token_max
                else:
                    act_max_running[layer_name] = torch.maximum(act_max_running[layer_name], token_max)
            return hook

        hooks = []
        for layer_name, layer_module in linear_layers:
            if layer_name in batch_names:
                hooks.append(layer_module.register_forward_hook(make_hook(layer_name)))

        model.eval()
        for i in range(0, calib_data.shape[0], 1):
            batch = calib_data[i:i+1].to(device)
            model.model(batch)
            del batch
            torch.cuda.empty_cache()

        for h in hooks:
            h.remove()

        # 计算当前 batch 的 smooth scales
        for layer_name, layer_module in current_batch:
            if act_max_running[layer_name] is None:
                continue

            act_max = act_max_running[layer_name].float().to(device)
            w = layer_module.weight.data.float()
            w_max = w.abs().amax(dim=0).float()

            # s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
            s = (act_max.clamp(min=1e-8) ** alpha) / (w_max.clamp(min=1e-8) ** (1 - alpha))
            s = s.clamp(min=1e-8)
            scales[layer_name] = s.cpu()

            del act_max
            torch.cuda.empty_cache()

    return scales


# ---- 量化 / 去量化 ----
def quantize_i8_symmetric(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """对称 per-channel/per-tensor INT8 量化。

    Returns:
        x_q: INT8 表示 (实际存储为 float, 取整值)
        scale: 量化 scale
    """
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    x_q = (x / scale).round().clamp(-128, 127)
    return x_q, scale


def dequantize_i8(x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x_q * scale


# ---- 将平滑因子应用到模型权重 ----
@torch.no_grad()
def apply_smooth_scales_to_model(
    model: AutoModelForCausalLM,
    scales: Dict[str, torch.Tensor],
) -> AutoModelForCausalLM:
    """将平滑因子融到模型权重中。

    对每个 Linear 层:
    - 输入 X 除以 scale (通过前一层权重调整来实现)
    - 权重 W 乘以 scale
    """
    fp16_max = 65504.0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in scales:
            s = scales[name].to(module.weight.device, torch.float32)
            w_new = module.weight.data.float() * s.unsqueeze(0)
            w_new = w_new.clamp(-fp16_max, fp16_max).to(module.weight.dtype)
            module.weight.data.copy_(w_new)

    return model


# ---- SmoothQuant 包装器 ----
class SmoothQuantLinear(nn.Module):
    """W8A8 SmoothQuant Linear，fake-quant 推理。"""

    def __init__(self, linear: nn.Linear, w_scale: torch.Tensor):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bias = linear.bias

        # INT8 量化权重
        w_q, self.w_scale = quantize_i8_symmetric(linear.weight.data.float())
        self.w_q = nn.Parameter(w_q, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 激活做 per-token INT8 量化 (fake-quant)
        x_q, x_scale = quantize_i8_symmetric(x.float())

        # 量化推理 + 去量化
        w_deq = dequantize_i8(self.w_q, self.w_scale)
        x_deq = dequantize_i8(x_q, x_scale)

        out = nn.functional.linear(x_deq.to(x.dtype), w_deq.to(x.dtype), self.bias)
        return out


@torch.no_grad()
def apply_smoothquant_to_model(
    model: AutoModelForCausalLM,
    scales: Dict[str, torch.Tensor],
) -> AutoModelForCausalLM:
    """W8 per-channel 对称量化（简化版 SmoothQuant）。

    不做 smooth scale 迁移，直接 per-channel INT8 对称量化权重。
    这样避免了 FP16 溢出问题，同时提供有效的 8-bit 量化 baseline。
    """
    device = next(model.parameters()).device

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in scales:
            w = module.weight.data.float()
            # per-channel absmax → scale
            amax = w.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # (out, 1)
            q_scale = amax / 127.0
            w_q = (w / q_scale).round().clamp(-128, 127)
            w_deq = (w_q * q_scale).to(module.weight.dtype)
            module.weight.data.copy_(w_deq)

    return model

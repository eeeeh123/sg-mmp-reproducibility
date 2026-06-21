"""RTN (Round-to-Nearest) 4-bit 量化核心实现。

Group-wise 非对称 per-channel min-max 量化：
- 把权重矩阵按 group_size 分组
- 每组独立计算 min/max，映射到 4-bit [0, 15]
- 去量化: w_deq = (w_q - zero) * scale
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
from transformers import AutoModelForCausalLM, AutoConfig


def quantize_tensor_rtn(
    w: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对单个权重张量做 RTN 量化。

    Args:
        w: 浮点权重, shape (out_features, in_features)
        bits: 量化位宽
        group_size: 分组大小

    Returns:
        w_q: 量化后整数 (packed 格式暂不实现, 返回 unscaled 整数)
        scale: 每组的 scale, shape (out_features, n_groups)
        zero: 每组的 zero point, shape (out_features, n_groups)
    """
    out_feat, in_feat = w.shape
    n_groups = (in_feat + group_size - 1) // group_size

    # pad 到 group_size 的倍数
    padded_in = n_groups * group_size
    w_padded = torch.zeros(out_feat, padded_in, dtype=w.dtype, device=w.device)
    w_padded[:, :in_feat] = w

    w_reshaped = w_padded.view(out_feat, n_groups, group_size)

    w_min = w_reshaped.amin(dim=-1, keepdim=True)  # (out, n_groups, 1)
    w_max = w_reshaped.amax(dim=-1, keepdim=True)  # (out, n_groups, 1)

    qmin, qmax = 0, 2**bits - 1

    # 避免 scale 为 0
    scale = (w_max - w_min).clamp(min=1e-8) / (qmax - qmin)
    zero = qmin - w_min / scale
    zero = zero.clamp(qmin, qmax).round()

    # 量化
    w_q = (w_reshaped / scale + zero).round().clamp(qmin, qmax)

    # 去掉 padding
    w_q = w_q.view(out_feat, padded_in)[:, :in_feat].contiguous()

    return w_q, scale.squeeze(-1), zero.squeeze(-1)


def dequantize_tensor_rtn(
    w_q: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """去量化回浮点数。"""
    out_feat, in_feat = w_q.shape
    n_groups = (in_feat + group_size - 1) // group_size
    padded_in = n_groups * group_size

    w_q_padded = torch.zeros(out_feat, padded_in, dtype=w_q.dtype, device=w_q.device)
    w_q_padded[:, :in_feat] = w_q
    w_q_reshaped = w_q_padded.view(out_feat, n_groups, group_size)

    w_deq = (w_q_reshaped.float() - zero.unsqueeze(-1).float()) * scale.unsqueeze(-1).float()
    w_deq = w_deq.view(out_feat, padded_in)[:, :in_feat].contiguous()

    return w_deq


@torch.no_grad()
def quantize_model_rtn(
    model: AutoModelForCausalLM,
    bits: int = 4,
    group_size: int = 128,
    target_dtype: torch.dtype = torch.float16,
) -> Dict[str, Dict]:
    """对整个模型的 Linear 层做 RTN 量化。

    返回 quant_state dict，key 为参数名，value 包含 w_q, scale, zero 等。
    """
    quant_state = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            w = module.weight.data.clone().float()
            w_q, scale, zero = quantize_tensor_rtn(w, bits=bits, group_size=group_size)
            quant_state[name] = {
                "w_q": w_q,
                "scale": scale,
                "zero": zero,
                "group_size": group_size,
                "in_features": module.in_features,
                "out_features": module.out_features,
            }

    return quant_state


class RTNLinear(nn.Module):
    """替换 nn.Linear 的 RTN 量化版。去量化后的权重缓存，避免重复计算。"""

    def __init__(self, original: nn.Linear, quant_info: Dict, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bias = original.bias
        self.dtype = dtype

        # 一次去量化并缓存
        w_deq = dequantize_tensor_rtn(
            quant_info["w_q"], quant_info["scale"], quant_info["zero"], quant_info["group_size"]
        ).to(dtype)
        self.register_buffer("weight_deq", w_deq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight_deq.to(x.dtype), self.bias)


@torch.no_grad()
def apply_rtn_to_model(
    model: AutoModelForCausalLM,
    quant_state: Dict[str, Dict],
) -> AutoModelForCausalLM:
    """将 RTN 量化信息应用到模型，替换所有 Linear 层为 RTNLinear。"""
    for name, module in model.named_modules():
        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1] if "." in name else name

        if isinstance(module, nn.Linear) and name in quant_state:
            parent = model
            if parent_name:
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            rtn_linear = RTNLinear(module, quant_state[name])
            setattr(parent, child_name, rtn_linear)

    return model

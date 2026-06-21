"""GPTQ 量化自定义实现。

参考: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers" (Frantar et al., 2023)

核心算法：
1. 用校准数据计算每层权重的逆 Hessian H^(-1)
2. 逐列量化权重，用 H^(-1) 补偿剩余列的量化误差
3. group-wise 量化，group_size=128
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from transformers import AutoModelForCausalLM


@torch.no_grad()
def _hessian_from_calib(
    layer: nn.Linear,
    inp: torch.Tensor,
) -> torch.Tensor:
    """计算一个 Linear 层输入的 Hessian 近似 H = X^T X。

    inp: (total_tokens, in_features)
    返回: H (in_features, in_features)
    """
    H = inp.t().matmul(inp)  # (in_features, in_features)
    # 添加 damping 保证正定性
    H.diagonal().add_(0.01 * H.diagonal().mean())
    return H


@torch.no_grad()
def gptq_quantize_linear(
    layer: nn.Linear,
    inp: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Dict:
    """对单个 Linear 层做 GPTQ 量化。

    Args:
        layer: nn.Linear 层
        inp: 该层的输入激活, shape (total_tokens, in_features)
        bits: 量化位宽
        group_size: 分组大小

    Returns:
        quant_info: {w_q, scale, zero, group_size}
    """
    W = layer.weight.data.clone().float()  # (out_features, in_features)
    out_feat, in_feat = W.shape
    dev = W.device

    H = _hessian_from_calib(layer, inp.float().cpu()).cpu()
    try:
        L = torch.linalg.cholesky(H)
    except RuntimeError:
        H.diagonal().add_(0.1 * H.diagonal().mean())
        L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L).to(dev)

    Q = torch.zeros_like(W)
    qmin, qmax = 0, 2**bits - 1

    # 保存 scale / zero
    n_groups = (in_feat + group_size - 1) // group_size
    scales = torch.zeros(out_feat, n_groups, device=dev)
    zeros = torch.zeros(out_feat, n_groups, device=dev)

    # 逐列量化（in_features 方向）
    dead = torch.zeros(in_feat, dtype=torch.bool, device=dev)

    for i in range(in_feat):
        if i % group_size == 0:
            group_idx = i // group_size
            g_end = min(i + group_size, in_feat)
            # 计算当前 group 的 scale / zero
            w_group = W[:, i:g_end]
            w_min = w_group.amin(dim=-1, keepdim=True)
            w_max = w_group.amax(dim=-1, keepdim=True)
            scale = (w_max - w_min).clamp(min=1e-8) / (qmax - qmin)
            zero = qmin - w_min / scale
            zero = zero.clamp(qmin, qmax).round()
            scales[:, group_idx] = scale.squeeze(-1)
            zeros[:, group_idx] = zero.squeeze(-1)

        # 量化第 i 列
        group_idx = i // group_size
        scale_i = scales[:, group_idx : group_idx + 1]  # (out, 1)
        zero_i = zeros[:, group_idx : group_idx + 1]  # (out, 1)

        w_col = W[:, i : i + 1]  # (out, 1)
        q_col = (w_col / scale_i + zero_i).round().clamp(qmin, qmax)
        dq_col = (q_col - zero_i) * scale_i  # 去量化值

        Q[:, i : i + 1] = q_col

        # 量化误差
        err = w_col - dq_col  # (out, 1)

        # 用 H_inv 补偿剩余的列
        # 剩余列 j > i: W[:, j] -= err * H_inv[i, j] / H_inv[i, i]
        if i < in_feat - 1:
            remaining = torch.arange(i + 1, in_feat, device=dev)
            H_inv_row = H_inv[i, remaining]  # (remaining_len,)
            H_inv_ii = H_inv[i, i]
            correction = err @ (H_inv_row.unsqueeze(0) / H_inv_ii)  # (out, remaining_len)
            W[:, remaining] -= correction

    # GPTQ codes are in [0, 2**bits - 1]. int8 is enough for 4-bit and avoids
    # oversized state files that can crash PyTorch/CUDA on Windows when loaded.
    w_q = Q.to(torch.int8)
    return {
        "w_q": w_q,
        "scale": scales,
        "zero": zeros,
        "group_size": group_size,
        "in_features": in_feat,
        "out_features": out_feat,
    }


def dequantize_gptq(w_q: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, group_size: int) -> torch.Tensor:
    """去量化 GPTQ 权重。"""
    out_feat, in_feat = w_q.shape
    n_groups = (in_feat + group_size - 1) // group_size

    w_deq = torch.zeros(out_feat, in_feat, device=w_q.device, dtype=torch.float32)
    for g in range(n_groups):
        g_start = g * group_size
        g_end = min(g_start + group_size, in_feat)
        s = scale[:, g : g + 1]
        z = zero[:, g : g + 1]
        w_deq[:, g_start:g_end] = (w_q[:, g_start:g_end].float() - z.float()) * s.float()

    return w_deq


class GPTQLinear(nn.Module):
    """GPTQ 量化 Linear 层。去量化权重缓存。"""

    def __init__(self, quant_info: Dict, bias: Optional[nn.Parameter], dtype: torch.dtype = torch.float16):
        super().__init__()
        self.in_features = quant_info["in_features"]
        self.out_features = quant_info["out_features"]
        self.bias = bias
        self.dtype = dtype

        w_deq = dequantize_gptq(
            quant_info["w_q"], quant_info["scale"], quant_info["zero"], quant_info["group_size"]
        ).to(dtype)
        self.register_buffer("weight_deq", w_deq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight_deq.to(x.dtype), self.bias)


@torch.no_grad()
def quantize_model_gptq(
    model: AutoModelForCausalLM,
    calib_data: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Dict[str, Dict]:
    """逐层 GPTQ 量化（分批前向：每 BATCH_LAYERS 层一组，避免 CPU OOM）。"""
    import gc
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
        current_batch_layers = linear_layers[batch_start:batch_end]
        batch_names = {name for name, _ in current_batch_layers}

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

        for layer_idx, (layer_name, layer_module) in enumerate(current_batch_layers):
            global_idx = batch_start + layer_idx
            layer_inputs = batch_inputs[layer_name]
            inp = torch.cat(layer_inputs, dim=0).view(-1, layer_inputs[0].shape[-1])
            inp_gpu = inp[:2048].float().to(device)

            print(f"  GPTQ layer {global_idx+1}/{len(linear_layers)}: {layer_name}", flush=True)

            qi = gptq_quantize_linear(layer_module, inp_gpu, bits=bits, group_size=group_size)
            quant_state[layer_name] = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in qi.items()}

            del inp, inp_gpu, qi, batch_inputs[layer_name]
            gc.collect()
            torch.cuda.empty_cache()

    return quant_state


def _replace_module(model, full_name: str, new_module: nn.Module):
    """在模型中替换指定名称的子模块。"""
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_gptq_to_model(model: AutoModelForCausalLM, quant_state: Dict[str, Dict]) -> AutoModelForCausalLM:
    """将 GPTQ 量化应用到模型。逐层替换，每层清理旧权重释放内存。"""
    import gc
    replaced = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name not in quant_state:
            continue

        parent_name = ".".join(name.split(".")[:-1])
        child_name = name.split(".")[-1] if "." in name else name

        parent = model
        if parent_name:
            for part in parent_name.split("."):
                parent = getattr(parent, part)

        qi = quant_state.pop(name)
        gptq_linear = GPTQLinear(qi, module.bias)
        del qi

        # 清除旧 Linear 模块的权重引用，避免 CPU 内存堆积
        if hasattr(module, 'weight') and module.weight is not None:
            module.weight.data = module.weight.data.cpu()
            del module.weight
            module.weight = None

        setattr(parent, child_name, gptq_linear)
        replaced += 1

        if replaced % 20 == 0:
            gc.collect()

    gc.collect()
    return model


@torch.no_grad()
def apply_gptq_to_model_gpu(model: AutoModelForCausalLM, quant_state: Dict[str, Dict]) -> AutoModelForCausalLM:
    """将 GPTQ 量化应用到已加载到 GPU 的模型（原地替换权重，零额外显存）。

    用于 8GB 显存上跑 1.7B 模型的场景，避免 CPU 去量化后搬 GPU 的 OOM。
    """
    import gc
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

        out_feat, in_feat = w_q.shape
        n_groups = (in_feat + group_size - 1) // group_size
        w_deq = torch.empty(out_feat, in_feat, device=device, dtype=torch.float16)
        for g in range(n_groups):
            g_start = g * group_size
            g_end = min(g_start + group_size, in_feat)
            s = scale[:, g : g + 1].float()
            z = zero[:, g : g + 1].float()
            w_deq[:, g_start:g_end] = (w_q[:, g_start:g_end].float() - z) * s

        module.weight = nn.Parameter(w_deq, requires_grad=False)

        del qi, w_q, scale, zero, w_deq
        replaced += 1
        if replaced % 40 == 0:
            torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    gc.collect()
    return model

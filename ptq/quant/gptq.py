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
    H = inp.t().matmul(inp)  # (in_features, in_features), on inp.device
    # 添加 damping 保证正定性
    damping = 0.01 * H.diagonal().mean().abs().clamp(min=1e-6)
    H.diagonal().add_(damping)
    return H


@torch.no_grad()
def collect_linear_inputs(
    model: AutoModelForCausalLM,
    calib_data: torch.Tensor,
    max_tokens: int = 4096,
) -> tuple[list[tuple[str, nn.Linear]], Dict[str, torch.Tensor]]:
    """Collect a deterministic, sample-balanced activation reservoir once.

    Every calibration sequence contributes either ``floor`` or ``ceil`` of the
    fixed token budget. Hooks cover all eligible linear modules in one model
    pass, avoiding the previous full-model replay for every eight modules.
    """
    if calib_data.ndim != 2 or calib_data.shape[0] <= 0:
        raise ValueError("calib_data must have shape (samples, sequence_length)")
    n_samples = int(calib_data.shape[0])
    if max_tokens < n_samples:
        raise ValueError("max_tokens must be at least the calibration sample count")
    device = next(model.parameters()).device
    model.eval()
    linear_layers = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and "lm_head" not in name
    ]
    collected: Dict[str, list[torch.Tensor]] = {name: [] for name, _ in linear_layers}
    current_quota = {"tokens": 0}

    def make_hook(layer_name: str):
        def hook(module, args, output):
            quota = current_quota["tokens"]
            if quota <= 0:
                return
            flat = args[0].detach().reshape(-1, args[0].shape[-1])
            if quota >= flat.shape[0]:
                chosen = flat
            else:
                positions = torch.linspace(
                    0, flat.shape[0] - 1, quota, device=flat.device
                ).round().long()
                chosen = flat.index_select(0, positions)
            collected[layer_name].append(chosen.to("cpu"))

        return hook

    hooks = [module.register_forward_hook(make_hook(name)) for name, module in linear_layers]
    base_quota, remainder = divmod(max_tokens, n_samples)
    try:
        for index in range(n_samples):
            current_quota["tokens"] = base_quota + int(index < remainder)
            batch = calib_data[index : index + 1].to(device)
            model.model(input_ids=batch, use_cache=False)
            del batch
            if (index + 1) % 16 == 0 or index + 1 == n_samples:
                print(f"  calibration capture {index + 1}/{n_samples}", flush=True)
    finally:
        for hook in hooks:
            hook.remove()

    merged: Dict[str, torch.Tensor] = {}
    for name, _ in linear_layers:
        parts = collected.pop(name)
        if not parts:
            raise RuntimeError(f"No calibration activations captured for {name}")
        merged[name] = torch.cat(parts, dim=0)[:max_tokens]
        del parts
        if merged[name].shape[0] != max_tokens:
            raise RuntimeError(
                f"Captured {merged[name].shape[0]}/{max_tokens} tokens for {name}"
            )
    return linear_layers, merged


@torch.no_grad()
def gptq_quantize_linear_multi(
    layer: nn.Linear,
    inp: torch.Tensor,
    bits_values: tuple[int, ...] = (4,),
    group_size: int = 128,
) -> Dict[int, Dict]:
    """Quantize one linear layer at multiple bits with one Hessian inverse.

    Args:
        layer: nn.Linear 层
        inp: 该层的输入激活, shape (total_tokens, in_features)
        bits_values: one or more quantization bit widths from 2 through 7
        group_size: 分组大小
    """
    bits_values = tuple(dict.fromkeys(int(bits) for bits in bits_values))
    if not bits_values or any(bits < 2 or bits > 7 for bits in bits_values):
        raise ValueError("GPTQ code tensors support bit widths from 2 through 7")
    original = layer.weight.data.detach().float()
    out_feat, in_feat = original.shape
    dev = original.device

    # On a 24-GiB server GPU, keeping this matrix on-device avoids making the
    # Cholesky decomposition the dominant CPU bottleneck. Only one module's
    # Hessian is live at a time.
    H = _hessian_from_calib(layer, inp.float().to(dev))
    try:
        L = torch.linalg.cholesky(H)
    except RuntimeError:
        H.diagonal().add_(0.1 * H.diagonal().mean().abs().clamp(min=1e-6))
        L = torch.linalg.cholesky(H)
    H_inv = torch.cholesky_inverse(L)
    del H, L
    results: Dict[int, Dict] = {}
    for bits in bits_values:
        W = original.clone()
        Q = torch.empty((out_feat, in_feat), device=dev, dtype=torch.int8)
        qmin, qmax = 0, 2**bits - 1
        n_groups = (in_feat + group_size - 1) // group_size
        scales = torch.zeros(out_feat, n_groups, device=dev)
        zeros = torch.zeros(out_feat, n_groups, device=dev)
        # A group-aligned block update is algebraically equivalent to the
        # column-wise update, but applies cross-block corrections with GEMM.
        # This removes thousands of wide rank-one kernel launches per module.
        for block_start in range(0, in_feat, group_size):
            block_end = min(block_start + group_size, in_feat)
            block_width = block_end - block_start
            W_block = W[:, block_start:block_end].clone()
            H_block = H_inv[block_start:block_end, block_start:block_end]
            normalized_errors = torch.zeros_like(W_block)
            group_idx = block_start // group_size
            weight_min = W_block.amin(dim=-1, keepdim=True)
            weight_max = W_block.amax(dim=-1, keepdim=True)
            scale = (weight_max - weight_min).clamp(min=1e-8) / (qmax - qmin)
            zero = (qmin - weight_min / scale).clamp(qmin, qmax).round()
            scales[:, group_idx] = scale.squeeze(-1)
            zeros[:, group_idx] = zero.squeeze(-1)

            for local_index in range(block_width):
                weight_column = W_block[:, local_index : local_index + 1]
                quantized_column = (
                    weight_column / scale + zero
                ).round().clamp(qmin, qmax)
                dequantized_column = (quantized_column - zero) * scale
                Q[:, block_start + local_index : block_start + local_index + 1] = quantized_column
                diagonal = H_block[local_index, local_index].clamp(min=1e-12)
                normalized_error = (weight_column - dequantized_column) / diagonal
                normalized_errors[:, local_index : local_index + 1] = normalized_error
                if local_index + 1 < block_width:
                    W_block[:, local_index + 1 :] -= normalized_error * H_block[
                        local_index, local_index + 1 :
                    ].unsqueeze(0)

            if block_end < in_feat:
                W[:, block_end:] -= normalized_errors.matmul(
                    H_inv[block_start:block_end, block_end:]
                )
            del W_block, H_block, normalized_errors

        results[bits] = {
            "w_q": Q,
            "scale": scales,
            "zero": zeros,
            "bits": bits,
            "method": f"gptq_w{bits}",
            "group_size": group_size,
            "in_features": in_feat,
            "out_features": out_feat,
        }
    return results


@torch.no_grad()
def gptq_quantize_linear(
    layer: nn.Linear,
    inp: torch.Tensor,
    bits: int = 4,
    group_size: int = 128,
) -> Dict:
    return gptq_quantize_linear_multi(layer, inp, (bits,), group_size)[bits]


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
    linear_layers, layer_inputs = collect_linear_inputs(
        model, calib_data, max_tokens=4096
    )
    quant_state = {}
    for layer_idx, (layer_name, layer_module) in enumerate(linear_layers):
        inp_gpu = layer_inputs.pop(layer_name).to(device=device, dtype=torch.float32)
        print(f"  GPTQ layer {layer_idx+1}/{len(linear_layers)}: {layer_name}", flush=True)
        qi = gptq_quantize_linear(
            layer_module, inp_gpu, bits=bits, group_size=group_size
        )
        quant_state[layer_name] = {
            key: value.cpu() if hasattr(value, "cpu") else value
            for key, value in qi.items()
        }
        del inp_gpu, qi
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

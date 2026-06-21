# exp01_baseline: 基础 PTQ 方法对比

## 实验目的

在 3 个小模型上对比 5 种 PTQ 方法对下游任务精度的影响，确定最优量化方法。

## 配置

| 项目 | 值 |
|------|-----|
| 模型 | Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM-1.7B |
| 方法 | fp16, RTN (4-bit), GPTQ (4-bit), AWQ (4-bit), SmoothQuant (W8A8) |
| 任务 | ARC Challenge, HellaSwag, MMLU, GSM8K (300 采样) |
| 校准数据 | WikiText-2 (128 samples, max_length=2048) |
| 硬件 | RTX 5060 Ti 8GB |

## 关键发现

- SmoothQuant 在所有模型上最接近 fp16 baseline
- RTN 在 GSM8K 上退化严重（0.5B: 16.7%, SmolLM: 18.3%）
- AWQ 在最小模型 (0.5B) 上表现最差
- SmolLM-1.7B 的 GSM8K GPTQ 因显存限制未跑（13GB state 文件无法加载）

## 复现命令

```bash
python experiments/exp01_baseline/run.py
```

## 已知问题

- WDDM 显存碎片化：需设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- SmolLM GPTQ state 文件 13GB，需 compact 化（w_q float16→uint8）

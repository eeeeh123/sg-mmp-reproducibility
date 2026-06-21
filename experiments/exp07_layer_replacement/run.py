"""实验4：Layer Replacement — 原地权重交换版（避免 WDDM 重复加载崩溃）。

单次加载模型，CPU 备份 FP16 + W4 权重，逐层原地替换后 eval。
"""
import os, sys
sys.path.insert(0, ".")
import torch, gc, time, json
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptq.eval import run_eval_on_model, cleanup_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
GPTQ_STATE = "results/Qwen2.5-0.5B_gptq_compact.pt"
OUT_DIR = "experiments/exp07_layer_replacement"
GSM8K_LIMIT = 100

os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. Load FP16 model, save all weights to CPU ----
print("Loading FP16 model (once)...")
cleanup_gpu()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
    device_map="cuda:0", low_cpu_mem_usage=True)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Save FP16 originals to CPU
fp16_weights = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        fp16_weights[name] = module.weight.data.clone().cpu()
fp16_lm_head = model.lm_head.weight.data.clone().cpu()
fp16_embed = model.model.embed_tokens.weight.data.clone().cpu()
print(f"  Saved {len(fp16_weights)} Linear + lm_head + embedding to CPU")

# ---- 2. Apply GPTQ-W4 ----
print("Applying GPTQ-W4...")
from ptq.quant.gptq import apply_gptq_to_model_gpu
qs = torch.load(GPTQ_STATE, map_location="cpu", weights_only=False)
apply_gptq_to_model_gpu(model, qs)
del qs; gc.collect(); torch.cuda.empty_cache()

# Save W4 (dequantized FP16) weights for restoration
w4_weights = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        w4_weights[name] = module.weight.data.clone().cpu()
w4_lm_head = model.lm_head.weight.data.clone().cpu()
w4_embed = model.model.embed_tokens.weight.data.clone().cpu()

# ---- Helper: replace one layer's weights ----
LAYER_MODULES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

def replace_layer_weights(target_layer, weight_dict):
    """Replace all 7 linear modules in target_layer with weights from weight_dict."""
    prefix = f"model.layers.{target_layer}."
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.startswith(prefix) and name in weight_dict:
            module.weight.data.copy_(weight_dict[name].to(module.weight.device, module.weight.dtype))

def run_gsm8k_and_record(label):
    sc = run_eval_on_model(model, tokenizer, ["gsm8k"], batch_size=4,
                            max_gen_toks=256, limit=GSM8K_LIMIT)
    return sc.get("gsm8k", 0)

# ---- 3. Baseline: all W4 ----
print("\n--- Baseline: GPTQ-W4 ---")
baseline = run_gsm8k_and_record("baseline")
print(f"  GPTQ-W4 GSM8K(limit={GSM8K_LIMIT}) = {baseline:.2f}")

results = [{"layer": "baseline_gptq_w4", "gsm8k": baseline, "delta": 0}]

# ---- 4. Per-layer replacement ----
layers = sorted(set(
    int(n.split(".")[2]) for n in fp16_weights if n.startswith("model.layers.")
))
print(f"Layers: {layers}")

for layer_idx in layers:
    print(f"\n--- Layer {layer_idx}/{max(layers)} ---")
    # Replace with FP16
    replace_layer_weights(layer_idx, fp16_weights)
    gsm = run_gsm8k_and_record(f"layer_{layer_idx}_fp16")
    delta = gsm - baseline
    print(f"  GSM8K={gsm:.2f} (delta={delta:+.2f})")
    results.append({"layer": layer_idx, "gsm8k": gsm, "delta": delta})
    # Restore W4
    replace_layer_weights(layer_idx, w4_weights)
    gc.collect(); torch.cuda.empty_cache()
    # Save incrementally
    with open(os.path.join(OUT_DIR, "layer_replacement.jsonl"), "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

# ---- 5. lm_head ----
print("\n--- lm_head ---")
model.lm_head.weight.data.copy_(fp16_lm_head.to(model.lm_head.weight.device, model.lm_head.weight.dtype))
gsm = run_gsm8k_and_record("lm_head_fp16")
results.append({"layer": "lm_head", "gsm8k": gsm, "delta": gsm - baseline})
print(f"  GSM8K={gsm:.2f} (delta={gsm - baseline:+.2f})")
model.lm_head.weight.data.copy_(w4_lm_head.to(model.lm_head.weight.device, model.lm_head.weight.dtype))

# ---- 6. embedding ----
print("\n--- embedding ---")
model.model.embed_tokens.weight.data.copy_(fp16_embed.to(model.model.embed_tokens.weight.device, model.model.embed_tokens.weight.dtype))
gsm = run_gsm8k_and_record("embed_fp16")
results.append({"layer": "embedding", "gsm8k": gsm, "delta": gsm - baseline})
print(f"  GSM8K={gsm:.2f} (delta={gsm - baseline:+.2f})")

# ---- Save final results ----
import csv
with open(os.path.join(OUT_DIR, "layer_replacement.jsonl"), "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")
with open(os.path.join(OUT_DIR, "layer_replacement.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["layer", "gsm8k", "delta"])
    w.writeheader()
    w.writerows(results)

# ---- Bar chart ----
layers_labels = [str(r["layer"]) for r in results[1:]]
deltas = [r["delta"] for r in results[1:]]
fig, ax = plt.subplots(figsize=(14, 6))
colors = ["green" if d > 0 else "red" for d in deltas]
ax.bar(layers_labels, deltas, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_xlabel("Layer replaced back to FP16")
ax.set_ylabel("Delta from GPTQ-W4 baseline")
ax.set_title(f"Layer Replacement: GSM8K Improvement (limit={GSM8K_LIMIT})\nBaseline GPTQ-W4 = {baseline:.2f}")
ax.tick_params(axis="x", rotation=45, labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "layer_replacement.png"), dpi=150)
plt.close(fig)

# ---- Summary ----
print(f"\n{'='*60}")
sorted_r = sorted(results[1:], key=lambda r: r["delta"], reverse=True)
print("Top 5 layers by delta:")
for r in sorted_r[:5]:
    print(f"  {r['layer']}: delta={r['delta']:+.2f}")

config_b_layers = {2, 6, 7, 11}
print(f"\nconfig_b sensitive layers [2,6,7,11] ranking:")
for r in sorted_r:
    if isinstance(r["layer"], int) and r["layer"] in config_b_layers:
        rank = sorted_r.index(r) + 1
        print(f"  Layer {r['layer']}: rank #{rank}, delta={r['delta']:+.2f}")

cleanup_gpu()
print(f"\nResults saved to {OUT_DIR}/")

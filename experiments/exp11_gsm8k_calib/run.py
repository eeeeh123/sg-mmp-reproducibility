"""GPTQ(GSM8K校准) + config_b 量化 + GSM8K 评测。
校准数据: GSM8K train question 文本，128 条。
"""
import os, sys
sys.path.insert(0, ".")
import torch, gc, json, re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptq.quant.mixed_precision import (
    quantize_model_mixed_precision, apply_mixed_precision_to_model_gpu,
    ATTN_PROJ, FFN_PROJ,
)
from ptq.eval import run_eval_on_model, cleanup_gpu, save_result

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
STATE_PATH = "results/Qwen2.5-0.5B_config_b_gsm8kcal.pt"
OUTPUT_FILE = "results/task_results_full.jsonl"
os.makedirs("results", exist_ok=True)

# ---- policy: config_b (layer 2/6/7/11 全 W8，其余模块分) ----
def _layer_num(name):
    m = re.search(r'layers\.(\d+)', name)
    return int(m.group(1)) if m else -1

def policy(layer_idx, layer_name, layer_short):
    ln = _layer_num(layer_name)
    if ln in {2, 6, 7, 11}:
        return "w8"
    if layer_short in FFN_PROJ:
        return "w4"
    elif layer_short in ATTN_PROJ:
        return "w8"
    return "skip"

# ---- 1. calibration data: GSM8K train questions (128 条) ----
print("Preparing GSM8K calibration data...")
model_tmp = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
    device_map="cuda:0", low_cpu_mem_usage=True)
model_tmp.eval()
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

ds = load_dataset("openai/gsm8k", "main", split="train")
questions = ds["question"][:128]

# Tokenize + pad to max_length
max_len = 2048
calib = torch.zeros(128, max_len, dtype=torch.long)
for i, q in enumerate(questions):
    ids = tok.encode(q, add_special_tokens=True, truncation=True, max_length=max_len)
    calib[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

print(f"  calib shape: {calib.shape}, samples: {len(questions)}")
print(f"  avg question length: {sum(len(tok.encode(q)) for q in questions)/128:.0f} tokens")

del model_tmp; gc.collect(); torch.cuda.empty_cache()

# ---- 2. quantize with config_b policy ----
print("\nQuantizing with GSM8K calibration...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
    device_map="cuda:0", low_cpu_mem_usage=True)
model.eval()

state = quantize_model_mixed_precision(model, calib, bits_w4=4, group_size=128, layer_policy=policy)

n_w8 = sum(1 for v in state.values() if v.get('method') == 'w8_perchannel')
n_w4 = sum(1 for v in state.values() if v.get('method') == 'gptq_w4')
print(f"  {n_w8} W8 + {n_w4} W4 = {len(state)} layers")

torch.save(state, STATE_PATH)
print(f"  Saved: {STATE_PATH}")
del state; gc.collect(); torch.cuda.empty_cache()

# ---- 3. apply + eval GSM8K ----
print("\n--- GSM8K ---")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

qs = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
apply_mixed_precision_to_model_gpu(model, qs)
del qs

scores = run_eval_on_model(model, tok, ["gsm8k"], batch_size=4, max_gen_toks=256, limit=300)
gsm = scores.get("gsm8k")
print(f"  GSM8K={gsm:.2f}" if gsm else "  FAILED")

if gsm is not None:
    save_result(OUTPUT_FILE, MODEL_NAME, "config_b_gsm8kcal", {"gsm8k": gsm})

cleanup_gpu()
print("\nDone!")

"""补完 3 个未完成的评估：onecomp 64x1024 x2 + hadamard_gptq 的 hellaswag 和 mmlu."""
import os, sys, time, gc
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, ".")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq.eval import run_eval_on_model, cleanup_gpu, save_result
from ptq.data import get_calib_dataset
from ptq.quant.hadamard_gptq import quantize_model_hadamard_gptq, apply_hadamard_gptq_to_model_gpu

MODEL_NAME = "Qwen2.5-0.5B"
OUTPUT_FILE = "results/task_results_full.jsonl"
MISSING_TASKS = ["hellaswag", "mmlu"]

# ============================================================
# 1. onecomp_qep 64x1024 variants (full model directories exist)
# ============================================================
for gamma_tag in ["g0.0_64x1024", "g1.0_64x1024"]:
    method = f"onecomp_qep_{gamma_tag}"
    save_dir = f"results/Qwen2.5-0.5B_onecomp_qep_{gamma_tag}"

    if not os.path.isdir(save_dir):
        print(f"SKIP {method}: directory not found at {save_dir}")
        continue

    cleanup_gpu()
    print(f"\n=== {method}: loading from {save_dir} ===")
    model = AutoModelForCausalLM.from_pretrained(
        save_dir, torch_dtype=torch.float16, trust_remote_code=True, device_map="cuda:0")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)

    for task in MISSING_TASKS:
        print(f"  {task}...")
        t0 = time.time()
        try:
            scores = run_eval_on_model(model, tokenizer, [task])
            save_result(OUTPUT_FILE, MODEL_NAME, method, scores)
            elapsed = time.time() - t0
            sc = f"{scores[task]:.2f}%" if scores.get(task) is not None else "FAILED"
            print(f"    {task}: {sc} ({elapsed:.0f}s)")
        except Exception as e:
            print(f"    {task}: ERROR {e}")

    del model, tokenizer
    cleanup_gpu()

# ============================================================
# 2. hadamard_gptq (re-quantize then eval)
# ============================================================
cleanup_gpu()
print("\n=== hadamard_gptq: re-quantizing ===")
MODEL_PATH = "models/Qwen2.5-0.5B"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True, device_map="cuda:0")
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("  Preparing calibration data (64 samples, 1024 max_len)...")
calib_data = get_calib_dataset(tokenizer, n_samples=64, max_length=1024)

print("  Quantizing Hadamard + GPTQ-W4...")
t0 = time.time()
quant_state = quantize_model_hadamard_gptq(model, calib_data, bits=4, group_size=128)
print(f"  Quantize done in {time.time() - t0:.0f}s ({len(quant_state)} layers)")

print("  Applying quantized weights...")
apply_hadamard_gptq_to_model_gpu(model, quant_state)
del quant_state, calib_data
gc.collect()
torch.cuda.empty_cache()

for task in MISSING_TASKS:
    print(f"  {task}...")
    t0 = time.time()
    try:
        scores = run_eval_on_model(model, tokenizer, [task])
        save_result(OUTPUT_FILE, MODEL_NAME, "hadamard_gptq", scores)
        elapsed = time.time() - t0
        sc = f"{scores[task]:.2f}%" if scores.get(task) is not None else "FAILED"
        print(f"    {task}: {sc} ({elapsed:.0f}s)")
    except Exception as e:
        print(f"    {task}: ERROR {e}")

del model, tokenizer
cleanup_gpu()
print("\n=== Done ===")

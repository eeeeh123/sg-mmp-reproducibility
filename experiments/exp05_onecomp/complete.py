"""补跑 onecomp 模型的 hellaswag + mmlu."""
import os, sys, json, time, gc
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, ".")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq.eval import run_eval_on_model, cleanup_gpu, save_result

MODEL_NAME = "Qwen2.5-0.5B"
TASKS = ["hellaswag", "mmlu"]
OUTPUT_FILE = "results/task_results_full.jsonl"

for gamma in ["g0.0", "g1.0"]:
    method = f"onecomp_qep_{gamma}"
    save_dir = f"results/Qwen2.5-0.5B_onecomp_qep_{gamma}"

    cleanup_gpu()
    print(f"\n=== {method}: loading from {save_dir} ===")
    model = AutoModelForCausalLM.from_pretrained(
        save_dir, torch_dtype=torch.float16, trust_remote_code=True, device_map="cuda:0")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)

    for task in TASKS:
        print(f"  {task}...")
        t0 = time.time()
        scores = run_eval_on_model(model, tokenizer, [task])
        save_result(OUTPUT_FILE, MODEL_NAME, method, scores)
        elapsed = time.time() - t0
        sc = f"{scores[task]:.2f}%" if scores.get(task) is not None else "FAILED"
        print(f"  {task}: {sc} ({elapsed:.0f}s)")

    del model, tokenizer
    cleanup_gpu()

print("\nDone.")

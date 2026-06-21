"""修复校准公平性补实验：GPTQ/AWQ × WikiText128 / GSM8K128。

每个 method/calib 组合独立加载模型，完成 GSM8K(limit=300) + ARC-Challenge 评测后释放显存。
保存为 gptq_wiki128 / gptq_gsm128 / awq_wiki128 / awq_gsm128，不覆盖主表结果。
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, ".")

import torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq.data import get_calib_dataset
from ptq.eval import run_eval_on_model, cleanup_gpu, save_result
from ptq.quant.gptq import quantize_model_gptq, apply_gptq_to_model_gpu
from ptq.quant.awq import quantize_model_awq, apply_awq_to_model_gpu

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
RESULTS_FILE = "results/task_results_full.jsonl"
N_SAMPLES = 128
MAX_LENGTH = 2048
EVAL_TASKS = ["gsm8k", "arc_challenge"]
GSM8K_LIMIT = 300

COMBOS = [
    ("gptq", "wikitext",  "gptq_wiki128"),
    ("gptq", "gsm8k",    "gptq_gsm128"),
    ("awq",  "wikitext",  "awq_wiki128"),
    ("awq",  "gsm8k",     "awq_gsm128"),
]


def run_one(method, calib_name, save_name):
    print(f"\n{'='*60}")
    print(f"[{save_name}] {method.upper()} + {calib_name} ({N_SAMPLES} samples)")
    print(f"{'='*60}", flush=True)

    cleanup_gpu()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    calib = get_calib_dataset(tokenizer, n_samples=N_SAMPLES, max_length=MAX_LENGTH,
                               dataset_name=calib_name)
    print(f"  Calib shape: {calib.shape}", flush=True)

    if method == "gptq":
        print("  Quantizing GPTQ-W4...", flush=True)
        quant_state = quantize_model_gptq(model, calib, bits=4, group_size=128)
        apply_gptq_to_model_gpu(model, quant_state)
    else:
        print("  Quantizing AWQ...", flush=True)
        quant_state, awq_scales = quantize_model_awq(model, calib, bits=4, group_size=128)
        apply_awq_to_model_gpu(model, quant_state, awq_scales)
        del awq_scales

    del quant_state, calib
    gc.collect()
    torch.cuda.empty_cache()

    for task in EVAL_TASKS:
        print(f"  Eval: {task}...", flush=True)
        limit = GSM8K_LIMIT if task == "gsm8k" else None
        try:
            sc = run_eval_on_model(model, tokenizer, [task], batch_size=4,
                                    max_gen_toks=256, limit=limit)
            val = sc.get(task)
            if val is not None:
                save_result(RESULTS_FILE, MODEL_NAME, save_name, {task: val})
                print(f"    {task}: {val:.2f}", flush=True)
            else:
                print(f"    {task}: score extraction failed", flush=True)
        except Exception as e:
            print(f"    {task}: ERROR {e}", flush=True)

    del model, tokenizer
    cleanup_gpu()
    print(f"[{save_name}] done.", flush=True)


if __name__ == "__main__":
    for method, calib_name, save_name in COMBOS:
        run_one(method, calib_name, save_name)

    print("\nAll calibration fairness experiments done.")

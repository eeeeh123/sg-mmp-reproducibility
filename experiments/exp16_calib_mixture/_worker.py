"""实验7 Worker: 单个 (method, ratio) 组合的量化和评估。

由 run.py 通过子进程调用，每个组合独立进程运行以避免 WDDM 碎片化。
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, ".")

import torch, gc, json, argparse
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq.data import get_calib_dataset
from ptq.eval import run_eval_on_model, cleanup_gpu, save_result
from ptq.quant.gptq import quantize_model_gptq, apply_gptq_to_model_gpu
from ptq.quant.mixed_precision import quantize_model_mixed_precision, apply_mixed_precision_to_model_gpu, parse_layer_num

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
RESULTS_DIR = "results"
OUTPUT_FILE = "results/task_results_full.jsonl"
N_SAMPLES = 128
MAX_LENGTH = 2048
ALL_TASKS = ["arc_challenge", "hellaswag", "mmlu", "gsm8k"]
CONFIG_B_SENSITIVE_LAYERS = {2, 6, 7, 11}


def config_b_policy(layer_idx, layer_name, layer_short):
    ln = parse_layer_num(layer_name)
    if ln in CONFIG_B_SENSITIVE_LAYERS:
        return "w8"
    elif layer_short in {"q_proj", "k_proj", "v_proj"}:
        return "w8"
    elif layer_short in {"o_proj", "gate_proj", "up_proj", "down_proj"}:
        return "w4"
    else:
        return "skip"


def build_mixed_calib(tokenizer, n_wiki, n_gsm8k):
    if n_wiki > 0:
        wiki_data = get_calib_dataset(tokenizer, n_samples=n_wiki, max_length=MAX_LENGTH,
                                       dataset_name="wikitext")
    else:
        wiki_data = torch.empty((0, MAX_LENGTH), dtype=torch.long)
    if n_gsm8k > 0:
        gsm_data = get_calib_dataset(tokenizer, n_samples=n_gsm8k, max_length=MAX_LENGTH,
                                      dataset_name="gsm8k")
    else:
        gsm_data = torch.empty((0, MAX_LENGTH), dtype=torch.long)
    combined = torch.cat([wiki_data, gsm_data], dim=0)
    idx = torch.randperm(combined.shape[0])
    return combined[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--ratio_tag", type=str, required=True)
    parser.add_argument("--n_wiki", type=int, required=True)
    parser.add_argument("--n_gsm8k", type=int, required=True)
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--skip_quantize", action="store_true")
    args = parser.parse_args()

    is_phase2 = (args.phase == 2)
    limit = 300
    task_list = ALL_TASKS if is_phase2 else ["gsm8k"]

    cleanup_gpu()
    print(f"Worker: {args.method} @ {args.ratio_tag} (wiki={args.n_wiki}, gsm8k={args.n_gsm8k})", flush=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantize
    if not args.skip_quantize:
        print("  Building calibration data...", flush=True)
        calib = build_mixed_calib(tokenizer, args.n_wiki, args.n_gsm8k)

        if "config_b" in args.method:
            state_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_config_b_mix{args.ratio_tag}.pt")
            if not os.path.exists(state_path):
                print(f"  Quantizing config_b...", flush=True)
                qs = quantize_model_mixed_precision(model, calib, layer_policy=config_b_policy)
                torch.save(qs, state_path)
            else:
                print(f"  Using cached state", flush=True)
                qs = torch.load(state_path, map_location="cpu", weights_only=False)
            apply_mixed_precision_to_model_gpu(model, qs)
        else:
            state_path = os.path.join(RESULTS_DIR, f"{MODEL_NAME}_gptq_mix{args.ratio_tag}.pt")
            if not os.path.exists(state_path):
                print(f"  Quantizing GPTQ-W4...", flush=True)
                qs = quantize_model_gptq(model, calib, bits=4, group_size=128)
                torch.save(qs, state_path)
            else:
                print(f"  Using cached state", flush=True)
                qs = torch.load(state_path, map_location="cpu", weights_only=False)
            apply_gptq_to_model_gpu(model, qs)

        del qs, calib
        gc.collect()
        torch.cuda.empty_cache()

    # Evaluate
    scores = {}
    for task in task_list:
        print(f"  Eval: {task}...", flush=True)
        try:
            sc = run_eval_on_model(model, tokenizer, [task], batch_size=4,
                                    max_gen_toks=256, limit=limit)
            scores.update(sc)
            val = sc.get(task)
            print(f"    {task}: {val:.2f}" if val is not None else f"    {task}: FAILED", flush=True)
        except Exception as e:
            print(f"    {task}: ERROR {e}", flush=True)
            scores[task] = None

    full_method = f"{args.method}_mix{args.ratio_tag}_fixed"
    save_result(OUTPUT_FILE, MODEL_NAME, full_method, scores)

    del model, tokenizer
    cleanup_gpu()
    print(f"Worker done: {args.method} @ {args.ratio_tag}", flush=True)


if __name__ == "__main__":
    main()

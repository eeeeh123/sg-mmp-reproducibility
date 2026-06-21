"""实验2：config_b 推广到 Qwen2.5-1.5B。

分两步（独立 Python 进程调用，避免 WDDM 碎片化）：
  step_quantize: 量化并保存 state
  step_eval:    加载 state → GSM8K quick screen

策略 2A: 敏感层 [2, 7, 8, 13] 全 W8, 其余 q/k/v W8 + o/gate/up/down W4
策略 2B: 敏感层 [0-9]+[18-27] 全 W8, 其余 q/k/v W8 + o/gate/up/down W4
"""
import os, sys
sys.path.insert(0, ".")

STEP = sys.argv[1] if len(sys.argv) > 1 else "all"

import torch, gc, time
from ptq.data import get_calib_dataset
from ptq.quant.mixed_precision import (
    quantize_model_mixed_precision,
    apply_mixed_precision_to_model_gpu,
    ATTN_PROJ, FFN_PROJ, parse_layer_num,
)
from ptq.eval import run_eval_on_model, cleanup_gpu, save_result
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen2.5-1.5B"
MODEL_PATH = "models/Qwen2.5-1.5B"
RESULTS_FILE = "results/task_results_full.jsonl"
GSM8K_LIMIT = 300

# ---- strategies ----
def strategy_2a(layer_idx, layer_name, layer_short):
    """敏感层 [2, 7, 8, 13] 全 W8，其余 q/k/v W8 + 其余 W4"""
    ln = parse_layer_num(layer_name)  # actual transformer layer (0-27)
    if ln in {2, 7, 8, 13}:
        return "w8"
    if layer_short in {"q_proj", "k_proj", "v_proj"}:
        return "w8"
    if layer_short in FFN_PROJ or layer_short == "o_proj":
        return "w4"
    return "skip"

def strategy_2b(layer_idx, layer_name, layer_short):
    """敏感层 [0-9]+[18-27] 全 W8，其余 q/k/v W8 + 其余 W4"""
    ln = parse_layer_num(layer_name)  # actual transformer layer (0-27)
    if ln <= 9 or ln >= 18:
        return "w8"
    if layer_short in {"q_proj", "k_proj", "v_proj"}:
        return "w8"
    if layer_short in FFN_PROJ or layer_short == "o_proj":
        return "w4"
    return "skip"

STRATEGIES = [
    ("config_b_2a", strategy_2a, "results/Qwen2.5-1.5B_config_b_2a.pt"),
    ("config_b_2b", strategy_2b, "results/Qwen2.5-1.5B_config_b_2b.pt"),
]

# ---- step_quantize ----
def step_quantize(name, policy, state_path, safe_mode=False):
    if os.path.exists(state_path):
        print(f"[{name}] State exists ({os.path.getsize(state_path)/1024**3:.1f} GB), skip")
        return True

    n_calib = 128
    print(f"\n[{name}] Quantizing (28 layers, ~196 linears, {n_calib} calib samples)...")
    cleanup_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    calib = get_calib_dataset(tokenizer, n_samples=n_calib, max_length=2048)

    try:
        state = quantize_model_mixed_precision(model, calib, bits_w4=4, group_size=128,
                                               layer_policy=policy)
        # CRITICAL: sync CUDA before CPU-heavy torch.save to catch delayed errors
        torch.cuda.synchronize()
        torch.save(state, state_path)
        sz_gb = os.path.getsize(state_path) / 1024**3
        print(f"  Saved: {state_path} ({sz_gb:.1f} GB, {len(state)} layers)")
        del state
    finally:
        del model, tokenizer, calib
        gc.collect()
        torch.cuda.empty_cache()
    cleanup_gpu()
    return True

# ---- step_eval ----
def step_eval_gsm8k(name, state_path):
    print(f"\n[{name}] Loading model + state for GSM8K eval...")
    cleanup_gpu()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    quant_state = torch.load(state_path, map_location="cpu", weights_only=False)
    apply_mixed_precision_to_model_gpu(model, quant_state)
    del quant_state
    gc.collect()
    torch.cuda.empty_cache()

    scores = run_eval_on_model(model, tokenizer, ["gsm8k"], batch_size=4,
                               max_gen_toks=256, limit=GSM8K_LIMIT)
    gsm8k = scores.get("gsm8k")
    print(f"  [{name}] GSM8K(limit={GSM8K_LIMIT}) = {gsm8k:.2f}" if gsm8k else f"  [{name}] FAILED")

    del model, tokenizer
    cleanup_gpu()
    return gsm8k

# ---- full benchmark on winner ----
def step_full(name, state_path):
    print(f"\n[{name}] Full benchmark...")
    cleanup_gpu()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    quant_state = torch.load(state_path, map_location="cpu", weights_only=False)
    apply_mixed_precision_to_model_gpu(model, quant_state)
    del quant_state
    gc.collect()
    torch.cuda.empty_cache()

    TASKS = ["arc_challenge", "hellaswag", "mmlu", "gsm8k"]
    results = {}
    for task in TASKS:
        print(f"  [{name}] {task}...")
        limit = None if task != "gsm8k" else 300
        try:
            sc = run_eval_on_model(model, tokenizer, [task], batch_size=4,
                                    max_gen_toks=256, limit=limit)
            s = sc.get(task)
            if s is not None:
                save_result(RESULTS_FILE, MODEL_NAME, name, {task: s})
                results[task] = s
                print(f"    {task}: {s:.2f}")
            else:
                print(f"    {task}: score extraction failed")
        except Exception as e:
            print(f"    {task} ERROR: {e}")
            gc.collect()
            torch.cuda.empty_cache()

    del model, tokenizer
    cleanup_gpu()
    return results


# ================================================================
if __name__ == "__main__":
    if STEP == "quantize_2a":
        step_quantize("config_b_2a", strategy_2a, STRATEGIES[0][2])
    elif STEP == "quantize_2b":
        step_quantize("config_b_2b", strategy_2b, STRATEGIES[1][2])
    elif STEP == "quantize_2a_safe":
        step_quantize("config_b_2a", strategy_2a, STRATEGIES[0][2], safe_mode=True)
    elif STEP == "quantize":
        for name, policy, sp in STRATEGIES:
            step_quantize(name, policy, sp)

    elif STEP == "eval":
        gsm8k_scores = {}
        for name, _, sp in STRATEGIES:
            if os.path.exists(sp):
                gsm = step_eval_gsm8k(name, sp)
                gsm8k_scores[name] = gsm
            else:
                print(f"[{name}] State missing: {sp}")

        print(f"\n{'='*60}")
        print("GSM8K quick screen results:")
        for n, s in gsm8k_scores.items():
            print(f"  {n}: {s:.2f}" if s else f"  {n}: FAILED")

        valid = {k: v for k, v in gsm8k_scores.items() if v is not None}
        if valid:
            winner = max(valid, key=valid.get)
            print(f"\nWinner: {winner} (GSM8K={valid[winner]:.2f})")
            winner_sp = next(sp for n, _, sp in STRATEGIES if n == winner)
            step_full(winner, winner_sp)

            if len(valid) > 1:
                ru = sorted(valid, key=valid.get)[-2]
                if valid[winner] - valid[ru] < 3:
                    print(f"Runner-up {ru} close ({valid[ru]:.2f}), also full benchmark")
                    ru_sp = next(sp for n, _, sp in STRATEGIES if n == ru)
                    step_full(ru, ru_sp)

    else:  # "all" - legacy, caller should use quantize then eval separately
        print("Use: python run.py quantize  OR  python run.py eval")
        print("Running quantize + eval in single process (may crash on WDDM)...")
        for name, policy, sp in STRATEGIES:
            step_quantize(name, policy, sp)
            step_eval_gsm8k(name, sp)

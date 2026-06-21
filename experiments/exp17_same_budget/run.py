"""实验1：Same Bit-Budget Control Groups.

证明 config_b 的层选择 {2,6,7,11} 优于随机/首N/尾N选择，
在相同 bit 预算下（4层全W8 + 其余20层 Q/K/V W8 + FFN W4）。

Usage:
  python run.py quantize    # 量化5组对照组（config_b 复用已有）
  python run.py eval        # GSM8K limit=100 评测全部6组
"""
import os, sys, random
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

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
RESULTS_FILE = "results/task_results_full.jsonl"
GSM8K_LIMIT = 300
NUM_LAYERS = 24
NUM_PROTECTED = 4

# ---- 生成随机层选择 ----
random.seed(42)
RANDOM_42 = sorted(random.sample(range(NUM_LAYERS), NUM_PROTECTED))
random.seed(123)
RANDOM_123 = sorted(random.sample(range(NUM_LAYERS), NUM_PROTECTED))
random.seed(456)
RANDOM_456 = sorted(random.sample(range(NUM_LAYERS), NUM_PROTECTED))

CONTROL_GROUPS = [
    ("random_42",  set(RANDOM_42)),
    ("random_123", set(RANDOM_123)),
    ("random_456", set(RANDOM_456)),
    ("first_4",    {0, 1, 2, 3}),
    ("last_4",     {20, 21, 22, 23}),
]

# config_b 也加入 eval（复用已有 .pt），也需要单独量化以获取 limit=100 的结果
GROUPS_WITH_CONFIG_B = CONTROL_GROUPS + [
    ("config_b", {2, 6, 7, 11}),
]


def make_policy(protected: set):
    """返回 layer_policy callback：protected 中的层全 W8，其余 Q/K/V W8 + FFN W4。"""
    def policy(layer_idx, layer_name, layer_short):
        ln = parse_layer_num(layer_name)
        if ln in protected:
            return "w8"
        if layer_short in ATTN_PROJ:  # q_proj, k_proj, v_proj
            return "w8"
        if layer_short in FFN_PROJ or layer_short == "o_proj":
            return "w4"
        return "skip"
    return policy


def state_path(name):
    return f"results/{MODEL_NAME}_{name}.pt"


# ---- step_quantize ----
def step_quantize(name, protected_set):
    sp = state_path(name)
    if os.path.exists(sp):
        print(f"[{name}] State exists ({os.path.getsize(sp)/1024**3:.1f} GB), skip")
        return True

    print(f"\n[{name}] Quantizing (protected={sorted(protected_set)})...")
    cleanup_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    calib = get_calib_dataset(tokenizer, n_samples=128, max_length=2048)

    try:
        state = quantize_model_mixed_precision(model, calib, bits_w4=4, group_size=128,
                                               layer_policy=make_policy(protected_set))
        torch.cuda.synchronize()
        torch.save(state, sp)
        sz_gb = os.path.getsize(sp) / 1024**3
        print(f"  Saved: {sp} ({sz_gb:.1f} GB, {len(state)} layers)")
        del state
    finally:
        del model, tokenizer, calib
        gc.collect()
        torch.cuda.empty_cache()
    cleanup_gpu()
    return True


# ---- step_eval ----
def step_eval(name):
    sp = state_path(name)
    if not os.path.exists(sp):
        print(f"[{name}] State NOT FOUND: {sp}")
        return None

    print(f"\n[{name}] Loading for GSM8K eval (limit={GSM8K_LIMIT})...")
    cleanup_gpu()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    qs = torch.load(sp, map_location="cpu", weights_only=False)
    apply_mixed_precision_to_model_gpu(model, qs)
    del qs
    gc.collect()
    torch.cuda.empty_cache()

    scores = run_eval_on_model(model, tokenizer, ["gsm8k"], batch_size=4,
                               max_gen_toks=256, limit=GSM8K_LIMIT)
    gsm8k = scores.get("gsm8k")
    if gsm8k is not None:
        save_name = "config_b_samebudget_300" if name == "config_b" else f"{name}_samebudget_300"
        save_result(RESULTS_FILE, MODEL_NAME, save_name, {"gsm8k": gsm8k})
        print(f"  [{name}] GSM8K(limit={GSM8K_LIMIT}) = {gsm8k:.2f}")
    else:
        print(f"  [{name}] FAILED (score extraction)")

    del model, tokenizer
    cleanup_gpu()
    return gsm8k


# ================================================================
if __name__ == "__main__":
    print("Selected random layers:")
    print(f"  random_42:  {RANDOM_42}")
    print(f"  random_123: {RANDOM_123}")
    print(f"  random_456: {RANDOM_456}")
    print(f"  first_4:    [0, 1, 2, 3]")
    print(f"  last_4:     [20, 21, 22, 23]")
    print(f"  config_b:   [2, 6, 7, 11]")

    if STEP == "quantize":
        for name, prot in CONTROL_GROUPS:  # config_b 已有 .pt，跳过
            step_quantize(name, prot)

    elif STEP == "eval":
        results = {}
        for name, _ in GROUPS_WITH_CONFIG_B:
            gsm = step_eval(name)
            results[name] = gsm

        # 打印对比表
        print(f"\n{'='*65}")
        print(f"Same Bit-Budget Comparison (GSM8K limit={GSM8K_LIMIT})")
        print(f"{'Group':<15} {'GSM8K':<10} {'vs config_b':<15} {'Protected':<20}")
        print("-" * 65)
        cb_score = results.get("config_b")
        for name, prot in GROUPS_WITH_CONFIG_B:
            s = results.get(name)
            if s is not None and cb_score is not None:
                delta = s - cb_score
                print(f"{name:<15} {s:<10.2f} {delta:+.2f}{'':>10} {sorted(prot)}")
            elif s is not None:
                print(f"{name:<15} {s:<10.2f} {'N/A':<15}")
            else:
                print(f"{name:<15} {'FAILED':<10} {'N/A':<15}")

        # 随机组统计
        random_scores = [results.get(f"random_{s}") for s in ["42", "123", "456"]]
        random_scores = [s for s in random_scores if s is not None]
        if random_scores and cb_score is not None:
            mean_r = sum(random_scores) / len(random_scores)
            print(f"\n  random mean ± std: {mean_r:.2f} ± "
                  f"{(sum((s-mean_r)**2 for s in random_scores)/len(random_scores))**0.5:.2f}")
            print(f"  config_b: {cb_score:.2f}")
            print(f"  config_b vs random mean: {cb_score - mean_r:+.2f}")

    else:
        print("Usage:")
        print("  python run.py quantize    # 量化 5 组对照")
        print("  python run.py eval        # 评测全部 6 组（含 config_b）")
        print(f"\nSelected random layers:")
        print(f"  random_42:  {RANDOM_42}")
        print(f"  random_123: {RANDOM_123}")
        print(f"  random_456: {RANDOM_456}")

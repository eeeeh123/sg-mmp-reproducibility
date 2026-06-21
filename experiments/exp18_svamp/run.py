"""实验2：SVAMP Cross-Benchmark Generalization.

证明 config_b 在 SVAMP（另一个数学推理 benchmark）上同样有效。

Usage:
  python run.py eval_05B      # Qwen2.5-0.5B: fp16, gptq, config_b, config_b+LoRA_6c
  python run.py eval_15B      # Qwen2.5-1.5B: fp16, gptq, config_b_2b
  python run.py eval_smollm   # SmolLM-1.7B:   fp16, gptq, config_b
  python run.py eval          # 自动先跑 0.5B，若 config_b vs gptq 提升 >3 分则继续 1.5B + SmolLM
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") in (None, "", "expandable_segments:True"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
# 首次运行需要下载 SVAMP，之后可以设 OFFLINE=1
sys.path.insert(0, ".")

STEP = sys.argv[1] if len(sys.argv) > 1 else "eval"

import torch, gc, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from ptq.eval import cleanup_gpu, save_result
from ptq.quant.gptq import apply_gptq_to_model_gpu
from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager

SVAMP_YAML_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = "results/task_results_full.jsonl"
BATCH_SIZE = 4
MAX_GEN_TOKS = 256

# ---- Model configs ----
MODEL_CONFIGS = {
    "Qwen2.5-0.5B": {
        "path": "models/Qwen2.5-0.5B",
        "gptq_state": "results/Qwen2.5-0.5B_gptq_compact.pt",
        "config_b_state": "results/Qwen2.5-0.5B_config_b.pt",
        "lora_6c": "results/Qwen2.5-0.5B_config_b_failure_lora_6c",
    },
    "Qwen2.5-1.5B": {
        "path": "models/Qwen2.5-1.5B",
        "gptq_state": "results/Qwen2.5-1.5B_gptq_compact.pt",
        "config_b_state": "results/Qwen2.5-1.5B_config_b_2b.pt",
    },
    "SmolLM-1.7B": {
        "path": "models/SmolLM-1.7B",
        "gptq_state": "results/SmolLM-1.7B_gptq_compact.pt",
        "config_b_state": "results/SmolLM-1.7B_config_b_compact.pt",
    },
}


def load_fp16_model(model_path):
    """加载 FP16 模型。"""
    load_kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True)
    if "SmolLM" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.to("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda:0", **load_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def load_state_safe(path):
    assert_state_safe(path)
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def assert_state_safe(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if os.path.getsize(path) > 6 * 1024**3 and "_compact" not in os.path.basename(path):
        raise RuntimeError(
            f"Refusing to load oversized non-compact state ({os.path.getsize(path) / 1024**3:.2f} GB): {path}. "
            "Regenerate a *_compact.pt state first."
        )


def eval_svamp(model, tokenizer, model_name, method_name):
    """在已加载的模型上跑 SVAMP 评测。"""
    lm_eval_model = HFLM(
        pretrained=model, tokenizer=tokenizer,
        batch_size=BATCH_SIZE, max_batch_size=BATCH_SIZE,
    )
    try:
        task_manager = TaskManager(include_path=SVAMP_YAML_DIR)
        results = simple_evaluate(
            model=lm_eval_model,
            tasks=["svamp"],
            task_manager=task_manager,
            batch_size=BATCH_SIZE,
            log_samples=False,
            gen_kwargs={"temperature": 0.0, "max_new_tokens": MAX_GEN_TOKS, "do_sample": False},
        )
    except Exception as e:
        print(f"  SVAMP eval ERROR: {e}")
        del lm_eval_model
        return None

    r = results.get("results", {}).get("svamp", {})
    score = None
    for metric in ["exact_match,flexible-extract", "exact_match,none",
                   "acc,none", "exact_match,strict-match"]:
        if metric in r:
            score = r[metric]
            break
    score_pct = round(score * 100, 2) if score is not None else None
    print(f"  [{model_name}] [{method_name}] SVAMP = {score_pct:.2f}" if score_pct else f"  [{model_name}] [{method_name}] SVAMP = FAILED")

    if score_pct is not None:
        save_result(RESULTS_FILE, model_name, method_name, {"svamp": score_pct})

    del lm_eval_model
    gc.collect()
    torch.cuda.empty_cache()
    return score_pct


def eval_05B():
    """Qwen2.5-0.5B: fp16, gptq, config_b, config_b+LoRA_6c"""
    cfg = MODEL_CONFIGS["Qwen2.5-0.5B"]
    model_name = "Qwen2.5-0.5B"
    scores = {}

    # fp16
    print("\n--- FP16 ---")
    cleanup_gpu()
    model, tok = load_fp16_model(cfg["path"])
    scores["fp16"] = eval_svamp(model, tok, model_name, "fp16")
    del model, tok; cleanup_gpu()

    # gptq
    print("\n--- GPTQ-W4 ---")
    cleanup_gpu()
    model, tok = load_fp16_model(cfg["path"])
    qs = load_state_safe(cfg["gptq_state"])
    apply_gptq_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()
    scores["gptq"] = eval_svamp(model, tok, model_name, "gptq")
    del model, tok; cleanup_gpu()

    # config_b
    print("\n--- config_b ---")
    cleanup_gpu()
    model, tok = load_fp16_model(cfg["path"])
    qs = load_state_safe(cfg["config_b_state"])
    apply_mixed_precision_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()
    scores["config_b"] = eval_svamp(model, tok, model_name, "config_b")
    del model, tok; cleanup_gpu()

    # config_b + LoRA 6c
    print("\n--- config_b + LoRA 6c ---")
    cleanup_gpu()
    model, tok = load_fp16_model(cfg["path"])
    qs = load_state_safe(cfg["config_b_state"])
    apply_mixed_precision_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()
    # 加载 LoRA adapter
    model = PeftModel.from_pretrained(model, cfg["lora_6c"])
    model = model.merge_and_unload()
    scores["config_b_failure_lora_6c"] = eval_svamp(model, tok, model_name, "config_b_failure_lora_6c")
    del model, tok; cleanup_gpu()

    print(f"\n{'='*55}")
    print(f"Qwen2.5-0.5B SVAMP Results:")
    for m, s in scores.items():
        print(f"  {m:<30} {s:.2f}" if s else f"  {m:<30} FAILED")

    gptq_s = scores.get("gptq")
    cb_s = scores.get("config_b")
    if gptq_s and cb_s:
        improvement = cb_s - gptq_s
        print(f"\n  config_b vs gptq: {improvement:+.2f} points")
        if improvement > 3:
            print("  >> Improvement > 3pt. Continue to 1.5B and SmolLM.")
            return True
        else:
            print("  >> Improvement <= 3pt. Stop here.")
            return False
    return False


def eval_15B():
    """Qwen2.5-1.5B: fp16, gptq, config_b_2b"""
    cfg = MODEL_CONFIGS["Qwen2.5-1.5B"]
    model_name = "Qwen2.5-1.5B"

    # fp16
    print("\n--- 1.5B FP16 ---")
    cleanup_gpu()
    model, tok = load_fp16_model(cfg["path"])
    eval_svamp(model, tok, model_name, "fp16")
    del model, tok; cleanup_gpu()

    # gptq
    print("\n--- 1.5B GPTQ-W4 ---")
    cleanup_gpu()
    model, tok = load_fp16_model(cfg["path"])
    qs = load_state_safe(cfg["gptq_state"])
    apply_gptq_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()
    eval_svamp(model, tok, model_name, "gptq")
    del model, tok; cleanup_gpu()

    # config_b_2b
    print("\n--- 1.5B config_b_2b ---")
    cleanup_gpu()
    model, tok = load_fp16_model(cfg["path"])
    qs = load_state_safe(cfg["config_b_state"])
    apply_mixed_precision_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()
    eval_svamp(model, tok, model_name, "config_b_2b")
    del model, tok; cleanup_gpu()


def eval_smollm():
    """SmolLM-1.7B: fp16, gptq, config_b"""
    cfg = MODEL_CONFIGS["SmolLM-1.7B"]
    model_name = "SmolLM-1.7B"
    assert_state_safe(cfg["gptq_state"])
    assert_state_safe(cfg["config_b_state"])

    # fp16
    print("\n--- SmolLM FP16 ---")
    cleanup_gpu()
    model, tok = load_fp16_model(cfg["path"])
    eval_svamp(model, tok, model_name, "fp16")
    del model, tok; cleanup_gpu()

    # gptq
    print("\n--- SmolLM GPTQ-W4 ---")
    cleanup_gpu()
    model, tok = load_fp16_model(cfg["path"])
    qs = load_state_safe(cfg["gptq_state"])
    apply_gptq_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()
    eval_svamp(model, tok, model_name, "gptq")
    del model, tok; cleanup_gpu()

    # config_b
    print("\n--- SmolLM config_b ---")
    cleanup_gpu()
    model, tok = load_fp16_model(cfg["path"])
    qs = load_state_safe(cfg["config_b_state"])
    apply_mixed_precision_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()
    eval_svamp(model, tok, model_name, "config_b")
    del model, tok; cleanup_gpu()


if __name__ == "__main__":
    if STEP == "eval_05B":
        eval_05B()
    elif STEP == "eval_15B":
        eval_15B()
    elif STEP == "eval_smollm":
        eval_smollm()
    elif STEP == "eval":
        print("Phase 1: Qwen2.5-0.5B")
        should_continue = eval_05B()
        if should_continue:
            print("\n\nPhase 2: Qwen2.5-1.5B")
            eval_15B()
            print("\n\nPhase 3: SmolLM-1.7B")
            eval_smollm()
        print("\nAll SVAMP evals done.")
    else:
        print("Usage:")
        print("  python run.py eval_05B     # 0.5B × 4 configs")
        print("  python run.py eval_15B     # 1.5B × 3 configs")
        print("  python run.py eval_smollm  # SmolLM × 3 configs")
        print("  python run.py eval         # Auto (Phase 1 → 2 → 3)")

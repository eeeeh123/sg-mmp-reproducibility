"""实验3：config_b 推广到 SmolLM-1.7B — 分进程版。

用法: python run.py quantize  |  python run.py eval
"""
import os, sys
sys.path.insert(0, ".")
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") in (None, "", "expandable_segments:True"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
import torch, gc, time

from ptq.data import get_calib_dataset
from ptq.quant.mixed_precision import (
    quantize_model_mixed_precision,
    apply_mixed_precision_to_model_gpu,
    ATTN_PROJ, FFN_PROJ, parse_layer_num,
)
from ptq.eval import run_eval_on_model, cleanup_gpu, save_result
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "SmolLM-1.7B"
MODEL_PATH = "models/SmolLM-1.7B"
STATE_PATH = "results/SmolLM-1.7B_config_b_compact.pt"
LEGACY_STATE_PATH = "results/SmolLM-1.7B_config_b.pt"
RESULTS_FILE = "results/task_results_full.jsonl"

SENSITIVE_LAYERS = {2, 6, 7, 11}

def smollm_config_b(layer_idx, layer_name, layer_short):
    ln = parse_layer_num(layer_name)
    if ln in SENSITIVE_LAYERS:
        return "w8"
    if layer_short in {"q_proj", "k_proj", "v_proj"}:
        return "w8"
    if layer_short in FFN_PROJ or layer_short == "o_proj":
        return "w4"
    return "skip"

# ---- quantize ----
def step_quantize():
    if os.path.exists(STATE_PATH):
        print(f"State exists ({os.path.getsize(STATE_PATH)/1024**3:.1f} GB), skip")
        return True

    print("Quantizing SmolLM-1.7B with config_b (128 calib samples, compact state)...")
    cleanup_gpu()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        low_cpu_mem_usage=True, local_files_only=True)
    model.to("cuda:0")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)
    calib = get_calib_dataset(tokenizer, n_samples=128, max_length=2048)

    try:
        state = quantize_model_mixed_precision(model, calib, bits_w4=4, group_size=128,
                                               layer_policy=smollm_config_b)
        torch.cuda.synchronize()
        tmp_path = STATE_PATH + ".tmp"
        torch.save(state, tmp_path)
        try:
            os.replace(tmp_path, STATE_PATH)
        except PermissionError:
            import shutil
            shutil.copy2(tmp_path, STATE_PATH)
        sz = os.path.getsize(STATE_PATH) / 1024**3
        print(f"  Saved: {STATE_PATH} ({sz:.1f} GB, {len(state)} layers)")
        del state
    finally:
        del model, tokenizer, calib
        gc.collect()
        torch.cuda.empty_cache()
    cleanup_gpu()
    return True

# ---- eval ----
def step_eval():
    print("Loading quantized SmolLM-1.7B...")
    cleanup_gpu()
    if not os.path.exists(STATE_PATH):
        if os.path.exists(LEGACY_STATE_PATH):
            raise RuntimeError(
                f"Missing compact state {STATE_PATH}. Refusing to load legacy state "
                f"{LEGACY_STATE_PATH}; it is known to trigger Windows/PyTorch native crashes. "
                "Run: python experiments/exp13_smollm_config_b/run.py quantize"
            )
        raise FileNotFoundError(STATE_PATH)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        low_cpu_mem_usage=True, local_files_only=True)
    model.to("cuda:0")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, local_files_only=True)

    qs = torch.load(STATE_PATH, map_location="cpu", weights_only=False, mmap=True)
    apply_mixed_precision_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()

    TASKS = ["arc_challenge", "hellaswag", "mmlu", "gsm8k"]
    results = {}
    for task in TASKS:
        print(f"  [{task}]...")
        limit = None if task != "gsm8k" else 300
        try:
            sc = run_eval_on_model(model, tokenizer, [task], batch_size=4,
                                    max_gen_toks=256, limit=limit)
            s = sc.get(task)
            if s is not None:
                save_result(RESULTS_FILE, MODEL_NAME, "config_b", {task: s})
                results[task] = s
                print(f"    {task}: {s:.2f}")
        except Exception as e:
            print(f"    {task} ERROR: {e}")
            gc.collect(); torch.cuda.empty_cache()

    del model, tokenizer
    cleanup_gpu()

    print(f"\n{'='*60}")
    print("SmolLM-1.7B config_b results:")
    for t, s in results.items():
        print(f"  {t}: {s:.2f}")
    if "gsm8k" in results:
        print(f"\nGPTQ-W4 baseline GSM8K: 23.33, delta: {results['gsm8k'] - 23.33:+.2f}")
    return results

# ================================================================
if __name__ == "__main__":
    STEP = sys.argv[1] if len(sys.argv) > 1 else "all"
    if STEP == "quantize":
        step_quantize()
    elif STEP == "eval":
        step_eval()
    else:
        step_quantize()
        step_eval()

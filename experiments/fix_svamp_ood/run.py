"""SVAMP OOD extension: evaluate config_b variants on SVAMP.

Targets:
  Qwen2.5-1.5B: fp16, gptq, config_b_2a, config_b_2b
  SmolLM-1.7B:  fp16, gptq, config_b

Skips methods that already have svamp score in task_results_full.jsonl.
Serial GPU execution: load → eval → unload per combo.
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") in (None, "", "expandable_segments:True"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.path.insert(0, ".")

import torch, gc, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq.eval import cleanup_gpu, save_result
from ptq.quant.gptq import apply_gptq_to_model_gpu
from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate
from lm_eval.tasks import TaskManager

SVAMP_YAML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
RESULTS_FILE = "results/task_results_full.jsonl"
BATCH_SIZE = 4
MAX_GEN_TOKS = 256

# Copy svamp.yaml from exp18
SVAMP_YAML_SRC = "experiments/exp18_svamp/svamp.yaml"


def has_svamp(model_name, method_name):
    """Check if svamp score already exists for this model×method."""
    if not os.path.exists(RESULTS_FILE):
        return False
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["model"] == model_name and r["method"] == method_name:
                if "svamp" in r.get("scores", {}):
                    return True
    return False


def eval_svamp(model, tokenizer, model_name, method_name):
    """Run SVAMP evaluation on loaded model."""
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
        print(f"  SVAMP ERROR: {e}")
        del lm_eval_model
        return None

    r = results.get("results", {}).get("svamp", {})
    score = None
    for metric in ["exact_match,flexible-extract", "exact_match,none", "acc,none"]:
        if metric in r:
            score = r[metric]
            break
    score_pct = round(score * 100, 2) if score is not None else None
    if score_pct is not None:
        save_result(RESULTS_FILE, model_name, method_name, {"svamp": score_pct})
        print(f"  [{model_name}] [{method_name}] SVAMP = {score_pct:.2f}", flush=True)
    else:
        print(f"  [{model_name}] [{method_name}] SVAMP = FAILED", flush=True)

    del lm_eval_model
    gc.collect()
    torch.cuda.empty_cache()
    return score_pct


def run_combo(model_name, model_path, method_name, quant_fn, state_path):
    if has_svamp(model_name, method_name):
        print(f"[{model_name}] [{method_name}] svamp exists, skip", flush=True)
        return

    print(f"\n--- [{model_name}] [{method_name}] ---", flush=True)
    if quant_fn and state_path:
        if not os.path.exists(state_path):
            raise FileNotFoundError(state_path)
        if os.path.getsize(state_path) > 6 * 1024**3 and "_compact" not in os.path.basename(state_path):
            raise RuntimeError(
                f"Refusing to load oversized non-compact state ({os.path.getsize(state_path) / 1024**3:.2f} GB): {state_path}. "
                "Regenerate a *_compact.pt state first."
            )

    cleanup_gpu()

    load_kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True)
    if "SmolLM" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        model.to("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="cuda:0", **load_kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if quant_fn and state_path:
        qs = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
        quant_fn(model, qs)
        del qs
        gc.collect()
        torch.cuda.empty_cache()

    eval_svamp(model, tokenizer, model_name, method_name)
    del model, tokenizer
    cleanup_gpu()


def main():
    # Ensure svamp.yaml is in place
    if not os.path.exists(os.path.join(SVAMP_YAML_DIR, "svamp.yaml")):
        import shutil
        shutil.copy(SVAMP_YAML_SRC, os.path.join(SVAMP_YAML_DIR, "svamp.yaml"))
        print("Copied svamp.yaml")

    # Qwen2.5-1.5B combos
    M15 = "Qwen2.5-1.5B"
    MP15 = "models/Qwen2.5-1.5B"
    GPTQ15 = "results/Qwen2.5-1.5B_gptq_compact.pt"
    C2A = "results/Qwen2.5-1.5B_config_b_2a.pt"
    C2B = "results/Qwen2.5-1.5B_config_b_2b.pt"

    for method, qfn, sp in [
        ("fp16", None, None),
        ("gptq", apply_gptq_to_model_gpu, GPTQ15),
        ("config_b_2a", apply_mixed_precision_to_model_gpu, C2A),
        ("config_b_2b", apply_mixed_precision_to_model_gpu, C2B),
    ]:
        run_combo(M15, MP15, method, qfn, sp)

    # SmolLM-1.7B combos
    MS = "SmolLM-1.7B"
    MPS = "models/SmolLM-1.7B"
    GPTQS = "results/SmolLM-1.7B_gptq_compact.pt"
    CBS = "results/SmolLM-1.7B_config_b_compact.pt"

    for method, qfn, sp in [
        ("fp16", None, None),
        ("gptq", apply_gptq_to_model_gpu, GPTQS),
        ("config_b", apply_mixed_precision_to_model_gpu, CBS),
    ]:
        run_combo(MS, MPS, method, qfn, sp)

    print("\nAll SVAMP OOD evals done.", flush=True)


if __name__ == "__main__":
    main()

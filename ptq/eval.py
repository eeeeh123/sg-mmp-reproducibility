"""下游任务评测核心逻辑。"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") in (None, "", "expandable_segments:True"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import torch
import gc
import json
import time
import traceback

from ptq.config import MODELS, QUANT_CONFIGS, TASK_FEWSHOT, TASK_LIMIT, TASKS_ORDER
from ptq.data import get_calib_dataset
from ptq.quant.rtn import dequantize_tensor_rtn
from ptq.quant.gptq import quantize_model_gptq, apply_gptq_to_model_gpu
from ptq.quant.awq import quantize_model_awq, apply_awq_to_model_gpu
from ptq.quant.smoothquant import apply_smoothquant_to_model
from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

LARGE_STATE_LIMIT = 6 * 1024**3


def _safe_torch_load(path: str, *, kind: str = "quantized state"):
    """Load a quantized state with a guard against legacy oversized .pt files."""
    if os.path.exists(path):
        size = os.path.getsize(path)
        if size > LARGE_STATE_LIMIT and "_compact" not in os.path.basename(path):
            if os.environ.get("PTQ_ALLOW_UNSAFE_LARGE_STATE_LOAD") != "1":
                raise RuntimeError(
                    f"Refusing to load oversized non-compact {kind} ({size / 1024**3:.2f} GB): {path}. "
                    "This legacy state format has triggered native Windows/PyTorch access violations. "
                    "Regenerate a *_compact.pt state, or set PTQ_ALLOW_UNSAFE_LARGE_STATE_LOAD=1 only for manual debugging."
                )
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_hf_model_safe(model_path: str, dtype=torch.float16):
    """Load SmolLM through CPU first to avoid Windows device_map native crashes."""
    kwargs = dict(torch_dtype=dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    if "SmolLM" in model_path:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        model.to("cuda:0")
        return model
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        **kwargs,
    )


def cleanup_gpu():
    """彻底清理 GPU 显存，避免 WDDM 碎片化导致下次加载 segfault。"""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    time.sleep(2)


def load_quantized_model(model_name: str, method: str, model_dir: str = "models",
                         results_dir: str = "results"):
    """加载模型并应用量化方法，返回 GPU 上的模型。"""
    import torch.nn as nn
    model_path = os.path.join(model_dir, model_name)
    dtype = torch.float16

    model = _load_hf_model_safe(model_path, dtype=dtype)
    model.eval()
    device = next(model.parameters()).device

    if method == "fp16":
        return model

    if method == "rtn":
        compact_path = os.path.join(results_dir, f"{model_name}_rtn_compact.pt")
        state_path = os.path.join(results_dir, f"{model_name}_rtn.pt")
        load_path = compact_path if os.path.exists(compact_path) else state_path
        quant_state = _safe_torch_load(load_path, kind="RTN state")
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and name in quant_state:
                qi = quant_state[name]
                w_deq = dequantize_tensor_rtn(
                    qi["w_q"].to(device), qi["scale"].to(device),
                    qi["zero"].to(device), qi["group_size"],
                ).to(module.weight.dtype)
                module.weight.data.copy_(w_deq)
        del quant_state
        gc.collect()
        torch.cuda.empty_cache()
        return model

    elif method == "gptq":
        compact_path = os.path.join(results_dir, f"{model_name}_gptq_compact.pt")
        state_path = os.path.join(results_dir, f"{model_name}_gptq.pt")
        if os.path.exists(compact_path):
            quant_state = _safe_torch_load(compact_path, kind="GPTQ state")
            apply_gptq_to_model_gpu(model, quant_state)
            return model
        elif os.path.exists(state_path):
            quant_state = _safe_torch_load(state_path, kind="GPTQ state")
            apply_gptq_to_model_gpu(model, quant_state)
            return model
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            calib_data = get_calib_dataset(tokenizer, n_samples=128, max_length=2048)
            quant_state = quantize_model_gptq(model, calib_data, bits=4, group_size=128)
            torch.save(quant_state, state_path)
            apply_gptq_to_model_gpu(model, quant_state)
            return model

    elif method == "awq":
        compact_path = os.path.join(results_dir, f"{model_name}_awq_compact.pt")
        state_path = os.path.join(results_dir, f"{model_name}_awq.pt")
        scales_path = os.path.join(results_dir, f"{model_name}_awq_scales.pt")
        if os.path.exists(compact_path) and os.path.exists(scales_path):
            quant_state = _safe_torch_load(compact_path, kind="AWQ state")
            awq_scales = _safe_torch_load(scales_path, kind="AWQ scales")
            apply_awq_to_model_gpu(model, quant_state, awq_scales)
            return model
        elif os.path.exists(state_path) and os.path.exists(scales_path):
            quant_state = _safe_torch_load(state_path, kind="AWQ state")
            awq_scales = _safe_torch_load(scales_path, kind="AWQ scales")
            apply_awq_to_model_gpu(model, quant_state, awq_scales)
            return model
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            calib_data = get_calib_dataset(tokenizer, n_samples=32, max_length=2048)
            quant_state, awq_scales = quantize_model_awq(model, calib_data, bits=4, group_size=128)
            torch.save(quant_state, state_path)
            torch.save(awq_scales, scales_path)
            apply_awq_to_model_gpu(model, quant_state, awq_scales)
            return model

    elif method == "smoothquant":
        scales_path = os.path.join(results_dir, f"{model_name}_smoothquant.pt")
        scales = _safe_torch_load(scales_path, kind="SmoothQuant scales")
        apply_smoothquant_to_model(model, scales)
        return model

    elif method == "mixed_precision":
        compact_path = os.path.join(results_dir, f"{model_name}_mixed_precision_compact.pt")
        state_path = os.path.join(results_dir, f"{model_name}_mixed_precision.pt")
        state_path = compact_path if os.path.exists(compact_path) else state_path
        if os.path.exists(state_path):
            quant_state = _safe_torch_load(state_path, kind="mixed-precision state")
            apply_mixed_precision_to_model_gpu(model, quant_state)
            return model
        else:
            from ptq.data import get_calib_dataset
            from ptq.quant.mixed_precision import quantize_model_mixed_precision
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            calib_data = get_calib_dataset(tokenizer, n_samples=128, max_length=2048)
            quant_state = quantize_model_mixed_precision(model, calib_data)
            torch.save(quant_state, state_path)
            apply_mixed_precision_to_model_gpu(model, quant_state)
            return model

    raise ValueError(f"Unknown method: {method}")


def run_eval(model_name: str, method: str, tasks: list, batch_size: int = 4,
             max_gen_toks: int = 256, limit=None, model_dir: str = "models",
             results_dir: str = "results") -> dict:
    """运行一次评测，返回 {task: score}。"""
    cleanup_gpu()

    model_path = os.path.join(model_dir, model_name)
    model = load_quantized_model(model_name, method, model_dir, results_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    lm_eval_model = HFLM(
        pretrained=model, tokenizer=tokenizer,
        batch_size=batch_size, max_batch_size=batch_size,
    )

    results = simple_evaluate(
        model=lm_eval_model, tasks=tasks, batch_size=batch_size,
        limit=limit, log_samples=False,
        gen_kwargs={"temperature": 0.0, "max_new_tokens": max_gen_toks, "do_sample": False},
    )

    task_scores = {}
    for task in tasks:
        r = results["results"].get(task, {}) or results.get("groups", {}).get(task, {})
        score = None
        for metric in ["acc_norm,none", "acc,none", "exact_match,flexible-extract",
                       "exact_match,strict-match", "exact_match,none", "flexible_extract,none"]:
            if metric in r:
                score = r[metric]
                break
        task_scores[task] = round(score * 100, 2) if score is not None else None

    del model, lm_eval_model, tokenizer, results
    cleanup_gpu()
    return task_scores


def run_eval_on_model(model, tokenizer, tasks: list, batch_size: int = 4,
                      max_gen_toks: int = 256, limit=None) -> dict:
    """对已加载的模型直接评测，不重新加载。exp02/exp04 使用。

    model 保留在 GPU 上，只清理 HFLM 产生的临时显存。
    """
    lm_eval_model = HFLM(
        pretrained=model, tokenizer=tokenizer,
        batch_size=batch_size, max_batch_size=batch_size,
    )

    results = simple_evaluate(
        model=lm_eval_model, tasks=tasks, batch_size=batch_size,
        limit=limit, log_samples=False,
        gen_kwargs={"temperature": 0.0, "max_new_tokens": max_gen_toks, "do_sample": False},
    )

    task_scores = {}
    for task in tasks:
        r = results["results"].get(task, {}) or results.get("groups", {}).get(task, {})
        score = None
        for metric in ["acc_norm,none", "acc,none", "exact_match,flexible-extract",
                       "exact_match,strict-match", "exact_match,none", "flexible_extract,none"]:
            if metric in r:
                score = r[metric]
                break
        task_scores[task] = round(score * 100, 2) if score is not None else None

    del lm_eval_model, results
    gc.collect()
    torch.cuda.empty_cache()
    return task_scores


def load_existing_results(output_file: str) -> dict:
    """读取已有结果，返回 {(model, method): {task: score}}."""
    existing = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                key = (r["model"], r["method"])
                if key not in existing:
                    existing[key] = {}
                existing[key].update(r.get("scores", {}))
    return existing


def save_result(output_file: str, model_name: str, method: str, scores: dict):
    """追加/更新一条记录到 JSONL。"""
    records = []
    found = False
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                if r["model"] == model_name and r["method"] == method:
                    r["scores"].update(scores)
                    found = True
                records.append(r)
    if not found:
        records.append({"model": model_name, "method": method, "scores": scores})
    with open(output_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def run_experiment(model_names=None, methods=None, output_file="results/task_results_full.jsonl",
                   batch_size=4, max_gen_toks=256, retry=1,
                   model_dir="models", results_dir="results"):
    """运行完整实验：遍历 model×method 组合，逐任务评测，支持断点续跑。

    返回最终结果列表。
    """
    if model_names is None:
        model_names = [m["name"] for m in MODELS]
    if methods is None:
        methods = [q["method"] for q in QUANT_CONFIGS]

    existing = load_existing_results(output_file)
    total_combos = len(model_names) * len(methods)
    current_combo = 0

    for model_name in model_names:
        for method in methods:
            current_combo += 1
            existing_tasks = existing.get((model_name, method), {})
            pending_tasks = [t for t in TASKS_ORDER if existing_tasks.get(t) is None]

            if not pending_tasks:
                print(f"[{current_combo}/{total_combos}] {model_name} [{method}]: all done, skip")
                continue

            print(f"\n{'='*60}")
            print(f"[{current_combo}/{total_combos}] {model_name} [{method}]")
            print(f"  Pending: {pending_tasks}")
            print(f"{'='*60}")

            for task in pending_tasks:
                print(f"\n  --- {task} ---")
                t_start = time.time()

                for attempt in range(retry + 1):
                    try:
                        limit = TASK_LIMIT.get(task, None)
                        scores = run_eval(model_name, method, [task],
                                          batch_size=batch_size, max_gen_toks=max_gen_toks,
                                          limit=limit, model_dir=model_dir,
                                          results_dir=results_dir)
                        if scores.get(task) is not None:
                            save_result(output_file, model_name, method, scores)
                            elapsed = time.time() - t_start
                            print(f"  {task}: {scores[task]:.2f}% ({elapsed:.0f}s)")
                            break
                        else:
                            print(f"  {task}: score extraction failed, attempt {attempt+1}")
                    except Exception as e:
                        print(f"  {task}: ERROR (attempt {attempt+1}): {e}")
                        traceback.print_exc()
                        gc.collect()
                        torch.cuda.empty_cache()
                        if attempt < retry:
                            print("  Retrying in 10s...")
                            time.sleep(10)
                        else:
                            print(f"  {task}: FAILED after {retry+1} attempts, skipping")
                            save_result(output_file, model_name, method, {task: None})

    print("\n" + "=" * 60)
    print("All done. Results saved to", output_file)
    print("=" * 60)
    return load_existing_results(output_file)

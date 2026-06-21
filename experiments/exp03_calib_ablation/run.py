#!/usr/bin/env python
"""exp03_calib_ablation: 校准数据领域消融。

对比 GPTQ/AWQ 使用 WikiText vs GSM8K 校准数据时，下游任务表现差异。
3 模型 × 2 方法 × 2 校准源 = 12 组合，每组合跑 GSM8K + ARC。

用法:
  # Step 1: 量化 (先跑完所有需要的量化)
  python experiments/exp03_calib_ablation/run.py --step quantize
  python experiments/exp03_calib_ablation/run.py --step quantize --model Qwen2.5-0.5B --method gptq --calib gsm8k

  # Step 2: 评测 (加载已量化的 state 文件)
  python experiments/exp03_calib_ablation/run.py --step eval
  python experiments/exp03_calib_ablation/run.py --step eval --model Qwen2.5-0.5B --method gptq --calib gsm8k
"""
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import gc
import json
import time
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq.config import MODELS, TASK_LIMIT
from ptq.data import get_calib_dataset
from ptq.eval import run_eval_on_model, cleanup_gpu
from ptq.quant.gptq import quantize_model_gptq, apply_gptq_to_model_gpu
from ptq.quant.awq import quantize_model_awq, apply_awq_to_model_gpu


RESULTS_ROOT = os.path.join(_project_root, "results")


def step_quantize(models_to_run, methods, calibs):
    """Step 1: 对每个 model×method×calib 组合做量化，保存 state 文件。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    total = len(models_to_run) * len(methods) * len(calibs)
    current = 0

    for model_name in models_to_run:
        for method in methods:
            for calib in calibs:
                current += 1
                state_path = os.path.join(RESULTS_ROOT, f"{model_name}_{method}_{calib}.pt")
                scales_path = os.path.join(RESULTS_ROOT, f"{model_name}_awq_scales_{calib}.pt")

                # For wikitext, reuse existing state files if they exist
                if calib == "wikitext":
                    existing = os.path.join(RESULTS_ROOT, f"{model_name}_{method}.pt")
                    if os.path.exists(existing) and not os.path.exists(state_path):
                        # Symlink or copy
                        import shutil
                        shutil.copy(existing, state_path)
                        if method == "awq":
                            existing_scales = os.path.join(RESULTS_ROOT, f"{model_name}_awq_scales.pt")
                            if os.path.exists(existing_scales):
                                shutil.copy(existing_scales, scales_path)
                        print(f"[{current}/{total}] {model_name} {method}/{calib}: copied from existing")
                        continue

                if os.path.exists(state_path) and (method != "awq" or os.path.exists(scales_path)):
                    print(f"[{current}/{total}] {model_name} {method}/{calib}: state exists, skip")
                    continue

                print(f"\n{'='*60}")
                print(f"[{current}/{total}] Quantize: {model_name} [{method}] calib={calib}")
                print(f"{'='*60}")

                model = None
                tokenizer = None
                try:
                    cleanup_gpu()
                    model_path = os.path.join(_project_root, "models", model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, torch_dtype=torch.float16, trust_remote_code=True,
                        device_map="cuda:0", low_cpu_mem_usage=True,
                    )
                    model.eval()
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

                    print(f"  Loading calibration data ({calib})...")
                    n_samples = 64 if (method == "gptq" and calib == "gsm8k") else (128 if method == "gptq" else 32)
                    max_len = 1024 if calib == "gsm8k" else 2048
                    calib_data = get_calib_dataset(
                        tokenizer, n_samples=n_samples, max_length=max_len, dataset_name=calib)

                    print(f"  Running quantization...")
                    if method == "gptq":
                        quant_state = quantize_model_gptq(model, calib_data, bits=4, group_size=128)
                        torch.save(quant_state, state_path)
                        print(f"  Saved: {state_path}")
                    elif method == "awq":
                        quant_state, awq_scales = quantize_model_awq(model, calib_data, bits=4, group_size=128)
                        torch.save(quant_state, state_path)
                        torch.save(awq_scales, scales_path)
                        print(f"  Saved: {state_path}, {scales_path}")

                except Exception as e:
                    print(f"  QUANTIZE ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    if model is not None:
                        del model
                    if tokenizer is not None:
                        del tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    cleanup_gpu()

    print("\nQuantization step done.")


def step_eval(models_to_run, methods, calibs, batch_size):
    """Step 2: 加载量化 state 文件，跑评测。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "calib_ablation.jsonl")

    # Load existing results
    existing = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                key = (r["model"], r["method"], r.get("calib", "wikitext"))
                existing[key] = r.get("scores", {})

    tasks = ["gsm8k", "arc_challenge"]
    total = len(models_to_run) * len(methods) * len(calibs)
    current = 0

    for model_name in models_to_run:
        for method in methods:
            for calib in calibs:
                current += 1
                key = (model_name, method, calib)
                existing_scores = existing.get(key, {})
                pending = [t for t in tasks if existing_scores.get(t) is None]

                if not pending:
                    print(f"[{current}/{total}] {model_name} {method}/{calib}: all done, skip")
                    continue

                state_path = os.path.join(RESULTS_ROOT, f"{model_name}_{method}_{calib}.pt")
                scales_path = os.path.join(RESULTS_ROOT, f"{model_name}_awq_scales_{calib}.pt")

                if not os.path.exists(state_path):
                    print(f"[{current}/{total}] {model_name} {method}/{calib}: state file missing, run --step quantize first")
                    continue

                print(f"\n{'='*60}")
                print(f"[{current}/{total}] Eval: {model_name} [{method}] calib={calib}")
                print(f"  Pending: {pending}")
                print(f"{'='*60}")

                try:
                    cleanup_gpu()
                    model_path = os.path.join(_project_root, "models", model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path, torch_dtype=torch.float16, trust_remote_code=True,
                        device_map="cuda:0", low_cpu_mem_usage=True,
                    )
                    model.eval()
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

                    print(f"  Loading quant state: {state_path}")
                    if method == "gptq":
                        quant_state = torch.load(state_path, map_location="cpu", weights_only=False)
                        apply_gptq_to_model_gpu(model, quant_state)
                    elif method == "awq":
                        quant_state = torch.load(state_path, map_location="cpu", weights_only=False)
                        awq_scales = torch.load(scales_path, map_location="cpu", weights_only=False)
                        apply_awq_to_model_gpu(model, quant_state, awq_scales)

                    all_scores = {}
                    for task in pending:
                        limit = TASK_LIMIT.get(task, None)
                        t_start = time.time()
                        scores = run_eval_on_model(
                            model, tokenizer, [task], batch_size=batch_size,
                            max_gen_toks=256, limit=limit)
                        all_scores.update(scores)
                        elapsed = time.time() - t_start
                        print(f"  {task}: {scores.get(task)}% ({elapsed:.0f}s)")

                    # Save
                    records = {}
                    if os.path.exists(output_file):
                        with open(output_file) as f:
                            for line in f:
                                r = json.loads(line)
                                k = (r["model"], r["method"], r.get("calib", "wikitext"))
                                records[k] = r
                    if key in records:
                        records[key]["scores"].update(all_scores)
                    else:
                        records[key] = {"model": model_name, "method": method, "calib": calib, "scores": all_scores}
                    with open(output_file, "w") as f:
                        for r in records.values():
                            f.write(json.dumps(r) + "\n")

                except Exception as e:
                    print(f"  EVAL ERROR: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    del model, tokenizer
                    gc.collect()
                    torch.cuda.empty_cache()
                    cleanup_gpu()

    print(f"\nEval step done. Results: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="exp03_calib_ablation")
    parser.add_argument("--step", type=str, required=True, choices=["quantize", "eval"])
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--calib", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    models_to_run = args.model.split(",") if args.model else [m["name"] for m in MODELS]
    methods = args.method.split(",") if args.method else ["gptq", "awq"]
    calibs = args.calib.split(",") if args.calib else ["wikitext", "gsm8k"]

    if args.step == "quantize":
        step_quantize(models_to_run, methods, calibs)
    elif args.step == "eval":
        step_eval(models_to_run, methods, calibs, args.batch_size)


if __name__ == "__main__":
    main()

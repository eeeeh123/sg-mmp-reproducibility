"""方差测试：对指定 model 的指定 tasks 跑多轮，输出均值±标准差。

用法:
  python scripts/variance_test.py --model "Qwen2.5-0.5B" --method fp16 --tasks "arc_challenge,hellaswag" --runs 3
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import sys
import torch
import gc
import json
import numpy as np

sys.path.insert(0, ".")

from transformers import AutoTokenizer
from lm_eval.models.huggingface import HFLM
from lm_eval import simple_evaluate

TASK_FEWSHOT = {
    "mmlu": 5,
    "hellaswag": 10,
    "arc_challenge": 0,
    "gsm8k": 5,
}


def load_quantized_model(model_name: str, method: str):
    from transformers import AutoModelForCausalLM
    model_path = f"models/{model_name}"
    dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    model.eval()

    if method == "fp16":
        return model.cuda()

    if method == "rtn":
        from ptq.quant.rtn import apply_rtn_to_model
        quant_state = torch.load(f"results/{model_name}_rtn.pt", map_location="cpu", weights_only=False)
        apply_rtn_to_model(model, quant_state)
        return model.cuda()

    elif method == "gptq":
        from ptq.quant.gptq import apply_gptq_to_model_gpu
        model = model.cuda()
        quant_state = torch.load(f"results/{model_name}_gptq.pt", map_location="cpu", weights_only=False)
        apply_gptq_to_model_gpu(model, quant_state)
        return model

    elif method == "awq":
        from ptq.quant.awq import apply_awq_to_model_gpu
        model = model.cuda()
        quant_state = torch.load(f"results/{model_name}_awq.pt", map_location="cpu", weights_only=False)
        awq_scales = torch.load(f"results/{model_name}_awq_scales.pt", map_location="cpu", weights_only=False)
        apply_awq_to_model_gpu(model, quant_state, awq_scales)
        return model

    elif method == "smoothquant":
        from ptq.quant.smoothquant import apply_smoothquant_to_model
        scales = torch.load(f"results/{model_name}_smoothquant.pt", map_location="cpu", weights_only=False)
        apply_smoothquant_to_model(model, scales)
        return model.cuda()

    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_gen_toks", type=int, default=256)
    args = parser.parse_args()

    tasks = args.tasks.split(",")
    print(f"=== Variance Test: {args.model} [{args.method}] × {args.runs} runs ===")
    print(f"  Tasks: {tasks}")

    # Load once
    model_path = f"models/{args.model}"
    model = load_quantized_model(args.model, args.method)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    all_scores = {t: [] for t in tasks}

    for run_idx in range(args.runs):
        print(f"\n--- Run {run_idx + 1}/{args.runs} ---")

        lm_eval_model = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_batch_size=args.batch_size,
        )

        results = simple_evaluate(
            model=lm_eval_model,
            tasks=tasks,
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=False,
            gen_kwargs={
                "temperature": 0.0,
                "max_new_tokens": args.max_gen_toks,
                "do_sample": False,
            },
        )

        for task in tasks:
            score = None
            r = results["results"].get(task, {})
            for metric in ["acc_norm,none", "acc,none", "exact_match,flexible-extract",
                           "exact_match,strict-match", "exact_match,none", "flexible_extract,none"]:
                if metric in r:
                    score = r[metric]
                    break
            all_scores[task].append(score)
            print(f"  {task}: {score*100:.2f}%" if score else f"  {task}: None")

        del lm_eval_model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n=== Summary: {args.model} [{args.method}] ===")
    for task in tasks:
        scores = [s * 100 for s in all_scores[task] if s is not None]
        if len(scores) >= 2:
            mean = np.mean(scores)
            std = np.std(scores, ddof=1)
            print(f"  {task}: {mean:.2f}% ± {std:.2f}%  (values: {[f'{v:.2f}' for v in scores]})")
        elif len(scores) == 1:
            print(f"  {task}: {scores[0]:.2f}% (single run)")
        else:
            print(f"  {task}: no valid scores")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

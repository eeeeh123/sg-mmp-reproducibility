#!/usr/bin/env python
"""统一评测入口 — 薄 CLI 包装 ptq.eval。

用法:
  python scripts/evaluate.py                          # 全部 model×method
  python scripts/evaluate.py --model Qwen2.5-0.5B --method fp16
  python scripts/evaluate.py --model SmolLM-1.7B --method awq,smoothquant
  python scripts/evaluate.py --output experiments/exp01_baseline/results/task_results.jsonl
"""
import argparse
from ptq.eval import run_experiment


def main():
    parser = argparse.ArgumentParser(description="PTQ Benchmark — Downstream Task Evaluation")
    parser.add_argument("--model", type=str, default=None, help="Model name or comma-separated")
    parser.add_argument("--method", type=str, default=None, help="Method or comma-separated")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_gen_toks", type=int, default=256)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--output", type=str, default="results/task_results_full.jsonl")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    model_names = args.model.split(",") if args.model else None
    methods = args.method.split(",") if args.method else None

    run_experiment(
        model_names=model_names,
        methods=methods,
        output_file=args.output,
        batch_size=args.batch_size,
        max_gen_toks=args.max_gen_toks,
        retry=args.retry,
        model_dir=args.model_dir,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()

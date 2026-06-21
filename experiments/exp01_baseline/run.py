#!/usr/bin/env python
"""exp01_baseline: 基础 PTQ 方法对比。

在 3 个模型 (Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM-1.7B) 上对比 5 种量化方法
(fp16, RTN, GPTQ, AWQ, SmoothQuant) 在 4 个下游任务 (ARC, HellaSwag, MMLU, GSM8K)
上的表现。

结果输出到 exp01_baseline/results/task_results.jsonl

用法:
  python experiments/exp01_baseline/run.py
  python experiments/exp01_baseline/run.py --model Qwen2.5-0.5B --method fp16
"""
import os
import sys

# 确保项目根目录在 sys.path 中
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ptq.eval import run_experiment

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "task_results.jsonl")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="exp01_baseline")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--retry", type=int, default=1)
    args = parser.parse_args()

    model_names = args.model.split(",") if args.model else None
    methods = args.method.split(",") if args.method else None

    run_experiment(
        model_names=model_names,
        methods=methods,
        output_file=OUTPUT_FILE,
        batch_size=args.batch_size,
        retry=args.retry,
        model_dir=os.path.join(_project_root, "models"),
        results_dir=os.path.join(_project_root, "results"),
    )


if __name__ == "__main__":
    main()

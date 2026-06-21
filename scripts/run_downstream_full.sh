#!/bin/bash
# 完整数据集下游任务评测
# 支持随时中断：Ctrl+C 停止后，重新运行此脚本自动从断点继续
# 输出: results/task_results_full.jsonl

set +e  # 不因单任务失败退出

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

cd "$(dirname "$0")/.." || exit 1

echo "============================================"
echo "Full Dataset Downstream Task Evaluation"
echo "Output: results/task_results_full.jsonl"
echo "Ctrl+C to stop, re-run to resume"
echo "============================================"

python -u scripts/evaluate.py --retry 1 2>&1

echo "Script finished."

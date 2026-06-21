#!/bin/bash
# 批量评测脚本：补齐所有缺失的 WikiText-2 和下游任务评测
# 用法: bash scripts/run_all_evals.sh
set -e

cd "$(dirname "$0")/.."

MODELS=("Qwen2.5-0.5B" "Qwen2.5-1.5B" "SmolLM-1.7B")
METHODS=("fp16" "rtn" "gptq" "awq" "smoothquant")

echo "=========================================="
echo "Phase 1: WikiText-2 Perplexity (all missing)"
echo "=========================================="

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    # 跳过已完成的：grep 检查 perplexity.jsonl 中是否已有该 model+method
    if grep -q "\"model\": \"$model\".*\"method\": \"$method\"" results/perplexity.jsonl 2>/dev/null; then
      echo "SKIP (done): $model [$method] perplexity"
      continue
    fi
    echo ">>> $model [$method] WikiText-2 PPL"
    CUDA_VISIBLE_DEVICES=0 python scripts/03_eval_perplexity.py --model "$model" --method "$method"
  done
done

echo ""
echo "=========================================="
echo "Phase 2: Downstream Tasks (all 4 tasks; GSM8K re-run)"
echo "=========================================="

# 先备份旧结果，清空 task_results 以便全量重跑
if [ -f results/task_results.jsonl ]; then
  mv results/task_results.jsonl "results/task_results_backup_$(date +%Y%m%d_%H%M%S).jsonl"
fi

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    echo ">>> $model [$method] downstream tasks"
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
      --model "$model" --method "$method"
  done
done

echo ""
echo "=========================================="
echo "Phase 3: Aggregate Results"
echo "=========================================="
python scripts/04_aggregate.py

echo "Done!"

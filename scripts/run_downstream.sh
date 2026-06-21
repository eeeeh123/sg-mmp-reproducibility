#!/bin/bash
# 全量下游任务评测脚本 (GSM8K 5-shot, MMLU 5-shot, HellaSwag 10-shot, ARC 25-shot)
# limit=100 加速评测
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0

MODELS=("Qwen2.5-0.5B" "Qwen2.5-1.5B" "SmolLM-1.7B")
METHODS=("fp16" "rtn" "gptq" "awq" "smoothquant")
TASKS="mmlu,hellaswag,arc_challenge,gsm8k"

for model in "${MODELS[@]}"; do
  for method in "${METHODS[@]}"; do
    # Check if already completed
    if grep -q "\"model\": \"$model\".*\"method\": \"$method\"" results/task_results.jsonl 2>/dev/null; then
      # Check if ALL 4 tasks present
      HAS_GSM8K=$(grep "\"model\": \"$model\"" results/task_results.jsonl | grep "\"method\": \"$method\"" | grep -c "gsm8k" || true)
      if [ "$HAS_GSM8K" -gt 0 ]; then
        echo "SKIP (done): $model [$method]"
        continue
      fi
    fi
    echo "=== $(date): $model [$method] ==="
    python -u scripts/evaluate.py \
      --model "$model" --method "$method"
  done
done

echo "=== Done! ==="
python scripts/04_aggregate.py

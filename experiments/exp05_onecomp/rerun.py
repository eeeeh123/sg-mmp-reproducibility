"""exp05 重跑：更大校准数据 64 samples × 1024 tokens.

仅跑 GSM8K + ARC-C。支持断点续跑：量化模型存在则跳过量化，
JSONL 已有分数则跳过评测。

用法:
  .../Python/Python313/python.exe experiments/exp05_onecomp/rerun.py
"""

import os, sys, json, time, gc
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, ".")
import torch

from ptq.eval import run_eval_on_model, cleanup_gpu, save_result

MODEL_ID = "models/Qwen2.5-0.5B"
MODEL_NAME = "Qwen2.5-0.5B"
TASKS = ["arc_challenge", "gsm8k"]
OUTPUT_FILE = "results/task_results_full.jsonl"

# 新方法名，与旧版区分
GAMMAS = [
    (0.0, "onecomp_qep_g0.0_64x1024"),
    (1.0, "onecomp_qep_g1.0_64x1024"),
]

from onecomp import Runner, ModelConfig, CalibrationConfig, QEPConfig
from onecomp.quantizer.gptq import GPTQ


def load_existing_scores(method: str) -> dict:
    """读取 JSONL 中已有的分数。"""
    if not os.path.exists(OUTPUT_FILE):
        return {}
    with open(OUTPUT_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r.get("model") == MODEL_NAME and r.get("method") == method:
                return r.get("scores", {})
    return {}


def quantize_if_needed(gamma: float, method: str) -> str:
    """如果量化模型不存在则量化，返回保存目录。"""
    save_dir = f"results/{MODEL_NAME}_{method}"
    if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "model.safetensors")):
        print(f"[SKIP] Quantized model exists: {save_dir}")
        return save_dir

    print(f"\n{'='*60}")
    print(f"Quantizing {MODEL_NAME} GPTQ+QEP gamma={gamma} (64×1024)")
    print(f"{'='*60}")
    t0 = time.time()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizer = GPTQ(wbits=4, groupsize=128)
    calib_config = CalibrationConfig(
        calibration_dataset="wikitext2",
        num_calibration_samples=64,
        max_length=1024,
    )
    qep_config = QEPConfig(
        perccorr=gamma,
        percdamp=0.01,
        general=False,  # 架构感知，内部分 batch_size=16，省显存
        device="cuda:0",
    )

    runner = Runner(
        model_config=model_config, quantizer=quantizer,
        calibration_config=calib_config, qep=True, qep_config=qep_config,
    )
    runner.run()
    print(f"Quantization done in {time.time() - t0:.0f}s")

    runner.save_dequantized_model(save_dir)
    print(f"Saved to {save_dir}")

    del runner, quantizer, model_config
    gc.collect()
    torch.cuda.empty_cache()
    return save_dir


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"exp05 rerun: 64×1024 calib, GSM8K+ARC-C")
    print(f"Model: {MODEL_NAME}")

    for gamma, method in GAMMAS:
        print(f"\n{'#'*60}")
        print(f"# {method}")
        print(f"{'#'*60}")

        # 检查已有分数
        existing = load_existing_scores(method)
        pending = [t for t in TASKS if t not in existing]
        if not pending:
            print(f"[SKIP] All tasks done: {existing}")
            continue
        print(f"Existing: {list(existing.keys())}, Pending: {pending}")

        # 1. 量化
        save_dir = quantize_if_needed(gamma, method)

        # 2. 加载模型
        cleanup_gpu()
        print(f"Loading model from {save_dir}...")
        model = AutoModelForCausalLM.from_pretrained(
            save_dir, torch_dtype=torch.float16, trust_remote_code=True,
            device_map="cuda:0",
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)

        # 3. 评测
        for task in pending:
            print(f"\nEvaluating {task} (limit={300 if task == 'gsm8k' else 'full'})...")
            t0 = time.time()
            try:
                limit = 300 if task == "gsm8k" else None
                scores = run_eval_on_model(model, tokenizer, [task], limit=limit)
                save_result(OUTPUT_FILE, MODEL_NAME, method, scores)
                elapsed = time.time() - t0
                sc = f"{scores[task]:.2f}%" if scores.get(task) is not None else "FAILED"
                print(f"  {task}: {sc} ({elapsed:.0f}s)")
            except Exception as e:
                print(f"  {task}: ERROR: {e}")
                save_result(OUTPUT_FILE, MODEL_NAME, method, {task: None})

        del model, tokenizer
        cleanup_gpu()

    # 汇总
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    with open(OUTPUT_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r["model"] == MODEL_NAME and "onecomp" in r["method"]:
                sc = r["scores"]
                arc = f"{sc.get('arc_challenge', 'N/A'):.2f}%" if isinstance(sc.get('arc_challenge'), (int, float)) else 'N/A'
                gsm = f"{sc.get('gsm8k', 'N/A'):.2f}%" if isinstance(sc.get('gsm8k'), (int, float)) else 'N/A'
                print(f"{r['method']}: ARC-C={arc}, GSM8K={gsm}")

    print("Done.")


if __name__ == "__main__":
    main()

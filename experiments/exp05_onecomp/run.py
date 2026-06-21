"""exp05: onecomp GPTQ+QEP 量化实验 (创新一方法).

gamma = perccorr，控制 QEP 修正强度：
  - gamma=0: 纯 GPTQ，无误差传播修正
  - gamma=1.0: 全量 QEP 修正

必须用 Python 3.13 (onecomp 所在环境):
  C:/Users/.../Python/Python313/python.exe experiments/exp05_onecomp/run.py
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys
sys.path.insert(0, ".")

import json
import time
import gc
import torch
from datetime import datetime

from ptq.eval import run_eval_on_model, cleanup_gpu, save_result
from ptq.config import TASK_LIMIT, EVAL_MAX_LENGTH

MODEL_ID = "models/Qwen2.5-0.5B"
MODEL_NAME = "Qwen2.5-0.5B"

GAMMA_VALUES = [0.0, 1.0]
TASKS = ["arc_challenge", "gsm8k"]
OUTPUT_FILE = "results/task_results_full.jsonl"
PPL_FILE = "results/perplexity.jsonl"

from onecomp import Runner, ModelConfig, CalibrationConfig, QEPConfig
from onecomp.quantizer.gptq import GPTQ


def run_onecomp_quantize(gamma: float) -> str:
    """GPTQ+QEP 量化，保存 dequantized 模型，返回保存路径。"""
    save_dir = f"results/{MODEL_NAME}_onecomp_qep_g{gamma}"
    if os.path.exists(save_dir):
        import shutil
        shutil.rmtree(save_dir)

    print(f"\n{'='*60}")
    print(f"Quantizing {MODEL_NAME} with GPTQ+QEP, gamma={gamma}")
    print(f"{'='*60}")

    t0 = time.time()

    model_config = ModelConfig(model_id=MODEL_ID, device="cuda:0")
    quantizer = GPTQ(wbits=4, groupsize=128)
    calib_config = CalibrationConfig(
        calibration_dataset="wikitext2",
        num_calibration_samples=32,
        max_length=256,
    )
    qep_config = QEPConfig(
        perccorr=gamma,
        percdamp=0.01,
        general=True,  # 逐层 QEP，8GB 显存可用
        device="cuda:0",
    )

    runner = Runner(
        model_config=model_config,
        quantizer=quantizer,
        calibration_config=calib_config,
        qep=True,
        qep_config=qep_config,
    )
    runner.run()

    elapsed = time.time() - t0
    print(f"Quantization done in {elapsed:.1f}s")

    runner.save_dequantized_model(save_dir)
    print(f"Dequantized model saved to {save_dir}")

    del runner, quantizer, model_config
    gc.collect()
    torch.cuda.empty_cache()

    return save_dir


@torch.no_grad()
def compute_wikitext_ppl(model, tokenizer, device) -> float:
    """逐文档计算 WikiText-2 困惑度。"""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    model.eval()

    nll_sum = 0.0
    n_tokens = 0
    max_length = EVAL_MAX_LENGTH

    for doc in ds["text"]:
        if not doc or not doc.strip():
            continue
        tokens = tokenizer.encode(doc.strip(), add_special_tokens=False)
        if len(tokens) < 2:
            continue
        tokens = [tokenizer.bos_token_id] + tokens if tokenizer.bos_token_id is not None else tokens
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        L = token_tensor.shape[1]
        if L <= max_length:
            logits = model(token_tensor).logits
            logits = logits[:, :-1, :].contiguous()
            targets = token_tensor[:, 1:].contiguous()
            nll = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1), reduction="sum")
            nll_sum += nll.item()
            n_tokens += L - 1
        else:
            stride = max_length // 2
            for begin in range(0, L, stride):
                end = min(begin + max_length, L)
                chunk = token_tensor[:, begin:end]
                logits = model(chunk).logits
                if begin == 0:
                    logits = logits[:, :-1, :].contiguous()
                    targets = chunk[:, 1:].contiguous()
                    trg_len = chunk.shape[1] - 1
                else:
                    n_prev = max_length // 2
                    logits = logits[:, n_prev:-1, :].contiguous()
                    targets = chunk[:, n_prev + 1:].contiguous()
                    trg_len = chunk.shape[1] - n_prev - 1
                if trg_len > 0:
                    nll = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]), targets.view(-1), reduction="sum")
                    nll_sum += nll.item()
                    n_tokens += trg_len
                if end >= L:
                    break

    return torch.exp(torch.tensor(nll_sum / max(1, n_tokens))).item()


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"exp05: onecomp GPTQ+QEP @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {MODEL_NAME}, Gammas: {GAMMA_VALUES}")

    all_results = []

    for gamma in GAMMA_VALUES:
        method_label = f"onecomp_qep_g{gamma}"

        # 1. GPTQ+QEP 量化
        save_dir = run_onecomp_quantize(gamma)

        # 2. 加载 dequantized 模型
        cleanup_gpu()
        model = AutoModelForCausalLM.from_pretrained(
            save_dir, torch_dtype=torch.float16, trust_remote_code=True,
            device_map="cuda:0",
        )
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(save_dir, trust_remote_code=True)
        device = next(model.parameters()).device

        # 3. WikiText-2 PPL
        print(f"\nComputing WikiText-2 PPL for gamma={gamma}...")
        t0 = time.time()
        ppl = compute_wikitext_ppl(model, tokenizer, device)
        ppl_elapsed = time.time() - t0
        print(f"WikiText-2 PPL (gamma={gamma}): {ppl:.4f} ({ppl_elapsed:.0f}s)")

        with open(PPL_FILE, "a") as f:
            f.write(json.dumps({
                "model": MODEL_NAME, "method": method_label,
                "perplexity": round(ppl, 4),
            }) + "\n")

        # 4. GSM8K + ARC-Challenge
        task_scores = {}
        for task in TASKS:
            limit = TASK_LIMIT.get(task, None)
            print(f"\nEvaluating {task} (limit={limit}) for gamma={gamma}...")
            t0 = time.time()
            scores = run_eval_on_model(model, tokenizer, [task], limit=limit)
            task_scores.update(scores)
            elapsed = time.time() - t0
            sc = f"{scores[task]:.2f}%" if scores.get(task) is not None else "FAILED"
            print(f"  {task}: {sc} ({elapsed:.0f}s)")

        save_result(OUTPUT_FILE, MODEL_NAME, method_label, task_scores)

        all_results.append({"gamma": gamma, "ppl": ppl, "scores": task_scores})

        del model, tokenizer
        cleanup_gpu()

    # 汇总
    print(f"\n{'='*60}")
    print("exp05 Summary")
    print(f"{'='*60}")
    print(f"{'Gamma':<10} {'WikiText-2 PPL':<16} {'ARC-C':<12} {'GSM8K':<10}")
    print("-" * 52)
    for r in all_results:
        arc = str(r["scores"].get("arc_challenge", "N/A"))
        gsm = str(r["scores"].get("gsm8k", "N/A"))
        print(f"{r['gamma']:<10} {r['ppl']:<16.4f} {arc:<12} {gsm:<10}")

    # 与现有 fp16/gptq 对比
    print(f"\n--- 对比参考 ---")
    ppl_file = PPL_FILE
    if os.path.exists(ppl_file):
        with open(ppl_file) as f:
            for line in f:
                r = json.loads(line)
                if r["model"] == MODEL_NAME and r["method"] in ("fp16", "gptq"):
                    print(f"  {r['method']}: PPL={r['perplexity']}")

    task_file = OUTPUT_FILE
    if os.path.exists(task_file):
        with open(task_file) as f:
            for line in f:
                r = json.loads(line)
                if r["model"] == MODEL_NAME and r["method"] in ("fp16", "gptq"):
                    sc = r.get("scores", {})
                    print(f"  {r['method']}: ARC-C={sc.get('arc_challenge')}, GSM8K={sc.get('gsm8k')}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()

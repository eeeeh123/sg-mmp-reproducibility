"""exp06: Hadamard 旋转 + GPTQ-W4 vs 纯 GPTQ-W4.

对 Qwen2.5-0.5B 每层 Linear 权重做 Hadamard 双端旋转变换，
把离群值打散后跑标准 GPTQ。推理时反旋，零额外开销。

评测: GSM8K (300) + ARC-Challenge + WikiText-2 PPL
"""

import os, sys, json, time, gc
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, ".")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from ptq.data import get_calib_dataset
from ptq.eval import run_eval_on_model, cleanup_gpu, save_result
from ptq.quant.hadamard_gptq import quantize_model_hadamard_gptq, apply_hadamard_gptq_to_model_gpu

MODEL_PATH = "models/Qwen2.5-0.5B"
MODEL_NAME = "Qwen2.5-0.5B"
METHOD = "hadamard_gptq"
TASKS = ["arc_challenge", "gsm8k"]
OUTPUT_FILE = "results/task_results_full.jsonl"
PPL_FILE = "results/perplexity.jsonl"

torch.backends.cudnn.benchmark = False


@torch.no_grad()
def compute_wikitext_ppl(model, tokenizer, device) -> float:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    model.eval()
    nll_sum, n_tokens = 0.0, 0
    max_length = 2048
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
            logits = model(token_tensor).logits[:, :-1, :].contiguous()
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
    print(f"exp06: Hadamard + GPTQ-W4 @ {time.strftime('%H:%M:%S')}")

    # 1. 加载原始模型 + tokenizer
    cleanup_gpu()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True, device_map="cuda:0")
    model.eval()
    device = next(model.parameters()).device

    # 2. 校准数据
    print("Preparing calibration data (64 samples, 1024 max_len)...")
    calib_data = get_calib_dataset(tokenizer, n_samples=64, max_length=1024)
    print(f"  Calib shape: {calib_data.shape}")

    # 3. Hadamard + GPTQ 量化
    print("\nQuantizing with Hadamard + GPTQ-W4...")
    t0 = time.time()
    quant_state = quantize_model_hadamard_gptq(model, calib_data, bits=4, group_size=128)
    quant_time = time.time() - t0
    print(f"  Done in {quant_time:.0f}s ({len(quant_state)} layers)")

    # 4. 应用量化权重
    print("Applying quantized weights...")
    apply_hadamard_gptq_to_model_gpu(model, quant_state)
    del quant_state
    gc.collect()
    torch.cuda.empty_cache()

    # 5. PPL
    print("\nComputing WikiText-2 PPL...")
    t0 = time.time()
    ppl = compute_wikitext_ppl(model, tokenizer, device)
    print(f"  PPL: {ppl:.4f} ({time.time() - t0:.0f}s)")

    with open(PPL_FILE, "a") as f:
        f.write(json.dumps({"model": MODEL_NAME, "method": METHOD, "perplexity": round(ppl, 4)}) + "\n")

    # 6. ARC-C + GSM8K
    task_scores = {}
    for task in TASKS:
        limit = 300 if task == "gsm8k" else None
        print(f"\nEvaluating {task} (limit={limit})...")
        t0 = time.time()
        scores = run_eval_on_model(model, tokenizer, [task], limit=limit)
        task_scores.update(scores)
        elapsed = time.time() - t0
        sc = f"{scores[task]:.2f}%" if scores.get(task) is not None else "FAILED"
        print(f"  {task}: {sc} ({elapsed:.0f}s)")

    save_result(OUTPUT_FILE, MODEL_NAME, METHOD, task_scores)

    # 7. 对比
    print(f"\n{'='*60}")
    print("exp06 Summary")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'PPL':<10} {'ARC-C':<10} {'GSM8K':<10}")
    print("-" * 50)
    print(f"{'hadamard_gptq':<20} {ppl:<10.4f} {task_scores.get('arc_challenge','N/A'):<10} {task_scores.get('gsm8k','N/A'):<10}")

    # 读取现有结果做对比
    with open(OUTPUT_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r["model"] == MODEL_NAME and r["method"] in ("fp16", "gptq"):
                sc = r["scores"]
                print(f"{r['method']:<20} {'-':<10} {sc.get('arc_challenge','N/A'):<10} {sc.get('gsm8k','N/A'):<10}")

    with open(PPL_FILE) as f:
        for line in f:
            r = json.loads(line)
            if r["model"] == MODEL_NAME and r["method"] in ("fp16", "gptq"):
                print(f"{r['method']:<20} {r['perplexity']:<10}")

    print("Done.")
    del model, tokenizer
    cleanup_gpu()


if __name__ == "__main__":
    main()

"""WikiText-2 困惑度评测脚本。

支持 FP16 + 所有量化方法。

用法:
  python scripts/03_eval_perplexity.py --model "Qwen2.5-0.5B" --method fp16
  python scripts/03_eval_perplexity.py --model "Qwen2.5-0.5B" --method rtn
"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import sys
import torch
import json
import gc

sys.path.insert(0, ".")

from ptq.config import MODELS, QUANT_CONFIGS, EVAL_STRIDE, EVAL_MAX_LENGTH
from transformers import AutoModelForCausalLM, AutoTokenizer

STRIDE = EVAL_STRIDE
MAX_LENGTH = EVAL_MAX_LENGTH


def load_model_and_tokenizer(model_name: str, method: str):
    """加载模型和分词器，对量化方法应用量化包装。"""
    model_path = f"models/{model_name}"
    dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True
    )
    model.eval()

    if method == "fp16":
        pass
    elif method == "rtn":
        from ptq.quant.rtn import apply_rtn_to_model
        quant_state = torch.load(f"results/{model_name}_rtn.pt", map_location="cpu", weights_only=False)
        apply_rtn_to_model(model, quant_state)
    elif method == "gptq":
        from ptq.quant.gptq import apply_gptq_to_model
        state_path = f"results/{model_name}_gptq.pt"
        quant_state = torch.load(state_path, map_location="cpu", weights_only=False)
        apply_gptq_to_model(model, quant_state)
    elif method == "awq":
        from ptq.quant.awq import apply_awq_to_model
        state_path = f"results/{model_name}_awq.pt"
        scales_path = f"results/{model_name}_awq_scales.pt"
        quant_state = torch.load(state_path, map_location="cpu", weights_only=False)
        awq_scales = torch.load(scales_path, map_location="cpu", weights_only=False)
        apply_awq_to_model(model, quant_state, awq_scales)
    elif method == "smoothquant":
        from ptq.quant.smoothquant import apply_smoothquant_to_model
        scales = torch.load(f"results/{model_name}_smoothquant.pt", map_location="cpu", weights_only=False)
        apply_smoothquant_to_model(model, scales)
    elif method == "mixed_precision":
        from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu
        state_path = f"results/{model_name}_mixed_precision.pt"
        quant_state = torch.load(state_path, map_location="cpu", weights_only=False)
        apply_mixed_precision_to_model_gpu(model, quant_state)
    else:
        raise ValueError(f"Unknown method: {method}")

    return model, tokenizer


@torch.no_grad()
def compute_wikitext_ppl(model, tokenizer, device, max_length=MAX_LENGTH) -> float:
    """逐文档计算 WikiText-2 困惑度。

    标准做法：每个文档独立 tokenize，加 BOS token，
    用整个 (bos + tokens) 序列计算 NLL，取平均。
    文档超过 max_length 时用滑动窗口。
    """
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    model = model.to(device)
    model.eval()

    nll_sum = 0.0
    n_tokens = 0

    for doc in ds["text"]:
        if not doc or not doc.strip():
            continue
        tokens = tokenizer.encode(doc.strip(), add_special_tokens=False)
        if len(tokens) < 2:
            continue
        tokens = [tokenizer.bos_token_id] + tokens if tokenizer.bos_token_id is not None else tokens
        token_tensor = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)

        L = token_tensor.shape[1]
        if L <= max_length:
            logits = model(token_tensor).logits  # (1, L, vocab)
            logits = logits[:, :-1, :].contiguous()
            targets = token_tensor[:, 1:].contiguous()
            nll = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1), reduction="sum"
            )
            nll_sum += nll.item()
            n_tokens += L - 1
        else:
            # 长文档用滑动窗口
            stride = max_length // 2
            for begin in range(0, L, stride):
                end = min(begin + max_length, L)
                chunk = token_tensor[:, begin:end]
                logits = model(chunk).logits
                # 第一个窗口所有 token 都算，后续窗口只算新增的
                if begin == 0:
                    logits = logits[:, :-1, :].contiguous()
                    targets = chunk[:, 1:].contiguous()
                    trg_len = chunk.shape[1] - 1
                else:
                    n_prev = max_length // 2
                    logits = logits[:, n_prev:-1, :].contiguous()
                    targets = chunk[:, n_prev + 1 :].contiguous()
                    trg_len = chunk.shape[1] - n_prev - 1

                if trg_len > 0:
                    nll = torch.nn.functional.cross_entropy(
                        logits.view(-1, logits.shape[-1]), targets.view(-1), reduction="sum"
                    )
                    nll_sum += nll.item()
                    n_tokens += trg_len

                if end >= L:
                    break

    return torch.exp(torch.tensor(nll_sum / max(1, n_tokens))).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen2.5-0.5B")
    parser.add_argument("--method", type=str, default="fp16")
    parser.add_argument("--output", type=str, default="results/perplexity.jsonl")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== WikiText-2 PPL: {args.model} [{args.method}] ===")

    model, tokenizer = load_model_and_tokenizer(args.model, args.method)
    model = model.to(device)
    model.eval()

    print("Computing perplexity...")
    ppl = compute_wikitext_ppl(model, tokenizer, device)
    print(f"Perplexity: {ppl:.4f}")

    # 持久化
    result = {
        "model": args.model,
        "method": args.method,
        "perplexity": round(ppl, 4),
    }
    with open(args.output, "a") as f:
        f.write(json.dumps(result) + "\n")

    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

"""config_b 混合精度 + LoRA 微调实验。

加载 config_b 量化模型 → 加 LoRA → GSM8K 300条微调 → 全量评测。
"""

import os
import sys
sys.path.insert(0, ".")

import torch
import gc
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu
from ptq.eval import run_eval_on_model, cleanup_gpu

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
STATE_PATH = "results/Qwen2.5-0.5B_config_b.pt"
LORA_DIR = "results/Qwen2.5-0.5B_config_b_lora"
RESULTS_FILE = "results/task_results_full.jsonl"
PPL_FILE = "results/perplexity.jsonl"
os.makedirs(LORA_DIR, exist_ok=True)

# ============================================================
# 1. 加载模型 + 应用 config_b 量化
# ============================================================
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
    device_map="cuda:0", low_cpu_mem_usage=True,
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Applying config_b mixed precision...")
quant_state = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
apply_mixed_precision_to_model_gpu(model, quant_state)
del quant_state
gc.collect()
torch.cuda.empty_cache()

# ============================================================
# 2. 加载 GSM8K 训练数据（前 300 条）
# ============================================================
print("Loading GSM8K training data...")
ds = load_dataset("openai/gsm8k", "main", split="train")
ds = ds.select(range(300))

def format_gsm8k(example):
    prompt = f"Question: {example['question']}\nAnswer: {example['answer']}"
    return {"text": prompt}

ds = ds.map(format_gsm8k, remove_columns=["question", "answer"])
print(f"  Training samples: {len(ds)}")

# ============================================================
# 3. LoRA 配置
# ============================================================
print("Applying LoRA...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ============================================================
# 4. 训练
# ============================================================
print("Training...")
class SimpleCollator:
    """On-the-fly tokenization + padding，避免 DataCollator 嵌套问题。"""
    def __init__(self, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __call__(self, examples):
        texts = [ex["text"] for ex in examples]
        batch = self.tokenizer(texts, truncation=True, max_length=self.max_length,
                               padding=True, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        return batch

model.gradient_checkpointing_enable()

data_collator = SimpleCollator(tokenizer, max_length=256)

training_args = TrainingArguments(
    output_dir=LORA_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.0,
    warmup_ratio=0.0,
    lr_scheduler_type="constant",
    logging_steps=10,
    save_strategy="no",
    bf16=False,
    fp16=False,
    report_to="none",
    dataloader_drop_last=False,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

trainer.train()
print(f"  Training complete. Loss: {trainer.state.log_history[-1].get('loss', 'N/A')}")

# Save LoRA adapter
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
print(f"  LoRA saved to {LORA_DIR}")

# ============================================================
# 5. 评测
# ============================================================
print("\nEvaluating...")
model.eval()

# --- PPL ---
print("\n--- WikiText-2 PPL ---")
@torch.no_grad()
def _compute_ppl(model, tokenizer, device, max_length=2048):
    """WikiText-2 困惑度。"""
    from datasets import load_dataset as load_ds
    ds_ppl = load_ds("wikitext", "wikitext-2-raw-v1", split="test")
    model.eval()
    nll_sum = 0.0
    n_tokens = 0
    for doc in ds_ppl["text"]:
        if not doc or not doc.strip():
            continue
        tokens = tokenizer.encode(doc.strip(), add_special_tokens=False)
        if len(tokens) < 2:
            continue
        if tokenizer.bos_token_id is not None:
            tokens = [tokenizer.bos_token_id] + tokens
        t = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        L = t.shape[1]
        if L <= max_length:
            logits = model(t).logits[:, :-1].contiguous()
            targets = t[:, 1:].contiguous()
            nll = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]), targets.view(-1), reduction="sum")
            nll_sum += nll.item()
            n_tokens += L - 1
        else:
            stride = max_length // 2
            for begin in range(0, L, stride):
                end = min(begin + max_length, L)
                chunk = t[:, begin:end]
                logits = model(chunk).logits
                if begin == 0:
                    logits = logits[:, :-1].contiguous()
                    targets = chunk[:, 1:].contiguous()
                    trg_len = chunk.shape[1] - 1
                else:
                    n_prev = max_length // 2
                    logits = logits[:, n_prev:-1].contiguous()
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

device = next(model.parameters()).device
ppl = _compute_ppl(model, tokenizer, device)
print(f"  PPL={ppl:.4f}")

# Save PPL
ppl_result = {"model": MODEL_NAME, "method": "config_b_lora", "perplexity": round(ppl, 4)}
existing_ppl = []
if os.path.exists(PPL_FILE):
    with open(PPL_FILE) as f:
        for line in f:
            existing_ppl.append(json.loads(line))
found = False
for r in existing_ppl:
    if r["model"] == MODEL_NAME and r["method"] == "config_b_lora":
        r["perplexity"] = round(ppl, 4)
        found = True
if not found:
    existing_ppl.append(ppl_result)
with open(PPL_FILE, "w") as f:
    for r in existing_ppl:
        f.write(json.dumps(r) + "\n")

# --- Downstream tasks ---
from ptq.eval import save_result

TASKS = ["arc_challenge", "hellaswag", "mmlu", "gsm8k"]
LIMITS = {"arc_challenge": None, "hellaswag": None, "mmlu": None, "gsm8k": 300}

for task in TASKS:
    print(f"\n--- {task} ---")
    try:
        scores = run_eval_on_model(model, tokenizer, [task], batch_size=4,
                                   max_gen_toks=256, limit=LIMITS.get(task))
        score = scores.get(task)
        print(f"  {task}: {score:.2f}" if score else f"  {task}: FAILED")
        if score is not None:
            save_result(RESULTS_FILE, MODEL_NAME, "config_b_lora", {task: score})
    except Exception as e:
        print(f"  {task}: ERROR {e}")

cleanup_gpu()
print("\nDone!")

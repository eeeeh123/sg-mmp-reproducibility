"""config_b + LoRA v3: r=8, alpha=16, 400条, 2 epochs, q_proj+v_proj only.
GSM8K > 29 才跑全量。
"""
import os, sys
sys.path.insert(0, ".")
import torch, gc, json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu
from ptq.eval import run_eval_on_model, cleanup_gpu, save_result

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
STATE_PATH = "results/Qwen2.5-0.5B_config_b.pt"
LORA_DIR = "results/Qwen2.5-0.5B_config_b_lora_v3"
OUTPUT_FILE = "results/task_results_full.jsonl"
os.makedirs(LORA_DIR, exist_ok=True)

class C:
    def __init__(self, tok, ml=256):
        self.tok = tok; self.ml = ml
    def __call__(self, ex):
        texts = [e["text"] for e in ex]
        b = self.tok(texts, truncation=True, max_length=self.ml, padding=True, return_tensors="pt")
        b["labels"] = b["input_ids"].clone()
        return b

# 1. load + quantize
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
    device_map="cuda:0", low_cpu_mem_usage=True)
model.eval()
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

print("Applying config_b...")
qs = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
apply_mixed_precision_to_model_gpu(model, qs)
del qs; gc.collect(); torch.cuda.empty_cache()

# 2. data: GSM8K train 前 400
print("Loading GSM8K train...")
ds = load_dataset("openai/gsm8k", "main", split="train").select(range(400))
def fmt(ex):
    return {"text": f"Question: {ex['question']}\nAnswer: {ex['answer']}"}
ds = ds.map(fmt, remove_columns=["question", "answer"])
print(f"  {len(ds)} samples")

# 3. LoRA: r=8, alpha=16, q_proj+v_proj
print("Applying LoRA (r=8, alpha=16)...")
model.gradient_checkpointing_enable()
lora = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                  lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

# 4. train: 2 epochs, bs=4
print("Training (2 epochs, lr=1e-4)...")
ta = TrainingArguments(output_dir=LORA_DIR, num_train_epochs=2,
    per_device_train_batch_size=4, gradient_accumulation_steps=1,
    learning_rate=1e-4, weight_decay=0.0, warmup_ratio=0.0,
    lr_scheduler_type="constant", logging_steps=20, save_strategy="no",
    fp16=False, bf16=False, report_to="none",
    dataloader_drop_last=False, remove_unused_columns=False)
trainer = Trainer(model=model, args=ta, train_dataset=ds, data_collator=C(tok))
trainer.train()
model.save_pretrained(LORA_DIR)
print(f"  Saved: {LORA_DIR}")

# 5. GSM8K
model.eval()
print("\n--- GSM8K ---")
scores = run_eval_on_model(model, tok, ["gsm8k"], batch_size=4, max_gen_toks=256, limit=300)
gsm = scores.get("gsm8k")
print(f"  GSM8K={gsm:.2f}" if gsm else "  FAILED")
if gsm is not None:
    save_result(OUTPUT_FILE, MODEL_NAME, "config_b_lora_v3", {"gsm8k": gsm})

TARGET = 29.0
if gsm is not None and gsm > TARGET:
    print(f"\nGSM8K {gsm:.2f} > {TARGET}, running full benchmark!")
    for task in ["arc_challenge", "hellaswag", "mmlu"]:
        print(f"\n--- {task} ---")
        try:
            sc = run_eval_on_model(model, tok, [task], batch_size=4, max_gen_toks=256)
            s = sc.get(task)
            print(f"  {task}: {s:.2f}" if s else "  FAILED")
            if s is not None:
                save_result(OUTPUT_FILE, MODEL_NAME, "config_b_lora_v3", {task: s})
        except Exception as e:
            print(f"  ERROR: {e}")
else:
    print(f"\nGSM8K {gsm:.2f} <= {TARGET}, skip full benchmark.")

cleanup_gpu()
print("\nDone!")

"""实验6: Failure-Triggered LoRA — 仅对关键层加 LoRA 微调。

三个变体:
  6A: LoRA on layer 11 only (exp07 delta=+3.0, 最敏感)
  6B: LoRA on layers {1, 6, 11} (exp07 top-3 正向层)
  6C: LoRA on 实验5 最早发散层 (需实验5完成后填充)

在 config_b 量化模型基础上，仅对指定层的 attention 投影 (q/k/v/o) 加 LoRA。
r=4, alpha=8, GSM8K 300 条训练, 2 epochs, GSM8K > 27.33 筛选后全量评测。
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, ".")
import torch, gc, json, argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType

from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu
from ptq.eval import run_eval_on_model, cleanup_gpu, save_result

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
STATE_PATH = "results/Qwen2.5-0.5B_config_b.pt"
OUTPUT_FILE = "results/task_results_full.jsonl"
BASELINE_THRESHOLD = 27.33
ATTN_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_TASKS = ["arc_challenge", "hellaswag", "mmlu", "gsm8k"]

# 关键层定义 (from exp07_layer_replacement)
VARIANT_LAYERS = {
    "6a": [11],
    "6b": [1, 6, 11],
    "6c": None,  # filled from exp5 after it runs
}

class Collator:
    def __init__(self, tok, max_len=256):
        self.tok = tok; self.max_len = max_len
    def __call__(self, examples):
        texts = [e["text"] for e in examples]
        batch = self.tok(texts, truncation=True, max_length=self.max_len,
                         padding=True, return_tensors="pt")
        batch["labels"] = batch["input_ids"].clone()
        return batch


def build_target_modules(model, target_layers, module_types):
    """收集 base model 中指定层、指定类型的 Linear 全名。

    PEFT 用 endswith 匹配 target_modules，全名确保只命中指定层。
    """
    targets = []
    for name, mod in model.named_modules():
        import torch.nn as nn
        if not isinstance(mod, nn.Linear):
            continue
        for layer_idx in target_layers:
            layer_prefix = f"model.layers.{layer_idx}."
            if layer_prefix in name:
                for mtype in module_types:
                    if name.endswith(mtype) and name.endswith(mtype):  # redundant but safe
                        pass
                # Check: name contains layer prefix AND ends with one of module_types
                if name.startswith("model.layers."):
                    suffix = name.split(".")[-1]  # last part like q_proj
                    if suffix in module_types and f".{layer_idx}." in name:
                        targets.append(name)
    return sorted(set(targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default="6a", choices=["6a", "6b", "6c"])
    parser.add_argument("--early_layers", type=str, default=None,
                        help="Comma-separated layer indices for 6C, e.g. 11,6,2")
    args = parser.parse_args()

    variant = args.variant
    if variant == "6c":
        if args.early_layers:
            target_layers = [int(x) for x in args.early_layers.split(",")]
        else:
            print("ERROR: 6C requires --early_layers. Run exp14 first.")
            print("Usage: python experiments/exp15_failure_lora/run.py --variant 6c --early_layers 11,6,2")
            return
    else:
        target_layers = VARIANT_LAYERS[variant]

    method_name = f"config_b_failure_lora_{variant}"
    lora_dir = f"results/Qwen2.5-0.5B_{method_name}"
    os.makedirs(lora_dir, exist_ok=True)

    print(f"=== Experiment 6{variant.upper()}: LoRA on layers {target_layers} ===")

    # ---- 1. Load model + apply config_b ----
    print("Loading model...")
    cleanup_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Applying config_b mixed precision...")
    qs = torch.load(STATE_PATH, map_location="cpu", weights_only=False)
    apply_mixed_precision_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()

    # ---- 2. Build target modules for PEFT ----
    target_mods = build_target_modules(model, target_layers, ATTN_TYPES)
    if not target_mods:
        print(f"ERROR: no target modules found for layers {target_layers}, types {ATTN_TYPES}")
        return
    print(f"LoRA targets ({len(target_mods)} modules):")
    for t in target_mods:
        print(f"  {t}")

    # ---- 3. Load GSM8K training data ----
    print("Loading GSM8K training data (300 examples)...")
    ds = load_dataset("openai/gsm8k", "main", split="train").select(range(300))
    def fmt(ex):
        return {"text": f"Question: {ex['question']}\nAnswer: {ex['answer']}"}
    ds = ds.map(fmt, remove_columns=["question", "answer"])

    # ---- 4. Apply LoRA ----
    print("Applying LoRA (r=4, alpha=8)...")
    model.gradient_checkpointing_enable()
    lora = LoraConfig(
        r=4, lora_alpha=8,
        target_modules=target_mods,  # exact full module names
        lora_dropout=0.0, bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- 5. Train ----
    print("Training (2 epochs, lr=1e-4, bs=4)...")
    ta = TrainingArguments(
        output_dir=lora_dir, num_train_epochs=2,
        per_device_train_batch_size=4, gradient_accumulation_steps=1,
        learning_rate=1e-4, weight_decay=0.0, warmup_ratio=0.0,
        lr_scheduler_type="constant", logging_steps=10, save_strategy="no",
        fp16=False, bf16=False, report_to="none",
        dataloader_drop_last=False, remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=ta, train_dataset=ds, data_collator=Collator(tok))
    trainer.train()
    model.save_pretrained(lora_dir)
    print(f"  LoRA saved: {lora_dir}")

    # ---- 6. GSM8K screening ----
    print("\n--- GSM8K screening ---")
    model.eval()
    scores = run_eval_on_model(model, tok, ["gsm8k"], batch_size=4, max_gen_toks=256, limit=300)
    gsm = scores.get("gsm8k")
    print(f"  GSM8K={gsm:.2f}" if gsm else "  GSM8K: FAILED")

    if gsm is not None:
        save_result(OUTPUT_FILE, MODEL_NAME, method_name, {"gsm8k": gsm})

    # ---- 7. Full benchmark if GSM8K passes threshold ----
    if gsm is not None and gsm > BASELINE_THRESHOLD:
        print(f"\nGSM8K {gsm:.2f} > {BASELINE_THRESHOLD}, running full benchmark!")
        for task in ["arc_challenge", "hellaswag", "mmlu"]:
            print(f"\n--- {task} ---")
            try:
                sc = run_eval_on_model(model, tok, [task], batch_size=4, max_gen_toks=256)
                s = sc.get(task)
                print(f"  {task}: {s:.2f}" if s else f"  {task}: FAILED")
                if s is not None:
                    save_result(OUTPUT_FILE, MODEL_NAME, method_name, {task: s})
            except Exception as e:
                print(f"  ERROR: {e}")
    else:
        print(f"\nGSM8K {gsm:.2f} <= {BASELINE_THRESHOLD}, skip full benchmark.")

    # ---- 8. Summary comparison ----
    print(f"\n{'='*60}")
    print(f"Experiment 6{variant.upper()} Summary")
    print(f"  Target layers: {target_layers}")
    print(f"  Config: config_b + LoRA r=4, alpha=8 on {len(target_mods)} modules")
    if gsm is not None:
        print(f"  GSM8K: {gsm:.2f} (config_b baseline: {BASELINE_THRESHOLD})")

    # Compare with previous LoRA results if available
    comparisons = {
        "config_b": None, "config_b_lora": None, "config_b_lora_v2": None,
    }
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                r = json.loads(line)
                if r["model"] == MODEL_NAME and r["method"] in comparisons:
                    comparisons[r["method"]] = r["scores"].get("gsm8k")
    for m, s in comparisons.items():
        if s is not None:
            print(f"  vs {m}: GSM8K={s:.2f}")

    cleanup_gpu()
    print("\nDone.")


if __name__ == "__main__":
    main()

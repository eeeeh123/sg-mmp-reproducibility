"""实验4：错误类型分类。

对 exp14 同批 50 道 GSM8K 题，生成 GPTQ-W4、config_b、FP16 的完整输出，
然后分类错误类型：算术错误 / 逻辑错误 / 提取错误 / 格式错误 / 未完成。

Usage:
  python run.py generate     # GPU：生成三个模型的完整输出
  python run.py classify     # CPU：分类错误类型并输出表格
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, ".")

STEP = sys.argv[1] if len(sys.argv) > 1 else "generate"

import torch, gc, json, re
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from ptq.eval import cleanup_gpu
from ptq.quant.gptq import apply_gptq_to_model_gpu
from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
GPTQ_STATE = "results/Qwen2.5-0.5B_gptq_compact.pt"
CONFIG_B_STATE = "results/Qwen2.5-0.5B_config_b.pt"
OUT_DIR = "experiments/exp20_error_types/results"
N_EXAMPLES = 50
MAX_NEW_TOKENS = 256

os.makedirs(OUT_DIR, exist_ok=True)


def truncate_at_hallucination(text):
    """截断幻觉追问：模型可能在答完后编造新题目。取第一个 Q&A 段。"""
    for pat in ["\n[Question", "\nQuestion:", "\nQ:", "\n\nQuestion"]:
        idx = text.find(pat)
        if idx > 0:
            return text[:idx]
    return text


def extract_final_answer(text):
    """从生成文本中提取最终数字答案。"""
    text = truncate_at_hallucination(text)
    # 优先 #### <number>（GSM8K 官方格式）
    m = re.search(r"####\s*(-?[\d,./]+)", text)
    if m:
        s = m.group(1).replace(",", "").replace(" ", "")
        try:
            return int(float(s))
        except ValueError:
            pass
    # 其次 "answer is X" 或 "The answer is X"
    m = re.search(r"(?:answer\s+is|answer\s*[:=])\s*(-?[\d,./]+)", text, re.IGNORECASE)
    if m:
        s = m.group(1).replace(",", "").replace(" ", "")
        try:
            return int(float(s))
        except ValueError:
            pass
    # fallback: 取最后一个数字
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    if numbers:
        return float(numbers[-1])
    return None


def generate_for_variant(model, tokenizer, prompts, answers, variant_name):
    """生成一批题目的完整输出。"""
    out_file = os.path.join(OUT_DIR, f"generations_{variant_name}.jsonl")
    if os.path.exists(out_file):
        print(f"[{variant_name}] Generations file exists, skip")
        with open(out_file) as f:
            return [json.loads(line) for line in f]

    generations = []
    for idx, (prompt, answer) in enumerate(zip(prompts, answers)):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        # 编码停止序列，防止模型编造新题目
        stop_strings = ["\nQuestion:", "\n[Question", "\nQ:", "<|im_end|>"]
        stop_token_ids = []
        for s in stop_strings:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                stop_token_ids.append(ids[0])
        eos_ids = list(set([tokenizer.eos_token_id] + stop_token_ids))
        with torch.no_grad():
            gen_out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False, temperature=None,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=eos_ids)
        gen_text = tokenizer.decode(
            gen_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        generations.append({
            "example_id": idx, "prompt": prompt,
            "answer_reference": answer, "generation": gen_text,
        })
        if (idx + 1) % 10 == 0:
            print(f"  [{variant_name}] {idx + 1}/{len(prompts)}")

    with open(out_file, "w") as f:
        for g in generations:
            f.write(json.dumps(g) + "\n")
    print(f"[{variant_name}] Saved {len(generations)} generations to {out_file}")
    return generations


def step_generate():
    """生成 FP16、GPTQ-W4、config_b 三种模型的完整输出。"""
    # 加载 GSM8K 50 题（exp14 同批）
    print(f"Loading GSM8K test examples (first {N_EXAMPLES})...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    items = list(ds.select(range(N_EXAMPLES)))
    prompts = [f"Question: {ex['question']}\nAnswer:" for ex in items]
    answers = [ex["answer"] for ex in items]

    # ---- FP16 ----
    print("\n=== FP16 Generation ===")
    cleanup_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    generate_for_variant(model, tokenizer, prompts, answers, "fp16")
    del model, tokenizer
    cleanup_gpu()

    # ---- GPTQ-W4 ----
    print("\n=== GPTQ-W4 Generation ===")
    cleanup_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qs = torch.load(GPTQ_STATE, map_location="cpu", weights_only=False)
    apply_gptq_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()

    generate_for_variant(model, tokenizer, prompts, answers, "gptq")
    del model, tokenizer
    cleanup_gpu()

    # ---- config_b ----
    print("\n=== config_b Generation ===")
    cleanup_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    qs = torch.load(CONFIG_B_STATE, map_location="cpu", weights_only=False)
    apply_mixed_precision_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()

    generate_for_variant(model, tokenizer, prompts, answers, "config_b")
    del model, tokenizer
    cleanup_gpu()
    print("\nAll generations done.")


def classify_errors(gptq_gen, cb_gen, ref_answer):
    """判断两个模型是否答对，并做初步分类。"""
    gptq_ans = extract_final_answer(gptq_gen)
    cb_ans = extract_final_answer(cb_gen)
    ref_ans = extract_final_answer(ref_answer)

    if ref_ans is None:
        return "undetermined"

    gptq_correct = (gptq_ans is not None and abs(gptq_ans - ref_ans) < 1e-6)
    cb_correct = (cb_ans is not None and abs(cb_ans - ref_ans) < 1e-6)

    if gptq_correct and cb_correct:
        return "both_correct"
    elif not gptq_correct and not cb_correct:
        return "both_wrong"
    elif not gptq_correct and cb_correct:
        return "gptq_wrong_configb_correct"  # config_b 修复
    elif gptq_correct and not cb_correct:
        return "gptq_correct_configb_wrong"  # 回归
    return "undetermined"


def step_classify():
    """读取生成文件，分类错误类型。"""
    generations = {}
    for variant in ["gptq", "config_b", "fp16"]:
        gen_file = os.path.join(OUT_DIR, f"generations_{variant}.jsonl")
        if not os.path.exists(gen_file):
            print(f"ERROR: {gen_file} not found. Run 'generate' first.")
            return
        with open(gen_file) as f:
            generations[variant] = [json.loads(line) for line in f]

    counts = {
        "both_correct": 0,
        "both_wrong": 0,
        "gptq_wrong_configb_correct": 0,
        "gptq_correct_configb_wrong": 0,
        "undetermined": 0,
    }
    breakdown = []

    for i in range(N_EXAMPLES):
        gptq_gen = generations["gptq"][i]["generation"]
        cb_gen = generations["config_b"][i]["generation"]
        ref_answer = generations["gptq"][i]["answer_reference"]

        category = classify_errors(gptq_gen, cb_gen, ref_answer)
        counts[category] += 1
        breakdown.append({
            "example_id": i,
            "category": category,
            "ref_answer": ref_answer,
            "gptq_answer": extract_final_answer(gptq_gen),
            "config_b_answer": extract_final_answer(cb_gen),
        })

    # 输出分类表
    total = sum(counts.values())
    print(f"\n{'='*65}")
    print(f"Error Type Classification ({N_EXAMPLES} GSM8K examples)")
    print(f"{'Category':<40} {'Count':<10} {'%':<10}")
    print("-" * 65)

    labels = {
        "both_correct": "Both correct",
        "both_wrong": "Both wrong (shared failure)",
        "gptq_wrong_configb_correct": "GPTQ-W4 wrong, config_b CORRECT (FIXED)",
        "gptq_correct_configb_wrong": "GPTQ-W4 correct, config_b WRONG (REGRESSION)",
        "undetermined": "Undetermined",
    }
    for cat in ["both_correct", "both_wrong", "gptq_wrong_configb_correct",
                 "gptq_correct_configb_wrong", "undetermined"]:
        c = counts.get(cat, 0)
        pct = 100.0 * c / total if total > 0 else 0
        print(f"{labels[cat]:<40} {c:<10} {pct:<10.1f}")

    # 关键指标
    fixed = counts["gptq_wrong_configb_correct"]
    regressed = counts["gptq_correct_configb_wrong"]
    net = fixed - regressed
    print(f"\n  Net improvement: {net} examples ({net}/{N_EXAMPLES})")
    print(f"  Fix rate (among fixable): {fixed}/{fixed + counts['both_wrong']}")

    # 保存
    with open(os.path.join(OUT_DIR, "error_breakdown.json"), "w") as f:
        json.dump({"counts": counts, "per_example": breakdown}, f, indent=2)
    print(f"\nSaved error_breakdown.json")


if __name__ == "__main__":
    if STEP == "generate":
        step_generate()
    elif STEP == "classify":
        step_classify()
    else:
        print("Usage: python run.py generate  |  python run.py classify")

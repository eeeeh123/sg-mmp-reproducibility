"""实验5: First Divergent Step — GSM8K 生成中量化模型在哪一步开始发散。

Teacher-forcing 对比: FP16 模型生成 token 序列，同一序列分别通过 FP16 和
GPTQ-W4 模型前向传播，逐层逐步计算 cosine similarity，找到首次发散点。
"""
import os, sys
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, ".")
import torch, gc, time, json, argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptq.eval import cleanup_gpu
from ptq.quant.gptq import apply_gptq_to_model_gpu

MODEL_NAME = "Qwen2.5-0.5B"
MODEL_PATH = "models/Qwen2.5-0.5B"
GPTQ_STATE = "results/Qwen2.5-0.5B_gptq_compact.pt"
OUT_DIR = "experiments/exp14_first_divergent_step/results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- helpers ----
def backup_all_weights(model):
    """Save all Linear + lm_head + embed weights to CPU."""
    d = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            d[name] = module.weight.data.clone().cpu()
    d["__lm_head__"] = model.lm_head.weight.data.clone().cpu()
    d["__embed__"] = model.model.embed_tokens.weight.data.clone().cpu()
    return d

def restore_all_weights(model, saved):
    """Restore all weights from CPU backup."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name in saved:
            module.weight.data.copy_(saved[name].to(module.weight.device, module.weight.dtype))
    if "__lm_head__" in saved:
        model.lm_head.weight.data.copy_(saved["__lm_head__"].to(model.lm_head.weight.device, model.lm_head.weight.dtype))
    if "__embed__" in saved:
        model.model.embed_tokens.weight.data.copy_(saved["__embed__"].to(model.model.embed_tokens.weight.device, model.model.embed_tokens.weight.dtype))

def find_first_divergence(sim_matrix, threshold, prompt_len):
    """sim_matrix: (n_layers, seq_len) cosine similarities.
    Returns (step, layer) of first position where any layer drops below threshold,
    or None if never diverges. Only checks positions >= prompt_len (generated tokens).
    """
    n_layers, seq_len = sim_matrix.shape
    for pos in range(prompt_len, seq_len):
        col = sim_matrix[:, pos]
        min_idx = torch.argmin(col).item()
        if col[min_idx] < threshold:
            return int(pos - prompt_len), int(min_idx)
    return None

# ============================================================
# Config
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--thresholds", type=str, default="0.99,0.95,0.90")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    # ---- 1. Load FP16 model + save weights ----
    print("Loading FP16 model...")
    cleanup_gpu()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True,
        device_map="cuda:0", low_cpu_mem_usage=True)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Backing up FP16 weights...")
    fp16_weights = backup_all_weights(model)
    n_linears = sum(1 for k in fp16_weights if not k.startswith("__"))
    print(f"  Saved {n_linears} Linear layers + lm_head + embedding")

    # ---- 2. Apply GPTQ-W4 ----
    print("Applying GPTQ-W4...")
    qs = torch.load(GPTQ_STATE, map_location="cpu", weights_only=False)
    apply_gptq_to_model_gpu(model, qs)
    del qs; gc.collect(); torch.cuda.empty_cache()

    # Save W4 weights
    w4_weights = backup_all_weights(model)
    print(f"  W4 weights backed up, {len(w4_weights)} entries")

    # ---- 3. Load GSM8K prompts ----
    print(f"Loading GSM8K test examples (first {args.limit})...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    prompts = [f"Question: {ex['question']}\nAnswer:" for ex in ds.select(range(args.limit))]

    # ---- 4. Per-example analysis ----
    all_results = []
    all_sim_matrices = []  # each: (24, seq_len) but variable seq_len
    max_gen_len = 0

    for idx, prompt in enumerate(prompts):
        print(f"\n--- Example {idx+1}/{args.limit} ---")
        # 4a. Restore FP16
        restore_all_weights(model, fp16_weights)

        # 4b. FP16 generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]
        with torch.no_grad():
            gen_out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                      do_sample=False, temperature=None,
                                      pad_token_id=tokenizer.eos_token_id)
        full_ids = gen_out  # (1, prompt_len + generated_len)
        total_len = full_ids.shape[1]
        gen_len = total_len - prompt_len
        if gen_len < 5:
            print(f"  SKIP: only {gen_len} tokens generated")
            continue

        # 4c. FP16 forward with hidden states
        with torch.no_grad():
            fp16_out = model(input_ids=full_ids, output_hidden_states=True)
        fp16_hidden = torch.stack([h.squeeze(0).cpu() for h in fp16_out.hidden_states[1:]])  # skip embedding, (24, seq, 896)

        # 4d. Swap to W4
        restore_all_weights(model, w4_weights)

        # 4e. W4 forward with hidden states
        with torch.no_grad():
            w4_out = model(input_ids=full_ids, output_hidden_states=True)
        w4_hidden = torch.stack([h.squeeze(0).cpu() for h in w4_out.hidden_states[1:]])  # (24, seq, 896)

        # 4f. Compute cosine similarity matrix
        # fp16_hidden: (24, seq, 896), w4_hidden: (24, seq, 896)
        sim = F.cosine_similarity(fp16_hidden.float(), w4_hidden.float(), dim=-1)  # (24, seq)
        all_sim_matrices.append(sim)
        max_gen_len = max(max_gen_len, gen_len)

        # 4g. Find first divergence
        result = {"example_id": idx, "prompt_len": prompt_len, "generated_len": gen_len}
        for thresh in thresholds:
            found = find_first_divergence(sim, thresh, prompt_len)
            if found is not None:
                step, layer = found
                result[f"first_div_{str(thresh).replace('.', '')}"] = {"step": step, "layer": layer}
            else:
                result[f"first_div_{str(thresh).replace('.', '')}"] = None
        all_results.append(result)
        print(f"  gen_len={gen_len}, thresholds: " +
              " ".join(f"t{t}={result.get(f'first_div_{str(t).replace('.', '')}', 'none')}" for t in thresholds))

        # Restore FP16 for next example
        restore_all_weights(model, fp16_weights)
        gc.collect(); torch.cuda.empty_cache()

        # Incremental save
        with open(os.path.join(OUT_DIR, "per_example.jsonl"), "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")

    # ---- 5. Aggregate results ----
    print(f"\n{'='*60}")
    print(f"Aggregating across {len(all_results)} examples...")

    summary_rows = []
    for thresh in thresholds:
        key = f"first_div_{str(thresh).replace('.', '')}"
        found = [r for r in all_results if r.get(key) is not None]
        steps = [r[key]["step"] for r in found]
        layers = [r[key]["layer"] for r in found]

        row = {
            "threshold": thresh,
            "divergence_rate": len(found) / max(1, len(all_results)),
            "mean_first_step": np.mean(steps) if steps else None,
            "median_first_step": np.median(steps) if steps else None,
            "most_common_layer": max(set(layers), key=layers.count) if layers else None,
        }
        have_data = found and row['mean_first_step'] is not None
        if have_data:
            print(f"  threshold={thresh}: {len(found)}/{len(all_results)} diverged, "
                  f"mean_step={row['mean_first_step']:.1f}")
        else:
            print(f"  threshold={thresh}: no divergence detected")
        summary_rows.append(row)

    # Per-layer stats
    per_layer_rows = []
    for layer_idx in range(24):
        layer_row = {"layer": layer_idx}
        for thresh in thresholds:
            key = f"first_div_{str(thresh).replace('.', '')}"
            count = sum(1 for r in all_results if r.get(key) and r[key]["layer"] == layer_idx)
            layer_row[f"count_t{str(thresh).replace('.','')}"] = count
        per_layer_rows.append(layer_row)

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        import csv
        w = csv.DictWriter(f, fieldnames=["threshold", "divergence_rate", "mean_first_step",
                                           "median_first_step", "most_common_layer"])
        w.writeheader()
        w.writerows(summary_rows)

    with open(os.path.join(OUT_DIR, "per_layer_summary.csv"), "w", newline="") as f:
        import csv
        fieldnames = ["layer"] + [f"count_t{str(t).replace('.','')}" for t in thresholds]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_layer_rows)

    # ---- 6. Heatmap ----
    if all_sim_matrices:
        # Align all matrices to max_gen_len, pad with NaN for shorter sequences
        padded = []
        for sim in all_sim_matrices:
            gen_len_sim = sim.shape[1] - all_results[len(padded)]["prompt_len"]
            # Only take generated positions
            p_len = all_results[len(padded)]["prompt_len"]
            gen_sim = sim[:, p_len:]  # (24, gen_len)
            if gen_sim.shape[1] < max_gen_len:
                pad = torch.full((24, max_gen_len - gen_sim.shape[1]), float("nan"))
                gen_sim = torch.cat([gen_sim, pad], dim=1)
            padded.append(gen_sim)
            if len(padded) >= len(all_results):
                break

        if padded:
            stacked = torch.stack(padded, dim=0)  # (n_examples, 24, max_gen_len)
            avg_sim = torch.nanmean(stacked, dim=0).numpy()  # (24, max_gen_len)

            fig, ax = plt.subplots(figsize=(16, 8))
            im = ax.imshow(avg_sim, aspect="auto", cmap="RdYlGn", vmin=0.9, vmax=1.0,
                           origin="lower", interpolation="nearest")
            ax.set_xlabel("Generation Step (token position)")
            ax.set_ylabel("Layer Index")
            ax.set_title(f"Mean Cosine Similarity: FP16 vs GPTQ-W4 Hidden States\n"
                         f"({len(all_results)} GSM8K examples, Qwen2.5-0.5B)")
            plt.colorbar(im, ax=ax, label="Cosine Similarity")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT_DIR, "heatmap.png"), dpi=150)
            plt.close(fig)
            print(f"Heatmap saved (avg across {len(all_results)} examples, "
                  f"max gen_len={max_gen_len})")

            # Also save raw avg_sim as numpy
            np.save(os.path.join(OUT_DIR, "cosine_similarity.npy"), avg_sim)
            print("Raw cosine similarity matrix saved as .npy")

    # ---- 7. Summary ----
    print(f"\n{'='*60}")
    print("Top-5 earliest-diverging layers (threshold=0.99):")
    key_099 = "first_div_099"
    layer_counts = {}
    for r in all_results:
        if r.get(key_099):
            l = r[key_099]["layer"]
            layer_counts[l] = layer_counts.get(l, 0) + 1
    sorted_layers = sorted(layer_counts.items(), key=lambda x: -x[1])
    for layer, count in sorted_layers[:5]:
        print(f"  Layer {layer}: {count} times (first-to-diverge)")

    print("\nDone.")
    cleanup_gpu()

if __name__ == "__main__":
    main()

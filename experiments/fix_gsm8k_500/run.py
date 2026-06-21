"""GSM8K-500 robustness and paired-statistics experiments.

This script adds a stricter evaluation layer for the paper:

1. A fixed deterministic 500-example GSM8K test subset shared by all methods.
2. Core FP16/GPTQ/SG-MMP evaluation for the three SLMs.
3. Per-example correctness logs for paired McNemar and bootstrap analysis.
4. Optional Qwen-0.5B module ablations for SG-MMP design validation.

Outputs are written under experiments/fix_gsm8k_500/results/ and are safe to
resume: an existing sample file is not recomputed unless --force is passed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, ".")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") in (None, "", "expandable_segments:True"):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import torch
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptq.eval import cleanup_gpu
from ptq.quant.gptq import apply_gptq_to_model_gpu
from ptq.quant.mixed_precision import (
    ATTN_PROJ,
    FFN_PROJ,
    apply_mixed_precision_to_model_gpu,
    parse_layer_num,
    quantize_model_mixed_precision,
)
from ptq.data import get_calib_dataset


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "experiments" / "fix_gsm8k_500" / "results"
SAMPLE_DIR = OUT_DIR / "samples"
STATE_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

TASK = "gsm8k"
N_EXAMPLES = 500
INDEX_SEED = 20260615

MODELS = {
    "qwen05": {
        "name": "Qwen2.5-0.5B",
        "path": MODEL_DIR / "Qwen2.5-0.5B",
        "sg_method": "sg_mmp",
        "sg_state": STATE_DIR / "Qwen2.5-0.5B_config_b.pt",
    },
    "qwen15": {
        "name": "Qwen2.5-1.5B",
        "path": MODEL_DIR / "Qwen2.5-1.5B",
        "sg_method": "sg_mmp_2a",
        "sg_state": STATE_DIR / "Qwen2.5-1.5B_config_b_2a.pt",
    },
    "smollm": {
        "name": "SmolLM-1.7B",
        "path": MODEL_DIR / "SmolLM-1.7B",
        "sg_method": "sg_mmp",
        "sg_state": STATE_DIR / "SmolLM-1.7B_config_b_compact.pt",
    },
}

METHODS = {
    "fp16": {"kind": "fp16", "paper": "FP16"},
    "gptq": {"kind": "gptq", "paper": "GPTQ-W4"},
    "sg": {"kind": "mixed", "paper": "SG-MMP"},
}

QWEN05_ABLATIONS = {
    "abl_only_sensitive_w8": {
        "paper": "Only sensitive layers W8",
        "state": STATE_DIR / "Qwen2.5-0.5B_abl_only_sensitive_w8.pt",
        "protected": {2, 6, 7, 11},
        "mode": "only_sensitive",
    },
    "abl_only_qkv_w8": {
        "paper": "Only q/k/v W8",
        "state": STATE_DIR / "Qwen2.5-0.5B_abl_only_qkv_w8.pt",
        "protected": set(),
        "mode": "only_qkv",
    },
    "abl_sensitive_plus_ffn_w8": {
        "paper": "Sensitive layers + FFN W8",
        "state": STATE_DIR / "Qwen2.5-0.5B_abl_sensitive_plus_ffn_w8.pt",
        "protected": {2, 6, 7, 11},
        "mode": "sensitive_plus_ffn",
    },
}


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def jsonl_write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def make_indices(n_examples: int = N_EXAMPLES, seed: int = INDEX_SEED) -> list[int]:
    # GSM8K test split has 1319 examples. Keeping this explicit makes the subset
    # reproducible without importing task internals.
    all_indices = list(range(1319))
    rng = random.Random(seed)
    rng.shuffle(all_indices)
    return sorted(all_indices[:n_examples])


def ensure_indices(n_examples: int = N_EXAMPLES) -> list[int]:
    path = OUT_DIR / f"gsm8k_{n_examples}_indices.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["indices"]
    indices = make_indices(n_examples=n_examples)
    json_dump(
        path,
        {
            "task": TASK,
            "split": "test",
            "n": len(indices),
            "seed": INDEX_SEED,
            "selection": "sorted(random.Random(seed).shuffle(range(1319))[:n])",
            "indices": indices,
        },
    )
    return indices


def method_state_for(model_key: str, method_key: str) -> Path | None:
    model = MODELS[model_key]
    model_name = model["name"]
    if method_key == "gptq":
        compact = STATE_DIR / f"{model_name}_gptq_compact.pt"
        full = STATE_DIR / f"{model_name}_gptq.pt"
        if compact.exists():
            return compact
        if model_key == "smollm" and full.exists():
            raise RuntimeError(
                f"Missing compact GPTQ state for SmolLM: {compact}. "
                f"Refusing to load legacy state {full}; regenerate compact GPTQ first."
            )
        return full
    if method_key == "sg":
        return model["sg_state"]
    if method_key in QWEN05_ABLATIONS:
        return QWEN05_ABLATIONS[method_key]["state"]
    return None


def load_model(model_key: str, method_key: str):
    model_info = MODELS[model_key]
    print(f"Loading {model_info['name']} [{method_key}]", flush=True)

    state_path = None
    if method_key != "fp16":
        state_path = method_state_for(model_key, method_key)
        if state_path is None or not state_path.exists():
            raise FileNotFoundError(f"Missing state for {model_key}/{method_key}: {state_path}")
        if state_path.stat().st_size > 6 * 1024**3 and "_compact" not in state_path.name:
            raise RuntimeError(
                f"Refusing to load oversized non-compact state ({state_path.stat().st_size / 1024**3:.2f} GB): {state_path}. "
                "Regenerate a *_compact.pt state first."
            )

    cleanup_gpu()
    load_kwargs = dict(torch_dtype=torch.float16, trust_remote_code=True, low_cpu_mem_usage=True)
    if model_key == "smollm":
        model = AutoModelForCausalLM.from_pretrained(str(model_info["path"]), **load_kwargs)
        model.to("cuda:0")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_info["path"]),
            device_map="cuda:0",
            **load_kwargs,
        )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_info["path"]), trust_remote_code=True)

    if method_key == "fp16":
        return model, tokenizer

    quant_state = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)

    if method_key == "gptq":
        apply_gptq_to_model_gpu(model, quant_state)
    else:
        apply_mixed_precision_to_model_gpu(model, quant_state)

    del quant_state
    gc.collect()
    torch.cuda.empty_cache()
    return model, tokenizer


def metric_score(results: dict) -> float:
    r = results["results"][TASK]
    for key in [
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "exact_match,none",
        "flexible_extract,none",
    ]:
        if key in r:
            return round(float(r[key]) * 100, 2)
    raise KeyError(f"No GSM8K exact-match metric in {list(r)}")


def sample_correct(sample: dict) -> int:
    for key in [
        "exact_match",
        "exact_match,flexible-extract",
        "exact_match,strict-match",
        "flexible_extract",
    ]:
        if key in sample:
            value = sample[key]
            if isinstance(value, (list, tuple)):
                value = value[0]
            return int(float(value) > 0.5)
    raise KeyError(f"No exact-match sample key in {sample.keys()}")


def compact_sample(sample: dict) -> dict:
    doc = sample.get("doc") or {}
    filtered = sample.get("filtered_resps") or []
    resps = sample.get("resps") or []
    pred = filtered[0] if filtered else None
    raw_pred = resps[0][0] if resps and isinstance(resps[0], list) and resps[0] else (resps[0] if resps else None)
    return {
        "doc_id": sample.get("doc_id"),
        "question": doc.get("question"),
        "answer": doc.get("answer"),
        "target": sample.get("target"),
        "prediction": pred,
        "raw_prediction": raw_pred,
        "correct": sample_correct(sample),
        "doc_hash": sample.get("doc_hash"),
        "prompt_hash": sample.get("prompt_hash"),
        "target_hash": sample.get("target_hash"),
    }


def sample_path(model_key: str, method_key: str, n_examples: int = N_EXAMPLES) -> Path:
    return SAMPLE_DIR / f"{model_key}__{method_key}__gsm8k{n_examples}.jsonl"


def evaluate_one(
    model_key: str,
    method_key: str,
    batch_size: int,
    max_new_tokens: int,
    force: bool = False,
    n_examples: int = N_EXAMPLES,
):
    out_path = sample_path(model_key, method_key, n_examples)
    if out_path.exists() and not force:
        print(f"Skip existing {out_path}", flush=True)
        return

    indices = ensure_indices(n_examples)
    model, tokenizer = load_model(model_key, method_key)
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_batch_size=batch_size)
    t0 = time.time()
    print(f"Evaluating {model_key}/{method_key}: {len(indices)} GSM8K examples", flush=True)
    results = simple_evaluate(
        model=lm,
        tasks=[TASK],
        batch_size=batch_size,
        limit=None,
        samples={TASK: indices},
        log_samples=True,
        bootstrap_iters=0,
        cache_requests=False,
        gen_kwargs={"temperature": 0.0, "max_new_tokens": max_new_tokens, "do_sample": False},
    )
    score = metric_score(results)
    rows = [compact_sample(s) for s in results["samples"][TASK]]
    rows = sorted(rows, key=lambda r: r["doc_id"])
    jsonl_write(out_path, rows)
    summary = {
        "model_key": model_key,
        "model": MODELS[model_key]["name"],
        "method_key": method_key,
        "method": METHODS.get(method_key, QWEN05_ABLATIONS.get(method_key, {})).get("paper", method_key),
        "task": TASK,
        "n": len(rows),
        "n_requested": n_examples,
        "score": score,
        "correct": sum(r["correct"] for r in rows),
        "sample_file": str(out_path.relative_to(ROOT)),
        "elapsed_sec": round(time.time() - t0, 1),
        "max_new_tokens": max_new_tokens,
        "batch_size": batch_size,
    }
    append_summary(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    del lm, model, tokenizer, results
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_gpu()


def append_summary(record: dict) -> None:
    path = OUT_DIR / "gsm8k500_summary.jsonl"
    records = []
    if path.exists():
        records = load_jsonl(path)
    key = (record["model_key"], record["method_key"])
    replaced = False
    for i, r in enumerate(records):
        if (r.get("model_key"), r.get("method_key")) == key:
            records[i] = record
            replaced = True
            break
    if not replaced:
        records.append(record)
    jsonl_write(path, records)


def ablation_policy(mode: str, protected: set[int]):
    def policy(layer_idx, layer_name, layer_short):
        ln = parse_layer_num(layer_name)
        in_sensitive = ln in protected
        in_qkv = layer_short in ATTN_PROJ
        in_ffn_or_o = layer_short in FFN_PROJ

        if mode == "only_sensitive":
            return "w8" if in_sensitive else ("w4" if (in_qkv or in_ffn_or_o) else "skip")
        if mode == "only_qkv":
            return "w8" if in_qkv else ("w4" if in_ffn_or_o else "skip")
        if mode == "sensitive_plus_ffn":
            if in_sensitive:
                return "w8"
            return "w8" if in_ffn_or_o else ("w4" if in_qkv else "skip")
        raise ValueError(mode)

    return policy


def quantize_ablation(name: str):
    info = QWEN05_ABLATIONS[name]
    state_path = info["state"]
    if state_path.exists():
        print(f"State exists, skip: {state_path}", flush=True)
        return
    cleanup_gpu()
    model_info = MODELS["qwen05"]
    model = AutoModelForCausalLM.from_pretrained(
        str(model_info["path"]),
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_info["path"]), trust_remote_code=True)
    calib = get_calib_dataset(tokenizer, n_samples=128, max_length=2048)
    state = quantize_model_mixed_precision(
        model,
        calib,
        bits_w4=4,
        group_size=128,
        layer_policy=ablation_policy(info["mode"], info["protected"]),
    )
    torch.save(state, state_path)
    print(f"Saved {state_path} ({state_path.stat().st_size / 1024**3:.2f} GB)", flush=True)
    del state, model, tokenizer, calib
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_gpu()


def mcnemar_exact_p(b: int, c: int) -> float:
    # Two-sided exact binomial McNemar, conditional on discordant pairs.
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired_bootstrap_delta(a: list[int], b: list[int], iters: int = 10000, seed: int = 20260615):
    rng = random.Random(seed)
    n = len(a)
    deltas = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            i = rng.randrange(n)
            s += b[i] - a[i]
        deltas.append(100.0 * s / n)
    deltas.sort()
    lo = deltas[int(0.025 * iters)]
    hi = deltas[int(0.975 * iters)]
    return lo, hi, mean(deltas)


def analyze_pair(model_key: str, baseline: str = "gptq", repair: str = "sg", n_examples: int = N_EXAMPLES) -> dict:
    base_rows = load_jsonl(sample_path(model_key, baseline, n_examples))
    rep_rows = load_jsonl(sample_path(model_key, repair, n_examples))
    base = {r["doc_id"]: r for r in base_rows}
    rep = {r["doc_id"]: r for r in rep_rows}
    ids = sorted(set(base) & set(rep))
    if len(ids) != n_examples:
        raise ValueError(f"{model_key}: expected {n_examples} overlapping ids, got {len(ids)}")
    a = [int(base[i]["correct"]) for i in ids]
    b = [int(rep[i]["correct"]) for i in ids]
    both_correct = sum(1 for x, y in zip(a, b) if x and y)
    both_wrong = sum(1 for x, y in zip(a, b) if not x and not y)
    base_wrong_rep_correct = sum(1 for x, y in zip(a, b) if (not x) and y)
    base_correct_rep_wrong = sum(1 for x, y in zip(a, b) if x and (not y))
    delta = 100.0 * (sum(b) - sum(a)) / len(ids)
    ci_lo, ci_hi, boot_mean = paired_bootstrap_delta(a, b)
    return {
        "model_key": model_key,
        "model": MODELS[model_key]["name"],
        "baseline": baseline,
        "repair": repair,
        "n": len(ids),
        "baseline_acc": round(100.0 * sum(a) / len(ids), 2),
        "repair_acc": round(100.0 * sum(b) / len(ids), 2),
        "delta": round(delta, 2),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_wrong_repair_correct": base_wrong_rep_correct,
        "baseline_correct_repair_wrong": base_correct_rep_wrong,
        "mcnemar_p_exact": mcnemar_exact_p(base_wrong_rep_correct, base_correct_rep_wrong),
        "paired_bootstrap_delta_mean": round(boot_mean, 3),
        "paired_bootstrap_ci95": [round(ci_lo, 2), round(ci_hi, 2)],
    }


def analyze_all(n_examples: int = N_EXAMPLES) -> None:
    records = []
    for model_key in MODELS:
        if sample_path(model_key, "gptq", n_examples).exists() and sample_path(model_key, "sg", n_examples).exists():
            records.append(analyze_pair(model_key, n_examples=n_examples))
    suffix = f"gsm8k{n_examples}"
    json_dump(OUT_DIR / f"paired_stats_{suffix}.json", records)
    lines = [
        "# GSM8K-500 paired analysis",
        "",
        f"Subset: {n_examples} deterministic GSM8K test examples, seed {INDEX_SEED}.",
        "",
        "| Model | GPTQ-W4 | SG-MMP | Delta | GPTQ wrong / SG correct | GPTQ correct / SG wrong | McNemar p | Paired bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        lines.append(
            f"| {r['model']} | {r['baseline_acc']:.2f} | {r['repair_acc']:.2f} | "
            f"{r['delta']:+.2f} | {r['baseline_wrong_repair_correct']} | "
            f"{r['baseline_correct_repair_wrong']} | {r['mcnemar_p_exact']:.4g} | "
            f"[{r['paired_bootstrap_ci95'][0]:+.2f}, {r['paired_bootstrap_ci95'][1]:+.2f}] |"
        )
    (OUT_DIR / f"paired_stats_{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def parse_list(value: str, allowed: dict[str, object]) -> list[str]:
    if value == "all":
        return list(allowed)
    items = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in items if x not in allowed]
    if unknown:
        raise ValueError(f"Unknown entries {unknown}; allowed={list(allowed)}")
    return items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["indices", "run-core", "quantize-ablation", "eval-ablation", "analyze"])
    p.add_argument("--models", default="all", help="all or comma-list: qwen05,qwen15,smollm")
    p.add_argument("--methods", default="fp16,gptq,sg", help="comma-list for run-core")
    p.add_argument("--ablations", default="all", help="all or comma-list of ablation names")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--n-examples", type=int, default=N_EXAMPLES)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.cmd == "indices":
        indices = ensure_indices(args.n_examples)
        print(f"Wrote/loaded {len(indices)} indices under {OUT_DIR}")
        return

    if args.cmd == "run-core":
        model_keys = parse_list(args.models, MODELS)
        method_keys = parse_list(args.methods, METHODS)
        for model_key in model_keys:
            for method_key in method_keys:
                evaluate_one(
                    model_key,
                    method_key,
                    args.batch_size,
                    args.max_new_tokens,
                    force=args.force,
                    n_examples=args.n_examples,
                )
        return

    if args.cmd == "quantize-ablation":
        for name in parse_list(args.ablations, QWEN05_ABLATIONS):
            quantize_ablation(name)
        return

    if args.cmd == "eval-ablation":
        for name in parse_list(args.ablations, QWEN05_ABLATIONS):
            evaluate_one(
                "qwen05",
                name,
                args.batch_size,
                args.max_new_tokens,
                force=args.force,
                n_examples=args.n_examples,
            )
        return

    if args.cmd == "analyze":
        analyze_all(args.n_examples)
        return


if __name__ == "__main__":
    main()

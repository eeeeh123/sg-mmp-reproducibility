"""Direct GSM8K generation evaluator for larger paired comparisons.

Why this exists:
- lm-eval remains the source for the original broad 300-example tables.
- For 500-example paired robustness checks we need per-example correctness logs.
- lm-eval log_samples was slow/stalled in this Windows/Codex environment.

This script therefore implements a transparent, fixed evaluation protocol:
- deterministic fixed GSM8K test subset indices;
- fixed 5-shot prompt from the GSM8K training split;
- greedy generation with a shared max_new_tokens value;
- lm-eval-compatible flexible numeric extraction;
- per-example JSONL logs for paired bootstrap and McNemar tests.

The output should be described as a "direct GSM8K-500 validation" unless the
paper is later updated to use this script for all GSM8K tables.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

sys.path.insert(0, ".")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import torch
import pyarrow.ipc as pa_ipc
from transformers import AutoModelForCausalLM, AutoTokenizer

from ptq.eval import cleanup_gpu
from ptq.quant.gptq import apply_gptq_to_model_gpu
from ptq.quant.mixed_precision import apply_mixed_precision_to_model_gpu


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "fix_gsm8k_500" / "results_direct"
SAMPLE_DIR = OUT / "samples"
LOG_DIR = OUT / "logs"
STATE_DIR = ROOT / "results"
MODEL_DIR = ROOT / "models"
for p in [OUT, SAMPLE_DIR, LOG_DIR]:
    p.mkdir(parents=True, exist_ok=True)

INDEX_SEED = 20260615
DEFAULT_N = 500
HF_DATASETS_ROOT = Path(
    os.environ.get(
        "HF_DATASETS_CACHE",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        / "datasets",
    )
)
GSM8K_CACHE_ROOT = (
    HF_DATASETS_ROOT
    / "openai___gsm8k"
    / "main"
    / "0.0.0"
    / "740312add88f781978c0658806c59bc2815b9866"
)

MODEL_SPECS = {
    "qwen05": {
        "name": "Qwen2.5-0.5B",
        "path": MODEL_DIR / "Qwen2.5-0.5B",
        "gptq": STATE_DIR / "Qwen2.5-0.5B_gptq_compact.pt",
        "sg": STATE_DIR / "Qwen2.5-0.5B_config_b.pt",
    },
    "qwen15": {
        "name": "Qwen2.5-1.5B",
        "path": MODEL_DIR / "Qwen2.5-1.5B",
        "gptq": STATE_DIR / "Qwen2.5-1.5B_gptq_compact.pt",
        "sg": STATE_DIR / "Qwen2.5-1.5B_config_b_2a.pt",
    },
    "smollm": {
        "name": "SmolLM-1.7B",
        "path": MODEL_DIR / "SmolLM-1.7B",
        "gptq": STATE_DIR / "SmolLM-1.7B_gptq_compact.pt",
        "gptq_full": STATE_DIR / "SmolLM-1.7B_gptq.pt",
        "sg": STATE_DIR / "SmolLM-1.7B_config_b_compact.pt",
        "sg_full": STATE_DIR / "SmolLM-1.7B_config_b.pt",
    },
    "tinyllama": {
        "name": "TinyLlama-1.1B-intermediate-step-1431k-3T",
        "path": MODEL_DIR / "TinyLlama-1.1B-intermediate-step-1431k-3T",
        "gptq": STATE_DIR / "TinyLlama-1.1B-intermediate-step-1431k-3T_gptq_compact.pt",
        "sg": STATE_DIR / "TinyLlama-1.1B-intermediate-step-1431k-3T_sg_mmp_compact.pt",
        "prompt_style": "raw",
    },
    "llama32": {
        "name": "Llama-3.2-1B-Instruct",
        "path": MODEL_DIR / "Llama-3.2-1B-Instruct",
        "gptq": STATE_DIR / "Llama-3.2-1B-Instruct_gptq_compact.pt",
        "sg": STATE_DIR / "Llama-3.2-1B-Instruct_sg_mmp_compact.pt",
        "prompt_style": "chat",
    },
    "gemma2": {
        "name": "gemma-2-2b-it",
        "path": MODEL_DIR / "gemma-2-2b-it",
        "gptq": STATE_DIR / "gemma-2-2b-it_gptq_compact.pt",
        "sg": STATE_DIR / "gemma-2-2b-it_sg_mmp_compact.pt",
        "prompt_style": "chat",
    },
}

METHOD_SPECS = {
    "fp16": {"label": "FP16", "kind": "fp16"},
    "gptq": {"label": "GPTQ-W4", "kind": "gptq"},
    "sg": {"label": "SG-MMP", "kind": "mixed"},
    # Qwen2.5-0.5B same-budget controls from exp17.
    "random_42": {
        "label": "Random 42",
        "kind": "mixed",
        "state": STATE_DIR / "Qwen2.5-0.5B_random_42.pt",
        "models": {"qwen05"},
    },
    "random_123": {
        "label": "Random 123",
        "kind": "mixed",
        "state": STATE_DIR / "Qwen2.5-0.5B_random_123.pt",
        "models": {"qwen05"},
    },
    "random_456": {
        "label": "Random 456",
        "kind": "mixed",
        "state": STATE_DIR / "Qwen2.5-0.5B_random_456.pt",
        "models": {"qwen05"},
    },
    "first_4": {
        "label": "First 4",
        "kind": "mixed",
        "state": STATE_DIR / "Qwen2.5-0.5B_first_4.pt",
        "models": {"qwen05"},
    },
    "last_4": {
        "label": "Last 4",
        "kind": "mixed",
        "state": STATE_DIR / "Qwen2.5-0.5B_last_4.pt",
        "models": {"qwen05"},
    },
    # Qwen2.5-0.5B module ablations. The states are produced by run.py
    # quantize-ablation and evaluated here through the stable direct path.
    "abl_only_sensitive_w8": {
        "label": "Only sensitive layers W8",
        "kind": "mixed",
        "state": STATE_DIR / "Qwen2.5-0.5B_abl_only_sensitive_w8.pt",
        "models": {"qwen05"},
    },
    "abl_only_qkv_w8": {
        "label": "Only q/k/v W8",
        "kind": "mixed",
        "state": STATE_DIR / "Qwen2.5-0.5B_abl_only_qkv_w8.pt",
        "models": {"qwen05"},
    },
    "abl_sensitive_plus_ffn_w8": {
        "label": "Sensitive layers + FFN W8",
        "kind": "mixed",
        "state": STATE_DIR / "Qwen2.5-0.5B_abl_sensitive_plus_ffn_w8.pt",
        "models": {"qwen05"},
    },
}

CORE_METHODS = ["fp16", "gptq", "sg"]
LARGE_STATE_LIMIT = 6 * 1024**3


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def status(stage: str, **extra) -> None:
    record = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        **extra,
    }
    print("[status]", json.dumps(record, ensure_ascii=False), flush=True)
    write_json(OUT / "status.json", record)


def assert_safe_state_path(path: Path, kind: str) -> None:
    size = path.stat().st_size
    if size <= LARGE_STATE_LIMIT or "_compact" in path.name:
        return
    if os.environ.get("PTQ_ALLOW_UNSAFE_LARGE_STATE_LOAD") == "1":
        status("unsafe_large_state_allowed", state=str(path), gb=round(size / 1024**3, 2), kind=kind)
        return
    raise RuntimeError(
        f"Refusing to load oversized non-compact {kind} state ({size / 1024**3:.2f} GB): {path}. "
        "This legacy state format has triggered native Windows/PyTorch access violations in this environment. "
        "Regenerate a *_compact.pt state, or set PTQ_ALLOW_UNSAFE_LARGE_STATE_LOAD=1 only for manual debugging."
    )


def append_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_arrow(split: str) -> Path:
    direct = GSM8K_CACHE_ROOT / f"gsm8k-{split}.arrow"
    if direct.exists():
        return direct
    root = HF_DATASETS_ROOT / "openai___gsm8k"
    matches = sorted(root.rglob(f"gsm8k-{split}.arrow"), key=lambda p: p.stat().st_mtime, reverse=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Cannot find cached GSM8K {split} arrow file under {root}. "
        "Run the original lm-eval/GSM8K dataset download once, or place gsm8k-train.arrow and gsm8k-test.arrow in the cache."
    )


def read_arrow_split(split: str) -> list[dict]:
    path = find_arrow(split)
    status("load_arrow_start", split=split, path=str(path))
    with pa_ipc.open_stream(str(path)) as reader:
        rows = reader.read_all().to_pylist()
    status("load_arrow_done", split=split, rows=len(rows))
    return rows


def get_dataset():
    # Avoid datasets.load_dataset() here. In this Windows/Codex environment it
    # can stall while resolving the HF dataset builder even when the Arrow cache
    # is already present. Reading the cached Arrow files is deterministic and
    # keeps this validation fully offline.
    return read_arrow_split("train"), read_arrow_split("test")


def fixed_indices(n: int) -> list[int]:
    if not 1 <= n <= 1319:
        raise ValueError(f"GSM8K test size is 1319; got n={n}")
    path = OUT / f"gsm8k_{n}_indices.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["indices"]
    if n == 1319:
        selected = list(range(1319))
        seed = None
        selection = "all official GSM8K test examples in dataset order"
    else:
        rng = random.Random(INDEX_SEED)
        indices = list(range(1319))
        rng.shuffle(indices)
        selected = sorted(indices[:n])
        seed = INDEX_SEED
        selection = "sorted(shuffled(range(1319))[:n])"
    write_json(
        path,
        {
            "dataset": "openai/gsm8k",
            "split": "test",
            "n": n,
            "seed": seed,
            "selection": selection,
            "indices": selected,
        },
    )
    return selected


def gold_answer(answer: str) -> str:
    return answer.split("####")[-1].strip().replace(",", "").replace("$", "")


def normalize_num(s: str | None) -> str | None:
    if s is None:
        return None
    s = s.strip()
    s = s.replace(",", "").replace("$", "")
    s = re.sub(r"\.$", "", s)
    return s


FLEX_RE = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def extract_prediction(text: str) -> str | None:
    # lm-eval flexible-extract uses group_select=-1 and take_first on regex
    # filters. For generated text, the most robust equivalent is the final
    # numeric expression after truncating at the next question marker.
    for stop in ["Question:", "</s>", "<|im_end|>", "<|user|>", "<|system|>"]:
        if stop in text:
            text = text.split(stop, 1)[0]
    matches = []
    for m in FLEX_RE.finditer(text):
        matches.append(m.group(0))
    if not matches:
        return None
    return normalize_num(matches[-1])


def is_correct(pred: str | None, gold: str) -> int:
    pred_n = normalize_num(pred)
    gold_n = normalize_num(gold)
    return int(pred_n is not None and pred_n == gold_n)


def build_fewshot(train_split, k: int = 5) -> str:
    blocks = []
    for i in range(k):
        ex = train_split[i]
        blocks.append(f"Question: {ex['question']}\nAnswer: {ex['answer']}")
    return "\n\n".join(blocks) + "\n\n"


def build_prompt(prefix: str, question: str) -> str:
    return prefix + f"Question: {question}\nAnswer:"


def build_chat_fewshot_prompt(tok, train_split, question: str, k: int = 5) -> str:
    messages = []
    for i in range(k):
        ex = train_split[i]
        messages.append({"role": "user", "content": f"Question: {ex['question']}\nAnswer:"})
        messages.append({"role": "assistant", "content": ex["answer"]})
    messages.append({"role": "user", "content": f"Question: {question}\nAnswer:"})
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_model_prompts(model_key: str, tok, train_split, prefix: str, questions: list[str]) -> list[str]:
    prompts = [build_prompt(prefix, question) for question in questions]
    if MODEL_SPECS.get(model_key, {}).get("prompt_style") != "chat":
        return prompts
    if not getattr(tok, "chat_template", None):
        raise RuntimeError(f"{model_key} tokenizer has no chat_template; cannot run chat-formatted validation.")
    return [build_chat_fewshot_prompt(tok, train_split, question, k=5) for question in questions]


def load_model(model_key: str, method: str):
    spec = MODEL_SPECS[model_key]
    method_spec = METHOD_SPECS[method]
    allowed_models = method_spec.get("models")
    if allowed_models is not None and model_key not in allowed_models:
        raise ValueError(f"{method} is only defined for {sorted(allowed_models)}, got {model_key}")

    kind = method_spec["kind"]
    state_path = None
    if kind == "gptq":
        state_path = Path(method_spec.get("state") or spec["gptq"])
        if not state_path.exists():
            full_path = spec.get("gptq_full")
            if full_path is not None and Path(full_path).exists():
                raise FileNotFoundError(
                    f"Missing compact GPTQ state for {model_key}: {state_path}. "
                    f"Refusing to load oversized non-compact state {full_path}, "
                    "because it is known to trigger a native PyTorch/Windows access violation. "
                    "Regenerate a compact GPTQ state before evaluating this method."
                )
            raise FileNotFoundError(f"Missing GPTQ state for {model_key}/{method}: {state_path}")
        assert_safe_state_path(state_path, "GPTQ")
    elif kind == "mixed":
        state_path = Path(method_spec.get("state") or spec["sg"])
        if not state_path.exists():
            full_path = spec.get("sg_full")
            if full_path is not None and Path(full_path).exists():
                raise FileNotFoundError(
                    f"Missing compact SG-MMP state for {model_key}: {state_path}. "
                    f"Refusing to load oversized legacy state {full_path}; regenerate compact SG-MMP first."
                )
            raise FileNotFoundError(f"Missing quantized state for {model_key}/{method}: {state_path}")
        assert_safe_state_path(state_path, "mixed-precision")
    elif kind != "fp16":
        raise ValueError(method)

    status("load_model_start", model=model_key, method=method, path=str(spec["path"]))
    cleanup_gpu()
    load_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "local_files_only": True,
    }
    # SmolLM has repeatedly crashed in native Windows/PyTorch code when loaded
    # through the Transformers device_map path in this desktop environment.
    # TinyLlama is also loaded CPU-first for consistency with the long-running
    # new-family validation. Loading on CPU first avoids the device_map memory-
    # estimation allocation and gives finer checkpoints if the platform fails.
    if model_key in {"smollm", "tinyllama", "llama32", "gemma2"}:
        status("load_model_cpu_start", model=model_key, method=method)
        model = AutoModelForCausalLM.from_pretrained(str(spec["path"]), **load_kwargs)
        status("load_model_cpu_done", model=model_key, method=method)
        status("move_model_cuda_start", model=model_key, method=method)
        model.to("cuda:0")
        status("move_model_cuda_done", model=model_key, method=method)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(spec["path"]),
            device_map="cuda:0",
            **load_kwargs,
        )
        status("load_model_cuda_done", model=model_key, method=method)
    model.eval()
    status("load_tokenizer_start", model=model_key, method=method)
    tok = AutoTokenizer.from_pretrained(str(spec["path"]), trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    status("load_tokenizer_done", model=model_key, method=method)

    if kind == "gptq":
        status("apply_quant_start", model=model_key, method=method, state=str(state_path))
        status("load_state_start", model=model_key, method=method, state=str(state_path))
        state = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
        status("load_state_done", model=model_key, method=method, entries=len(state))
        status("apply_state_start", model=model_key, method=method)
        apply_gptq_to_model_gpu(model, state)
        status("apply_state_done", model=model_key, method=method)
        del state
    elif kind == "mixed":
        status("apply_quant_start", model=model_key, method=method, state=str(state_path))
        status("load_state_start", model=model_key, method=method, state=str(state_path))
        state = torch.load(state_path, map_location="cpu", weights_only=False, mmap=True)
        status("load_state_done", model=model_key, method=method, entries=len(state))
        status("apply_state_start", model=model_key, method=method)
        apply_mixed_precision_to_model_gpu(model, state)
        status("apply_state_done", model=model_key, method=method)
        del state
    gc.collect()
    torch.cuda.empty_cache()
    status(
        "load_model_done",
        model=model_key,
        method=method,
        cuda_memory_mb=round(torch.cuda.memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else None,
    )
    return model, tok


def sample_path(model_key: str, method: str, n: int) -> Path:
    return SAMPLE_DIR / f"{model_key}__{method}__gsm8k{n}.jsonl"


def done_doc_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line)["doc_id"])
            except Exception:
                pass
    return ids


# Revision pipelines may attach immutable provenance without changing the
# historical evaluator's CLI or legacy output schema.
ROW_METADATA: dict = {}


@torch.no_grad()
def evaluate(model_key: str, method: str, n: int, batch_size: int, max_new_tokens: int, force: bool = False):
    status("eval_start", model=model_key, method=method, n=n, batch_size=batch_size, max_new_tokens=max_new_tokens)
    out_path = sample_path(model_key, method, n)
    if force and out_path.exists():
        out_path.unlink()
    train, test = get_dataset()
    indices = fixed_indices(n)
    existing_rows = read_jsonl(out_path) if out_path.exists() else []
    existing_by_id = {
        int(row["doc_id"]): row
        for row in existing_rows
        if "doc_id" in row
    }
    existing_ids = [int(row["doc_id"]) for row in existing_rows if "doc_id" in row]
    expected_metadata = {
        **ROW_METADATA,
        "eval_batch_size_per_gpu": batch_size,
        "max_new_tokens": max_new_tokens,
    }
    if (
        len(existing_rows) != len(existing_ids)
        or len(existing_ids) != len(set(existing_ids))
        or any(index not in indices for index in existing_ids)
        or any(int(row.get("correct", -1)) not in (0, 1) for row in existing_rows)
        or any(
            any(row.get(key) != value for key, value in expected_metadata.items())
            for row in existing_rows
        )
    ):
        raise RuntimeError(
            f"Existing sample file has missing fields, duplicate IDs, or out-of-protocol rows: {out_path}. "
            "Inspect it or rerun this method with --force; do not append mixed evidence."
        )
    done = set(existing_by_id)
    running_correct = sum(int(row["correct"]) for row in existing_by_id.values())
    running_total = len(existing_by_id)
    pending = [i for i in indices if i not in done]
    if not pending:
        print(f"[skip] {model_key}/{method}: {out_path} already has {len(done)} rows", flush=True)
        summarize_one(model_key, method, n)
        return

    try:
        model, tok = load_model(model_key, method)
    except Exception as exc:
        status("eval_failed", model=model_key, method=method, error=repr(exc))
        raise
    prefix = build_fewshot(train, k=5)
    t0 = time.time()
    print(
        f"[run] {model_key}/{method}: pending {len(pending)}/{n}, batch={batch_size}, max_new_tokens={max_new_tokens}",
        flush=True,
    )
    report_every_batches = max(1, math.ceil(25 / batch_size))

    for start in range(0, len(pending), batch_size):
        batch_ids = pending[start : start + batch_size]
        prompts = build_model_prompts(model_key, tok, train, prefix, [test[i]["question"] for i in batch_ids])
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
        input_len = enc["input_ids"].shape[1]
        outputs = model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_ids = outputs[:, input_len:]
        decoded = tok.batch_decode(gen_ids, skip_special_tokens=True)
        rows = []
        for doc_id, gen_text, token_row in zip(batch_ids, decoded, gen_ids):
            ex = test[doc_id]
            gold = gold_answer(ex["answer"])
            pred = extract_prediction(gen_text)
            token_ids = token_row.tolist()
            eos_positions = [
                index for index, token_id in enumerate(token_ids)
                if token_id == tok.eos_token_id
            ]
            ended_with_eos = bool(eos_positions)
            generated_token_count = (
                eos_positions[0] + 1 if ended_with_eos else len(token_ids)
            )
            rows.append(
                {
                    **expected_metadata,
                    "doc_id": doc_id,
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "gold": gold,
                    "prediction": pred,
                    "correct": is_correct(pred, gold),
                    "generation": gen_text,
                    "generated_token_count": generated_token_count,
                    "ended_with_eos": ended_with_eos,
                    "truncated": generated_token_count >= max_new_tokens and not ended_with_eos,
                }
            )
        append_jsonl(out_path, rows)

        running_total += len(rows)
        running_correct += sum(int(row["correct"]) for row in rows)
        done_now = running_total
        acc_so_far = round(100 * running_correct / running_total, 2)
        elapsed = time.time() - t0
        batch_number = start // batch_size + 1
        if batch_number % report_every_batches == 0 or done_now == n:
            print(
                f"[progress] {model_key}/{method}: {done_now}/{n}, "
                f"acc={acc_so_far:.2f}, elapsed={elapsed:.0f}s",
                flush=True,
            )
            status(
                "batch_written",
                model=model_key,
                method=method,
                done=done_now,
                total=n,
                accuracy=acc_so_far,
                sample_file=str(out_path),
            )

        del enc, outputs, gen_ids

    summarize_one(model_key, method, n)
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    cleanup_gpu()


def summarize_rows(rows: list[dict]) -> dict:
    n = len(rows)
    correct = sum(int(r["correct"]) for r in rows)
    return {"n": n, "correct": correct, "accuracy": round(100 * correct / n, 2) if n else 0.0}


def summarize_one(model_key: str, method: str, n: int) -> dict:
    rows = read_jsonl(sample_path(model_key, method, n))
    summary = summarize_rows(rows)
    summary.update(
        {
            "model_key": model_key,
            "model": MODEL_SPECS[model_key]["name"],
            "method": method,
            "method_label": METHOD_SPECS[method]["label"],
            "sample_file": str(sample_path(model_key, method, n).relative_to(ROOT)),
        }
    )
    path = OUT / f"summary_gsm8k{n}.jsonl"
    records = []
    if path.exists():
        records = read_jsonl(path)
    records = [r for r in records if not (r.get("model_key") == model_key and r.get("method") == method)]
    records.append(summary)
    path.write_text("", encoding="utf-8")
    append_jsonl(path, records)
    print("[summary]", json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def mcnemar_exact_p(b: int, c: int) -> float:
    total = b + c
    if total == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(total, i) for i in range(k + 1)) / (2**total)
    return min(1.0, 2 * tail)


def paired_bootstrap(a: list[int], b: list[int], iters: int = 10000, seed: int = INDEX_SEED):
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(iters):
        s = 0
        for _ in range(n):
            j = rng.randrange(n)
            s += b[j] - a[j]
        vals.append(100 * s / n)
    vals.sort()
    return vals[int(0.025 * iters)], vals[int(0.975 * iters)], mean(vals)


def analyze(n: int):
    results = []
    for model_key in MODEL_SPECS:
        gptq_path = sample_path(model_key, "gptq", n)
        sg_path = sample_path(model_key, "sg", n)
        if not (gptq_path.exists() and sg_path.exists()):
            continue
        gptq = {r["doc_id"]: r for r in read_jsonl(gptq_path)}
        sg = {r["doc_id"]: r for r in read_jsonl(sg_path)}
        ids = sorted(set(gptq) & set(sg))
        a = [int(gptq[i]["correct"]) for i in ids]
        b = [int(sg[i]["correct"]) for i in ids]
        both_correct = sum(1 for x, y in zip(a, b) if x and y)
        both_wrong = sum(1 for x, y in zip(a, b) if not x and not y)
        gptq_wrong_sg_correct = sum(1 for x, y in zip(a, b) if (not x) and y)
        gptq_correct_sg_wrong = sum(1 for x, y in zip(a, b) if x and (not y))
        ci_lo, ci_hi, boot_mean = paired_bootstrap(a, b)
        results.append(
            {
                "model_key": model_key,
                "model": MODEL_SPECS[model_key]["name"],
                "n": len(ids),
                "gptq_acc": round(100 * sum(a) / len(ids), 2),
                "sg_acc": round(100 * sum(b) / len(ids), 2),
                "delta": round(100 * (sum(b) - sum(a)) / len(ids), 2),
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "gptq_wrong_sg_correct": gptq_wrong_sg_correct,
                "gptq_correct_sg_wrong": gptq_correct_sg_wrong,
                "mcnemar_p_exact": mcnemar_exact_p(gptq_wrong_sg_correct, gptq_correct_sg_wrong),
                "paired_bootstrap_delta_mean": round(boot_mean, 3),
                "paired_bootstrap_ci95": [round(ci_lo, 2), round(ci_hi, 2)],
            }
        )
    write_json(OUT / f"paired_stats_gsm8k{n}.json", results)
    lines = [
        f"# Direct GSM8K-{n} paired statistics",
        "",
        f"Fixed GSM8K test subset: n={n}, seed={INDEX_SEED}. All reported base-model rows use the lm-eval-style `Question: ...\\nAnswer:` 5-shot prompt.",
        "",
        "| Model | GPTQ-W4 | SG-MMP | Delta | GPTQ wrong / SG correct | GPTQ correct / SG wrong | McNemar p | Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['gptq_acc']:.2f} | {r['sg_acc']:.2f} | {r['delta']:+.2f} | "
            f"{r['gptq_wrong_sg_correct']} | {r['gptq_correct_sg_wrong']} | {r['mcnemar_p_exact']:.4g} | "
            f"[{r['paired_bootstrap_ci95'][0]:+.2f}, {r['paired_bootstrap_ci95'][1]:+.2f}] |"
        )
    (OUT / f"paired_stats_gsm8k{n}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


def smoke_load(model_key: str, method: str) -> None:
    status("smoke_load_start", model=model_key, method=method)
    try:
        model, tok = load_model(model_key, method)
    except Exception as exc:
        status("smoke_load_failed", model=model_key, method=method, error=repr(exc))
        raise
    status(
        "smoke_load_done",
        model=model_key,
        method=method,
        cuda_memory_mb=round(torch.cuda.memory_allocated() / 1024**2, 1) if torch.cuda.is_available() else None,
    )
    del model, tok
    cleanup_gpu()
    status("smoke_cleanup_done", model=model_key, method=method)


def parse_csv(value: str, allowed: dict) -> list[str]:
    if value == "all":
        return CORE_METHODS if allowed is METHOD_SPECS else list(allowed)
    items = [x.strip() for x in value.split(",") if x.strip()]
    bad = [x for x in items if x not in allowed]
    if bad:
        raise ValueError(f"Unknown {bad}; allowed={list(allowed)}")
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["run", "analyze", "indices", "smoke-load"])
    parser.add_argument("--models", default="all")
    parser.add_argument("--methods", default="fp16,gptq,sg")
    parser.add_argument("--n", type=int, default=DEFAULT_N)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.cmd == "indices":
        indices = fixed_indices(args.n)
        print(f"indices n={len(indices)} written under {OUT}", flush=True)
        return
    if args.cmd == "analyze":
        analyze(args.n)
        return
    if args.cmd == "smoke-load":
        for model_key in parse_csv(args.models, MODEL_SPECS):
            for method in parse_csv(args.methods, METHOD_SPECS):
                smoke_load(model_key, method)
        return

    for model_key in parse_csv(args.models, MODEL_SPECS):
        for method in parse_csv(args.methods, METHOD_SPECS):
            evaluate(model_key, method, args.n, args.batch_size, args.max_new_tokens, force=args.force)


if __name__ == "__main__":
    main()

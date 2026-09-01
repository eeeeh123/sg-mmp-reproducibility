"""校准数据加载工具"""

import os
import random
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path


def _datasets_cache_root() -> Path:
    configured = os.environ.get("HF_DATASETS_CACHE")
    if configured:
        return Path(configured)
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    return hf_home / "datasets"


def _latest_arrow(pattern: str) -> Path | None:
    root = _datasets_cache_root()
    matches = sorted(root.rglob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _read_arrow_rows(path: Path) -> list[dict]:
    import pyarrow.ipc as pa_ipc

    with pa_ipc.open_stream(str(path)) as reader:
        return reader.read_all().to_pylist()


def _load_wikitext_train_texts() -> list[str] | None:
    path = _latest_arrow("wikitext-train.arrow")
    if path is None:
        return None
    rows = _read_arrow_rows(path)
    return [r["text"] for r in rows if r.get("text")]


def _load_gsm8k_train_questions() -> list[str] | None:
    path = _latest_arrow("gsm8k-train.arrow")
    if path is None:
        return None
    rows = _read_arrow_rows(path)
    return [r["question"] for r in rows if r.get("question")]


def _packed_random_segments(
    tokenizer: AutoTokenizer,
    texts: list[str],
    n_samples: int,
    max_length: int,
    seed: int,
) -> torch.Tensor:
    """Build exact-length calibration segments from a deterministic token pool.

    The previous loader padded short, independent texts to ``max_length`` and
    passed the padding through the model without an attention mask. Packing a
    shared text stream avoids padded-token Hessian contamination while keeping
    the requested sample count and sequence length explicit.
    """
    if n_samples <= 0 or max_length <= 0:
        raise ValueError("n_samples and max_length must be positive")
    token_pool: list[int] = []
    target_pool_size = max(max_length + 1, 2 * n_samples * max_length)
    separator = tokenizer.eos_token_id
    for text in texts:
        if not text or not text.strip():
            continue
        token_pool.extend(
            tokenizer(text, add_special_tokens=False, return_attention_mask=False)[
                "input_ids"
            ]
        )
        if separator is not None:
            token_pool.append(int(separator))
        if len(token_pool) >= target_pool_size:
            break
    if len(token_pool) < max_length:
        raise RuntimeError(
            f"Calibration token pool has {len(token_pool)} tokens; need {max_length}"
        )
    rng = random.Random(seed)
    max_start = len(token_pool) - max_length
    starts = [rng.randint(0, max_start) for _ in range(n_samples)]
    return torch.stack(
        [
            torch.tensor(token_pool[start : start + max_length], dtype=torch.long)
            for start in starts
        ]
    )


def get_calib_dataset(
    tokenizer: AutoTokenizer,
    n_samples: int = 128,
    max_length: int = 2048,
    dataset_name: str = "allenai/c4",
    seed: int = 42,
) -> torch.Tensor:
    """加载校准数据集，截断并转成 token ids。

    返回 shape: (n_samples, max_length)
    """
    # 默认用 wikitext-2 训练集（更小更稳定，不需下载 C4）
    if dataset_name == "allenai/c4":
        dataset_name = "wikitext"
    if dataset_name == "wikitext":
        texts = _load_wikitext_train_texts()
        if texts is None:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = list(ds["text"])
        texts = [t for t in texts if t and t.strip()]
    elif dataset_name == "gsm8k":
        questions = _load_gsm8k_train_questions()
        if questions is None:
            ds = load_dataset("gsm8k", "main", split="train")
            questions = [ds[int(i)]["question"] for i in range(len(ds))]
        texts = questions
    else:
        ds = load_dataset(dataset_name, "en", split="validation", streaming=True)
        ds = ds.shuffle(seed=seed).take(max(n_samples * 4, n_samples))
        texts = [item["text"] for item in ds]
    return _packed_random_segments(tokenizer, texts, n_samples, max_length, seed)


def get_wikitext2(tokenizer: AutoTokenizer, max_length: int = 2048) -> torch.Tensor:
    """加载 WikiText-2 测试集。"""
    ds = load_dataset("wikitext-2-raw-v1", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt")
    return tokens.input_ids[0]

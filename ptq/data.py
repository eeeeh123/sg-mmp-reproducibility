"""校准数据加载工具"""

import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from pathlib import Path


def _latest_arrow(pattern: str) -> Path | None:
    root = Path.home() / ".cache" / "huggingface" / "datasets"
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
        # 过滤空文本，取足够长的
        texts = [t for t in texts if t and len(t.strip()) > 100]
        import random
        random.seed(seed)
        random.shuffle(texts)
        texts = texts[:n_samples]
    elif dataset_name == "gsm8k":
        questions = _load_gsm8k_train_questions()
        if questions is None:
            ds = load_dataset("gsm8k", "main", split="train")
            questions = [ds[int(i)]["question"] for i in range(len(ds))]
        import random
        random.seed(seed)
        indices = list(range(len(questions)))
        random.shuffle(indices)
        texts = [questions[int(i)] for i in indices[:n_samples]]
    else:
        ds = load_dataset(dataset_name, "en", split="validation", streaming=True)
        ds = ds.shuffle(seed=seed).take(n_samples)
        texts = [item["text"] for item in ds]

    samples = []
    for text in texts:
        tokens = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tok_ids = tokens.input_ids[0]
        # pad 到 max_length
        if tok_ids.shape[0] < max_length:
            tok_ids = torch.cat(
                [tok_ids, torch.zeros(max_length - tok_ids.shape[0], dtype=torch.long)]
            )
        samples.append(tok_ids)

    return torch.stack(samples)


def get_wikitext2(tokenizer: AutoTokenizer, max_length: int = 2048) -> torch.Tensor:
    """加载 WikiText-2 测试集。"""
    ds = load_dataset("wikitext-2-raw-v1", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    tokens = tokenizer(text, return_tensors="pt")
    return tokens.input_ids[0]

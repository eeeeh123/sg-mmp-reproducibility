"""Same-item GSM8K free-generation versus multiple-choice format control.

Multiple-choice candidates are built deterministically from GSM8K train
answers. No other test example's label is used to construct a target item's
distractors. Choices are ranked by conditional log likelihood.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

sys.path.insert(0, ".")

from experiments.fix_gsm8k_500.direct_eval import gold_answer
from experiments.revision_full.protocol import GSM8K_TEST_SIZE, MODEL_SPECS, RESULTS_DIR
from experiments.revision_full.run import configure_direct_eval, get_dataset


OUT = RESULTS_DIR / "format_control"
OUT.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUT / "gsm8k_mcq_manifest.json"
CHOICE_LABELS = ("A", "B", "C", "D")
CHOICE_SEED = 20260831


def normalize_answer(value: str) -> str:
    return gold_answer(value).strip()


def decimal_value(value: str) -> Decimal | None:
    try:
        return Decimal(value.replace(",", ""))
    except (InvalidOperation, ValueError):
        return None


def choose_distractors(gold: str, train_answers: list[str], seed: int) -> list[str]:
    unique = sorted({answer for answer in train_answers if answer != gold})
    gold_number = decimal_value(gold)
    if gold_number is not None:

        def distance(candidate: str):
            number = decimal_value(candidate)
            if number is None:
                return (1, math.inf, candidate)
            magnitude = abs(float(number - gold_number))
            scale = max(1.0, abs(float(gold_number)))
            return (0, magnitude / scale, candidate)

        unique.sort(key=distance)
        positions = [0, min(4, len(unique) - 1), min(9, len(unique) - 1)]
        candidates = []
        for position in positions:
            value = unique[position]
            if value not in candidates:
                candidates.append(value)
        if len(candidates) == 3:
            return candidates

    rng = random.Random(seed)
    rng.shuffle(unique)
    return unique[:3]


def make_item(question: str, answer: str, train_answers: list[str], seed: int) -> dict:
    gold = normalize_answer(answer)
    choices = [gold, *choose_distractors(gold, train_answers, seed)]
    if len(set(choices)) != 4:
        raise RuntimeError(f"Could not create four unique choices for answer {gold!r}")
    rng = random.Random(seed)
    rng.shuffle(choices)
    correct_index = choices.index(gold)
    return {
        "question": question,
        "choices": choices,
        "correct_index": correct_index,
        "correct_label": CHOICE_LABELS[correct_index],
    }


def choice_block(item: dict) -> str:
    return "\n".join(
        f"{label}. {choice}" for label, choice in zip(CHOICE_LABELS, item["choices"])
    )


def raw_prompt(demos: list[dict], target: dict) -> str:
    blocks = []
    for item in demos:
        blocks.append(
            f"Question: {item['question']}\nChoices:\n{choice_block(item)}\nAnswer: {item['correct_label']}"
        )
    blocks.append(
        f"Question: {target['question']}\nChoices:\n{choice_block(target)}\nAnswer:"
    )
    return "\n\n".join(blocks)


def chat_prompt(tokenizer, demos: list[dict], target: dict) -> str:
    messages = []
    for item in demos:
        messages.append(
            {
                "role": "user",
                "content": f"Question: {item['question']}\nChoices:\n{choice_block(item)}\nAnswer:",
            }
        )
        messages.append({"role": "assistant", "content": item["correct_label"]})
    messages.append(
        {
            "role": "user",
            "content": f"Question: {target['question']}\nChoices:\n{choice_block(target)}\nAnswer:",
        }
    )
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def score_choice_batch(model, tokenizer, prompts: list[str]) -> list[list[float]]:
    """Score four labels for several items in one padded forward pass."""
    import torch

    sequences = []
    target_starts = []
    owners = []
    for owner, prompt in enumerate(prompts):
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True
        )["input_ids"][0]
        for label in CHOICE_LABELS:
            completion = tokenizer(
                f" {label}", return_tensors="pt", add_special_tokens=False
            )["input_ids"][0]
            sequences.append(torch.cat([prompt_ids, completion]))
            target_starts.append(len(prompt_ids))
            owners.append(owner)
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length), tokenizer.pad_token_id, dtype=torch.long
    )
    attention = torch.zeros((len(sequences), max_length), dtype=torch.long)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = sequence
        attention[row, : len(sequence)] = 1
    input_ids = input_ids.to(model.device)
    attention = attention.to(model.device)
    with torch.no_grad():
        logits = model(
            input_ids=input_ids, attention_mask=attention, use_cache=False
        ).logits
    scores = [[] for _ in prompts]
    for row, sequence in enumerate(sequences):
        start = target_starts[row]
        score = 0.0
        for position in range(start, len(sequence)):
            token_logits = logits[row, position - 1].float()
            score += float(
                (
                    token_logits[sequence[position]]
                    - torch.logsumexp(token_logits, dim=-1)
                ).item()
            )
        scores[owners[row]].append(score)
    return scores


def score_choices(model, tokenizer, prompt: str) -> list[float]:
    """Compatibility wrapper for one item."""
    return score_choice_batch(model, tokenizer, [prompt])[0]


def output_path(model_key: str, method: str) -> Path:
    return OUT / f"{model_key}__{method}__gsm8k_mcq{GSM8K_TEST_SIZE}.jsonl"


def done_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as stream:
        return {int(json.loads(line)["doc_id"]) for line in stream if line.strip()}


def prepare_manifest(force: bool = False) -> dict:
    if MANIFEST_PATH.exists() and not force:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    train, test = get_dataset()
    if len(test) != GSM8K_TEST_SIZE:
        raise RuntimeError(f"Expected {GSM8K_TEST_SIZE} test rows, found {len(test)}")
    train_answers = [normalize_answer(row["answer"]) for row in train[5:]]
    demos = [
        make_item(row["question"], row["answer"], train_answers, CHOICE_SEED + index)
        for index, row in enumerate(train[:5])
    ]
    items = [
        {
            "doc_id": doc_id,
            **make_item(
                row["question"],
                row["answer"],
                train_answers,
                CHOICE_SEED + 10_000 + doc_id,
            ),
        }
        for doc_id, row in enumerate(test)
    ]
    manifest = {
        "dataset": "openai/gsm8k/main",
        "source_split": "test",
        "n": GSM8K_TEST_SIZE,
        "choice_seed": CHOICE_SEED,
        "distractor_source": "GSM8K train answers excluding the five prompt demonstrations",
        "scoring": "conditional log likelihood of answer labels A/B/C/D",
        "demos": demos,
        "items": items,
    }
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def evaluate(
    model_key: str,
    variant: str,
    calib_seed: int | None,
    force: bool,
    batch_size: int,
) -> None:
    from ptq.eval import cleanup_gpu

    direct, method = configure_direct_eval(model_key, variant, calib_seed)
    path = output_path(model_key, method)
    if force and path.exists():
        path.unlink()
    completed = done_ids(path)
    manifest = prepare_manifest()
    demos = manifest["demos"]
    model, tokenizer = direct.load_model(model_key, method)
    prompt_style = MODEL_SPECS[model_key]["prompt_style"]
    pending = [
        item for item in manifest["items"] if int(item["doc_id"]) not in completed
    ]
    with path.open("a", encoding="utf-8") as stream:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            prompts = [
                chat_prompt(tokenizer, demos, item)
                if prompt_style == "chat"
                else raw_prompt(demos, item)
                for item in batch
            ]
            batch_scores = score_choice_batch(model, tokenizer, prompts)
            for item, scores in zip(batch, batch_scores):
                prediction = max(range(4), key=scores.__getitem__)
                record = {
                    "doc_id": int(item["doc_id"]),
                    "choices": item["choices"],
                    "gold_label": item["correct_label"],
                    "prediction_label": CHOICE_LABELS[prediction],
                    "choice_log_likelihoods": scores,
                    "correct": int(prediction == item["correct_index"]),
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            done = len(completed) + start + len(batch)
            if done % 25 < len(batch) or done == GSM8K_TEST_SIZE:
                print(f"{model_key}/{method}: {done}/{GSM8K_TEST_SIZE}", flush=True)
    del model, tokenizer
    cleanup_gpu()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SPECS)
    parser.add_argument("--variant")
    parser.add_argument("--calib-seed", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.prepare_only:
        manifest = prepare_manifest(force=args.force)
        print(f"wrote {MANIFEST_PATH} with {len(manifest['items'])} items")
    else:
        if not args.model or not args.variant:
            raise SystemExit("--model and --variant are required unless --prepare-only is used")
        evaluate(
            args.model,
            args.variant,
            args.calib_seed,
            args.force,
            args.batch_size,
        )

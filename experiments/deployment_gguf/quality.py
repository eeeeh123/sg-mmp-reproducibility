"""Resume-safe GSM8K quality measurement on the pinned GGUF backend."""

from __future__ import annotations

import hashlib
import json
import os
import re
from functools import lru_cache
from pathlib import Path

from experiments.deployment_gguf.llama_server import LlamaServer
from experiments.deployment_gguf.protocol import (
    GSM8K_TEST_SIZE,
    CONTEXT_TOKENS_PER_SLOT,
    MAX_NEW_TOKENS,
    MODEL_SPECS,
    PROTOCOL_VERSION,
    QUALITY_DIR,
    STATUS_DIR,
    artifact_manifest_path,
    artifact_path,
    atomic_write_json,
    binary_paths,
    json_sha256,
    sha256_file,
)
from experiments.revision_full.question_stop import canonical_answer_prefix


STRICT_RE = re.compile(r"####\s*(-?[$0-9][0-9,.$]*)")


@lru_cache(maxsize=1)
def _direct_eval():
    # That legacy evaluator imports torch and quantization kernels at module
    # import time.  Delay it so preflight, plan generation, artifact audits,
    # and readiness checks do not pay that cost or initialize CUDA.
    from experiments.fix_gsm8k_500 import direct_eval

    return direct_eval


def extract_prediction(text: str) -> str | None:
    return _direct_eval().extract_prediction(text)


def gold_answer(answer: str) -> str:
    return _direct_eval().gold_answer(answer)


def is_correct(prediction: str | None, gold: str) -> int:
    return _direct_eval().is_correct(prediction, gold)


def normalize_num(value: str | None) -> str | None:
    return _direct_eval().normalize_num(value)


def strict_prediction(text: str) -> str | None:
    canonical = canonical_answer_prefix(text).text
    matches = STRICT_RE.findall(canonical)
    return normalize_num(matches[0]) if matches else None


def load_prompts(model_key: str, split: str) -> tuple[list[dict], list[str], object]:
    if split not in {"train", "test"}:
        raise ValueError(split)
    from transformers import AutoTokenizer
    from experiments.revision_full.run import frozen_arrow_rows

    # Gates and workload construction are deliberately train-only.  Do not call
    # the older get_dataset() helper here: it materializes train *and* test even
    # when its caller only asks for train, which would violate the pre-test gate.
    train = frozen_arrow_rows("openai/gsm8k/main/train", "gsm8k-train.arrow")
    rows = (
        train
        if split == "train"
        else frozen_arrow_rows("openai/gsm8k/main/test", "gsm8k-test.arrow")
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_SPECS[model_key]["path"], local_files_only=True
    )
    prefix = _direct_eval().build_fewshot(train, k=5)
    prompts = _direct_eval().build_model_prompts(
        model_key,
        tokenizer,
        train,
        prefix,
        [row["question"] for row in rows],
    )
    return rows, prompts, tokenizer


def quality_sample_path(model_key: str, method: str) -> Path:
    return QUALITY_DIR / "samples" / f"{model_key}__{method}__gsm8k1319.jsonl"


def quality_summary_path(model_key: str, method: str) -> Path:
    return QUALITY_DIR / "summaries" / f"{model_key}__{method}.json"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _append_sync(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def require_quality_gates(model_key: str, method: str) -> dict:
    gate_dir = STATUS_DIR / "gates" / model_key
    required = [
        gate_dir / "conversion.json",
        gate_dir / f"packed__{method}.json",
        artifact_manifest_path(model_key, method),
    ]
    records = {}
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Quality is locked until this gate exists: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("gate_passed") is not True:
            raise RuntimeError(f"Quality is locked by failed gate: {path}")
        records[path] = record
    conversion, packed, manifest = (records[path] for path in required)
    if conversion.get("model_key") != model_key:
        raise RuntimeError("Conversion gate belongs to another model")
    if packed.get("model_key") != model_key or packed.get("method") != method:
        raise RuntimeError("Packed gate belongs to another model or method")
    if manifest.get("model_key") != model_key or manifest.get("method") != method:
        raise RuntimeError("Artifact manifest belongs to another model or method")
    if packed.get("artifact_sha256") != manifest.get("artifact_sha256"):
        raise RuntimeError("Packed gate and artifact manifest hashes disagree")
    return {"conversion": conversion, "packed": packed, "manifest": manifest}


def _validate_quality_row(
    row: dict,
    source_row: dict,
    prompt: str,
    expected_common: dict,
    output: Path,
) -> int:
    doc_id = int(row.get("doc_id", -1))
    if any(row.get(key) != value for key, value in expected_common.items()):
        raise RuntimeError(f"Out-of-protocol row in {output}: {doc_id}")
    if row.get("question_sha256") != hashlib.sha256(
        source_row["question"].encode("utf-8")
    ).hexdigest():
        raise RuntimeError(f"Question mismatch in resumable result {output}: {doc_id}")
    if row.get("prompt_sha256") != hashlib.sha256(prompt.encode("utf-8")).hexdigest():
        raise RuntimeError(f"Prompt mismatch in resumable result {output}: {doc_id}")
    generation = str(row.get("generation", ""))
    gold = gold_answer(source_row["answer"])
    flexible = extract_prediction(generation)
    strict = strict_prediction(generation)
    canonical = canonical_answer_prefix(generation)
    derived = {
        "gold": gold,
        "flexible_prediction": flexible,
        "flexible_correct": is_correct(flexible, gold),
        "strict_prediction": strict,
        "strict_correct": is_correct(strict, gold),
        "delimiter_present": strict is not None,
        "offline_question_marker_found": canonical.marker_found,
        "offline_prefix_flexible_prediction": extract_prediction(canonical.text),
    }
    for key, value in derived.items():
        if row.get(key) != value:
            raise RuntimeError(
                f"Derived field {key!r} is corrupt in {output} at doc_id={doc_id}"
            )
    tokens = row.get("generated_tokens")
    if not isinstance(tokens, list) or any(not isinstance(token, int) for token in tokens):
        raise RuntimeError(f"Invalid generated token record in {output}: {doc_id}")
    predicted = row.get("tokens_predicted")
    if not isinstance(predicted, int) or not 0 < predicted <= MAX_NEW_TOKENS:
        raise RuntimeError(f"Invalid generated token count in {output}: {doc_id}")
    if len(tokens) != predicted:
        raise RuntimeError(
            f"Returned token IDs/count disagree in {output} at doc_id={doc_id}"
        )
    return doc_id


def evaluate_quality(
    model_key: str,
    method: str,
    llama_cpp_dir: Path,
    *,
    gpu: int,
) -> dict:
    gate_records = require_quality_gates(model_key, method)
    artifact = artifact_path(model_key, method)
    artifact_record = json.loads(
        artifact_manifest_path(model_key, method).read_text(encoding="utf-8")
    )
    rows, prompts, tokenizer = load_prompts(model_key, "test")
    if len(rows) != GSM8K_TEST_SIZE:
        raise RuntimeError(f"Expected {GSM8K_TEST_SIZE} test rows, found {len(rows)}")
    if any(
        len(tokenizer.encode(prompt, add_special_tokens=False)) + MAX_NEW_TOKENS + 1
        > CONTEXT_TOKENS_PER_SLOT
        for prompt in prompts
    ):
        raise RuntimeError("A quality prompt exceeds the frozen llama-server slot context")
    output = quality_sample_path(model_key, method)
    existing = _read_jsonl(output)
    binaries = binary_paths(llama_cpp_dir)
    server_binary_sha256 = sha256_file(binaries["server"])
    if any(
        gate_records[name].get("server_binary_sha256") != server_binary_sha256
        for name in ("conversion", "packed")
    ):
        raise RuntimeError(
            "Quality server binary differs from the binary that passed the gates"
        )
    expected_common = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "method": method,
        "artifact_sha256": artifact_record["artifact_sha256"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "online_stop": False,
        "sampling": "greedy",
        "server_binary_sha256": server_binary_sha256,
    }
    existing_by_id = {}
    for row in existing:
        doc_id = int(row.get("doc_id", -1))
        if doc_id in existing_by_id or not 0 <= doc_id < len(rows):
            raise RuntimeError(f"Duplicate or invalid row in {output}: {doc_id}")
        _validate_quality_row(
            row, rows[doc_id], prompts[doc_id], expected_common, output
        )
        existing_by_id[doc_id] = row

    pending = [doc_id for doc_id in range(len(rows)) if doc_id not in existing_by_id]
    server_log = QUALITY_DIR / "server_logs" / f"{model_key}__{method}.jsonl"
    if pending:
        with LlamaServer(
            binaries["server"], artifact, server_log, gpu=gpu, slots=1, cuda=True
        ) as server:
            for completed_index, doc_id in enumerate(pending, start=1):
                response = server.complete(
                    prompts[doc_id], n_predict=MAX_NEW_TOKENS, ignore_eos=False
                )
                generation = str(response.get("content", ""))
                gold = gold_answer(rows[doc_id]["answer"])
                flexible = extract_prediction(generation)
                strict = strict_prediction(generation)
                canonical = canonical_answer_prefix(generation)
                record = {
                    **expected_common,
                    "doc_id": doc_id,
                    "question_sha256": hashlib.sha256(
                        rows[doc_id]["question"].encode("utf-8")
                    ).hexdigest(),
                    "prompt_sha256": hashlib.sha256(
                        prompts[doc_id].encode("utf-8")
                    ).hexdigest(),
                    "gold": gold,
                    "generation": generation,
                    "generated_tokens": [int(token) for token in response.get("tokens", [])],
                    "tokens_predicted": int(
                        response.get("tokens_predicted", len(response.get("tokens", [])))
                    ),
                    "stop_type": response.get("stop_type"),
                    "server_truncated": bool(response.get("truncated", False)),
                    "flexible_prediction": flexible,
                    "flexible_correct": is_correct(flexible, gold),
                    "strict_prediction": strict,
                    "strict_correct": is_correct(strict, gold),
                    "delimiter_present": strict is not None,
                    "offline_question_marker_found": canonical.marker_found,
                    "offline_prefix_flexible_prediction": extract_prediction(canonical.text),
                    "server_timings": response.get("timings"),
                }
                _validate_quality_row(
                    record, rows[doc_id], prompts[doc_id], expected_common, output
                )
                _append_sync(output, record)
                existing_by_id[doc_id] = record
                if completed_index % 28 == 0 or completed_index == len(pending):
                    correct = sum(
                        int(item["flexible_correct"])
                        for item in existing_by_id.values()
                    )
                    print(
                        f"[progress] {model_key}/{method}: {len(existing_by_id)}/"
                        f"{len(rows)}, acc={100 * correct / len(existing_by_id):.2f}",
                        flush=True,
                    )
        resource_record = server.resource_record()
    else:
        resource_record = None
        print(f"[skip] complete quality sample file: {output}", flush=True)

    complete_rows = _read_jsonl(output)
    if len(complete_rows) != GSM8K_TEST_SIZE:
        raise RuntimeError(f"Quality output is incomplete: {len(complete_rows)}/1319")
    complete_rows.sort(key=lambda row: int(row["doc_id"]))
    if [int(row["doc_id"]) for row in complete_rows] != list(range(GSM8K_TEST_SIZE)):
        raise RuntimeError(f"Quality output does not contain every test item exactly once: {output}")
    for doc_id, row in enumerate(complete_rows):
        _validate_quality_row(
            row, rows[doc_id], prompts[doc_id], expected_common, output
        )
    changed_by_offline_truncation = sum(
        row["flexible_prediction"] != row["offline_prefix_flexible_prediction"]
        for row in complete_rows
    )
    summary = {
        **expected_common,
        "n": len(complete_rows),
        "flexible_accuracy": 100
        * sum(int(row["flexible_correct"]) for row in complete_rows)
        / len(complete_rows),
        "strict_accuracy": 100
        * sum(int(row["strict_correct"]) for row in complete_rows)
        / len(complete_rows),
        "delimiter_coverage": sum(bool(row["delimiter_present"]) for row in complete_rows)
        / len(complete_rows),
        "server_truncation_count": sum(
            bool(row["server_truncated"]) or row["stop_type"] == "limit"
            for row in complete_rows
        ),
        "subsequent_question_marker_count": sum(
            bool(row["offline_question_marker_found"]) for row in complete_rows
        ),
        "flexible_predictions_changed_by_offline_question_truncation": changed_by_offline_truncation,
        "sample_file": str(output),
        "sample_file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "prompt_set_sha256": json_sha256(
            [hashlib.sha256(prompt.encode()).hexdigest() for prompt in prompts]
        ),
        "last_server_resource_record": resource_record,
        "complete": True,
    }
    atomic_write_json(quality_summary_path(model_key, method), summary)
    return summary

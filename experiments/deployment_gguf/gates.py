"""Pre-test conversion and packed-backend gates."""

from __future__ import annotations

import gc
import hashlib
import json
import re
import subprocess
from pathlib import Path

from experiments.deployment_gguf.llama_server import LlamaServer
from experiments.deployment_gguf.protocol import (
    LLAMA_CPP_COMMIT,
    MODEL_SPECS,
    PROTOCOL_VERSION,
    STATUS_DIR,
    artifact_manifest_path,
    artifact_path,
    atomic_write_json,
    binary_paths,
    sha256_file,
)
from experiments.deployment_gguf.quality import load_prompts


TRAIN_GATE_INDICES = tuple(range(5, 13))
TOKENS_PER_PROMPT = 16
MIN_ALIGNED_MATCHES = 126
LOGPROB_TOP_K = 8
MIN_LOGPROB_TOP_K_OVERLAP = 7
MAX_COMMON_LOGPROB_ABS_ERROR = 0.05


def _gate_prompts(model_key: str):
    rows, prompts, tokenizer = load_prompts(model_key, "train")
    chosen = [prompts[index] for index in TRAIN_GATE_INDICES]
    identities = [
        {
            "train_index": index,
            "question_sha256": hashlib.sha256(
                rows[index]["question"].encode("utf-8")
            ).hexdigest(),
            "prompt_sha256": hashlib.sha256(prompts[index].encode("utf-8")).hexdigest(),
        }
        for index in TRAIN_GATE_INDICES
    ]
    return chosen, identities, tokenizer


def _continuation_agreement(a: list[list[int]], b: list[list[int]]) -> dict:
    if len(a) != len(b) or len(a) != len(TRAIN_GATE_INDICES):
        raise RuntimeError("Gate continuation count mismatch")
    if any(len(row) != TOKENS_PER_PROMPT for row in a + b):
        raise RuntimeError("Gate did not return the frozen 16 tokens per prompt")
    first = sum(left[0] == right[0] for left, right in zip(a, b))
    aligned = sum(
        left_token == right_token
        for left, right in zip(a, b)
        for left_token, right_token in zip(left, right)
    )
    return {
        "prompts": len(a),
        "tokens_per_prompt": TOKENS_PER_PROMPT,
        "first_token_matches": first,
        "aligned_token_matches": aligned,
        "aligned_token_total": len(a) * TOKENS_PER_PROMPT,
        "required_first_token_matches": len(TRAIN_GATE_INDICES),
        "required_aligned_token_matches": MIN_ALIGNED_MATCHES,
        "passed": first == len(TRAIN_GATE_INDICES) and aligned >= MIN_ALIGNED_MATCHES,
    }


def _server_tokens(server: LlamaServer, prompts: list[str]) -> list[list[int]]:
    records = []
    for offset, prompt in enumerate(prompts):
        response = server.complete(
            prompt,
            n_predict=TOKENS_PER_PROMPT,
            ignore_eos=True,
            seed=20260908 + offset,
        )
        tokens = [int(token) for token in response.get("tokens", [])]
        if len(tokens) != TOKENS_PER_PROMPT:
            raise RuntimeError(
                f"Expected {TOKENS_PER_PROMPT} returned tokens, found {len(tokens)}"
            )
        records.append(tokens)
    return records


def _server_first_token_logprobs(
    server: LlamaServer, prompts: list[str]
) -> list[dict[int, float]]:
    records = []
    for offset, prompt in enumerate(prompts):
        response = server.complete(
            prompt,
            n_predict=1,
            ignore_eos=True,
            seed=20260918 + offset,
            temperature=-1.0,
            n_probs=LOGPROB_TOP_K,
        )
        probability_rows = response.get("probs")
        if probability_rows is None:
            probability_rows = response.get("completion_probabilities")
        if not isinstance(probability_rows, list) or len(probability_rows) != 1:
            raise RuntimeError("llama-server did not return one log-probability row")
        top = probability_rows[0].get("top_logprobs")
        if not isinstance(top, list) or len(top) < LOGPROB_TOP_K:
            raise RuntimeError("llama-server did not return the frozen top-8 log-probs")
        record = {int(item["id"]): float(item["logprob"]) for item in top}
        if len(record) != LOGPROB_TOP_K:
            raise RuntimeError("llama-server returned duplicate top-8 token IDs")
        records.append(record)
    return records


def conversion_gate(model_key: str, llama_cpp_dir: Path, *, gpu: int) -> dict:
    """Compare tokenizer identity and fixed HF/GGUF FP16 train continuations."""
    import torch
    from transformers import AutoModelForCausalLM

    fp16_manifest_path = artifact_manifest_path(model_key, "fp16")
    if not fp16_manifest_path.is_file():
        raise RuntimeError(f"Audit GGUF-FP16 before conversion gate: {fp16_manifest_path}")
    fp16_manifest = json.loads(fp16_manifest_path.read_text(encoding="utf-8"))
    if fp16_manifest.get("gate_passed") is not True:
        raise RuntimeError(f"GGUF-FP16 artifact gate failed: {fp16_manifest_path}")
    prompts, identities, tokenizer = _gate_prompts(model_key)
    binaries = binary_paths(llama_cpp_dir)
    gguf_tokens = []
    server_tokenizations = []
    log_path = STATUS_DIR / "gate_logs" / f"{model_key}__conversion_server.jsonl"
    with LlamaServer(
        binaries["server"],
        artifact_path(model_key, "fp16"),
        log_path,
        gpu=gpu,
        slots=1,
        cuda=True,
    ) as server:
        server_tokenizations = [server.tokenize(prompt) for prompt in prompts]
        gguf_tokens = _server_tokens(server, prompts)
        gguf_logprobs = _server_first_token_logprobs(server, prompts)
    server_resources = server.resource_record()

    hf_tokenizations = [
        [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
        for prompt in prompts
    ]
    tokenizer_matches = [
        left == right for left, right in zip(hf_tokenizations, server_tokenizations)
    ]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SPECS[model_key]["path"],
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(f"cuda:{gpu}")
    model.eval()
    hf_tokens = []
    hf_logprobs = []
    try:
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(
                model.device
            )
            input_length = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                first_token_logprobs = torch.log_softmax(
                    model(**encoded).logits[0, -1].float(), dim=-1
                )
                values, indices = torch.topk(
                    first_token_logprobs, k=LOGPROB_TOP_K
                )
                hf_logprobs.append(
                    {
                        int(token): float(value)
                        for token, value in zip(
                            indices.detach().cpu().tolist(),
                            values.detach().cpu().tolist(),
                        )
                    }
                )
                output = model.generate(
                    **encoded,
                    do_sample=False,
                    min_new_tokens=TOKENS_PER_PROMPT,
                    max_new_tokens=TOKENS_PER_PROMPT,
                    pad_token_id=(
                        tokenizer.pad_token_id
                        if tokenizer.pad_token_id is not None
                        else tokenizer.eos_token_id
                    ),
                    eos_token_id=tokenizer.eos_token_id,
                )
            hf_tokens.append(
                [int(token) for token in output[0, input_length:].detach().cpu().tolist()]
            )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    agreement = _continuation_agreement(hf_tokens, gguf_tokens)
    logprob_rows = []
    for hf_row, gguf_row in zip(hf_logprobs, gguf_logprobs):
        common = sorted(set(hf_row) & set(gguf_row))
        max_error = (
            max(abs(hf_row[token] - gguf_row[token]) for token in common)
            if common
            else None
        )
        logprob_rows.append(
            {
                "hf_top_logprobs": [
                    {"id": token, "logprob": value}
                    for token, value in hf_row.items()
                ],
                "gguf_top_logprobs": [
                    {"id": token, "logprob": value}
                    for token, value in gguf_row.items()
                ],
                "common_token_ids": common,
                "top_k_overlap": len(common),
                "max_common_logprob_abs_error": max_error,
                "passed": (
                    len(common) >= MIN_LOGPROB_TOP_K_OVERLAP
                    and max_error is not None
                    and max_error <= MAX_COMMON_LOGPROB_ABS_ERROR
                ),
            }
        )
    logprob_passed = len(logprob_rows) == len(prompts) and all(
        row["passed"] for row in logprob_rows
    )
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "gate": "conversion",
        "model_key": model_key,
        "test_data_used": False,
        "train_prompts": identities,
        "tokenizer_exact_matches": sum(tokenizer_matches),
        "tokenizer_total": len(tokenizer_matches),
        "tokenizer_passed": all(tokenizer_matches),
        "continuation": agreement,
        "first_token_logprob_check": {
            "top_k": LOGPROB_TOP_K,
            "required_overlap_per_prompt": MIN_LOGPROB_TOP_K_OVERLAP,
            "maximum_common_logprob_absolute_error": MAX_COMMON_LOGPROB_ABS_ERROR,
            "rows": logprob_rows,
            "passed": logprob_passed,
        },
        "fp16_artifact_sha256": sha256_file(artifact_path(model_key, "fp16")),
        "fp16_manifest_sha256": sha256_file(fp16_manifest_path),
        "server_binary_sha256": sha256_file(binaries["server"]),
        "server_resources": server_resources,
        "gate_passed": (
            all(tokenizer_matches) and agreement["passed"] and logprob_passed
        ),
    }
    path = STATUS_DIR / "gates" / model_key / "conversion.json"
    atomic_write_json(path, record)
    if not record["gate_passed"]:
        raise RuntimeError(f"Conversion gate failed; inspect {path}")
    return record


def run_official_backend_tests(llama_cpp_dir: Path) -> dict:
    root = llama_cpp_dir.resolve()
    build_dir = root / "build"
    commit_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    commit = commit_result.stdout.strip()
    binaries = binary_paths(root)
    command = [
        "ctest",
        "--test-dir",
        str(build_dir),
        "-R",
        "test-quantize-fns|test-backend-ops",
        "--output-on-failure",
    ]
    result = subprocess.run(command, text=True, capture_output=True)
    combined = f"{result.stdout}\n{result.stderr}"
    total_match = re.search(r"out of\s+(\d+)", combined, flags=re.IGNORECASE)
    tests_run = int(total_match.group(1)) if total_match else 0
    passed = (
        result.returncode == 0
        and tests_run > 0
        and "No tests were found" not in combined
        and "100% tests passed" in combined
        and commit_result.returncode == 0
        and commit == LLAMA_CPP_COMMIT
    )
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "tests_run": tests_run,
        "llama_cpp_commit": commit,
        "binary_sha256": {
            name: sha256_file(path) for name, path in binaries.items()
        },
        "gate_passed": passed,
    }
    path = STATUS_DIR / "gates" / "official_backend_tests.json"
    atomic_write_json(path, record)
    if not passed:
        raise RuntimeError(f"Pinned llama.cpp backend tests failed; inspect {path}")
    return record


def packed_backend_gate(
    model_key: str, method: str, llama_cpp_dir: Path, *, gpu: int
) -> dict:
    official_path = STATUS_DIR / "gates" / "official_backend_tests.json"
    official_record = (
        json.loads(official_path.read_text(encoding="utf-8"))
        if official_path.is_file()
        else {}
    )
    if (
        official_record.get("gate_passed") is not True
        or official_record.get("llama_cpp_commit") != LLAMA_CPP_COMMIT
    ):
        raise RuntimeError("Run and pass `backend-tests` before packed gates")
    artifact_record_path = artifact_manifest_path(model_key, method)
    if not artifact_record_path.is_file() or json.loads(
        artifact_record_path.read_text(encoding="utf-8")
    ).get("gate_passed") is not True:
        raise RuntimeError(f"Audit the artifact before packed gate: {artifact_record_path}")

    prompts, identities, _ = _gate_prompts(model_key)
    binaries = binary_paths(llama_cpp_dir)
    if official_record.get("binary_sha256", {}).get("server") != sha256_file(
        binaries["server"]
    ):
        raise RuntimeError("llama-server binary changed after official backend tests")
    artifact = artifact_path(model_key, method)
    with LlamaServer(
        binaries["server"],
        artifact,
        STATUS_DIR / "gate_logs" / f"{model_key}__{method}__cpu.jsonl",
        gpu=gpu,
        slots=1,
        cuda=False,
        startup_timeout=600,
    ) as cpu_server:
        cpu_tokens = _server_tokens(cpu_server, prompts)
    cpu_resources = cpu_server.resource_record()
    with LlamaServer(
        binaries["server"],
        artifact,
        STATUS_DIR / "gate_logs" / f"{model_key}__{method}__cuda.jsonl",
        gpu=gpu,
        slots=1,
        cuda=True,
    ) as cuda_server:
        cuda_tokens = _server_tokens(cuda_server, prompts)
    cuda_resources = cuda_server.resource_record()

    agreement = _continuation_agreement(cpu_tokens, cuda_tokens)
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "gate": "packed_backend",
        "model_key": model_key,
        "method": method,
        "test_data_used": False,
        "train_prompts": identities,
        "cpu_vs_cuda_continuation": agreement,
        "artifact_sha256": sha256_file(artifact),
        "server_binary_sha256": sha256_file(binaries["server"]),
        "official_backend_tests_sha256": sha256_file(official_path),
        "cpu_resources": cpu_resources,
        "cuda_resources": cuda_resources,
        "gate_passed": agreement["passed"],
    }
    path = STATUS_DIR / "gates" / model_key / f"packed__{method}.json"
    atomic_write_json(path, record)
    if not record["gate_passed"]:
        raise RuntimeError(f"Packed CPU/CUDA gate failed; inspect {path}")
    return record

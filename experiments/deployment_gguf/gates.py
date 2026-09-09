"""Pre-test conversion and packed-backend gates."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from experiments.deployment_gguf.artifacts import cmake_build_provenance
from experiments.deployment_gguf.llama_server import LlamaServer
from experiments.deployment_gguf.protocol import (
    LLAMA_CPP_COMMIT,
    BACKEND_TEST_TIMEOUT_SECONDS,
    MODEL_SPECS,
    PROTOCOL_VERSION,
    STATUS_DIR,
    artifact_manifest_path,
    artifact_path,
    atomic_write_json,
    binary_paths,
    conversion_gate_policy_sha256,
    sha256_file,
)
from experiments.deployment_gguf.quality import load_prompts


TRAIN_GATE_INDICES = tuple(range(5, 13))
TOKENS_PER_PROMPT = 16
MIN_ALIGNED_MATCHES = 126
LOGPROB_TOP_K = 8
MIN_LOGPROB_TOP_K_OVERLAP = 7
MAX_COMMON_LOGPROB_ABS_ERROR = 0.05
MAX_CRITICAL_LOGPROB_GAP_ERROR = 2 * MAX_COMMON_LOGPROB_ABS_ERROR
NEAR_TIE_MAX_MARGIN = MAX_CRITICAL_LOGPROB_GAP_ERROR


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


def _server_tokens(
    server: LlamaServer, prompts: list[str | list[int]]
) -> list[list[int]]:
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


def _server_next_token_logprobs(
    server: LlamaServer, prompt: list[int], *, seed: int
) -> dict[int, float]:
    response = server.complete(
        prompt,
        n_predict=1,
        ignore_eos=False,
        seed=seed,
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
    return record


def _reference_logprob_agreement(
    hf_row: dict[int, float], gguf_row: dict[int, float]
) -> dict:
    if len(hf_row) != LOGPROB_TOP_K or len(gguf_row) != LOGPROB_TOP_K:
        raise RuntimeError("Reference-logit gate requires two complete top-8 rows")
    hf_ranked = sorted(hf_row, key=hf_row.get, reverse=True)
    gguf_ranked = sorted(gguf_row, key=gguf_row.get, reverse=True)
    common = sorted(set(hf_row) & set(gguf_row))
    common_errors = {
        token: abs(hf_row[token] - gguf_row[token]) for token in common
    }
    hf_top1, gguf_top1 = hf_ranked[0], gguf_ranked[0]
    # The union of both top-two sets is the smallest set that validates the
    # winning decision and its local margin without letting low-ranked tail
    # noise veto an otherwise faithful conversion.
    critical = set(hf_ranked[:2]) | set(gguf_ranked[:2])
    critical_present = critical.issubset(common_errors)
    max_critical_error = (
        max(common_errors[token] for token in critical)
        if critical_present
        else None
    )
    critical_gap_errors = (
        {
            token: abs(
                (hf_row[hf_top1] - hf_row[token])
                - (gguf_row[gguf_top1] - gguf_row[token])
            )
            for token in critical
        }
        if critical_present
        else {}
    )
    max_critical_gap_error = (
        max(critical_gap_errors.values()) if critical_gap_errors else None
    )
    hf_margin = hf_row[hf_ranked[0]] - hf_row[hf_ranked[1]]
    gguf_margin = gguf_row[gguf_ranked[0]] - gguf_row[gguf_ranked[1]]
    same_top1 = hf_top1 == gguf_top1
    explained_near_tie = (
        not same_top1
        and hf_top1 in gguf_ranked[:2]
        and gguf_top1 in hf_ranked[:2]
        and hf_margin <= NEAR_TIE_MAX_MARGIN
        and gguf_margin <= NEAR_TIE_MAX_MARGIN
    )
    passed = (
        len(common) >= MIN_LOGPROB_TOP_K_OVERLAP
        and max_critical_gap_error is not None
        and max_critical_gap_error <= MAX_CRITICAL_LOGPROB_GAP_ERROR
        and (same_top1 or explained_near_tie)
    )
    return {
        "hf_top_logprobs": [
            {"id": token, "logprob": hf_row[token]} for token in hf_ranked
        ],
        "gguf_top_logprobs": [
            {"id": token, "logprob": gguf_row[token]} for token in gguf_ranked
        ],
        "common_token_ids": common,
        "top_k_overlap": len(common),
        "max_common_logprob_abs_error": (
            max(common_errors.values()) if common_errors else None
        ),
        "critical_token_ids": sorted(critical),
        "max_critical_logprob_abs_error": max_critical_error,
        "critical_logprob_gap_abs_errors": [
            {"id": token, "abs_error": critical_gap_errors[token]}
            for token in sorted(critical_gap_errors)
        ],
        "max_critical_logprob_gap_abs_error": max_critical_gap_error,
        "hf_top1_id": hf_top1,
        "gguf_top1_id": gguf_top1,
        "hf_top1_margin": hf_margin,
        "gguf_top1_margin": gguf_margin,
        "same_top1": same_top1,
        "explained_near_tie": explained_near_tie,
        "passed": passed,
    }


def _archive_previous_gate(path: Path, archive_directory: str) -> None:
    """Preserve the last immutable gate record before publishing a replacement."""
    if not path.is_file():
        return
    previous = json.loads(path.read_text(encoding="utf-8"))
    attempt_id = previous.get("attempt_id")
    if not isinstance(attempt_id, str) or not re.fullmatch(
        r"\d{8}T\d{12}Z", attempt_id
    ):
        attempt_id = f"legacy__{sha256_file(path)[:16]}"
    archive = path.parent / archive_directory / f"{attempt_id}.json"
    if archive.is_file():
        if json.loads(archive.read_text(encoding="utf-8")) != previous:
            raise RuntimeError(f"Gate history collision: {archive}")
        return
    atomic_write_json(archive, previous)


def conversion_gate(model_key: str, llama_cpp_dir: Path, *, gpu: int) -> dict:
    """Compare tokenizer identity and teacher-forced HF/GGUF FP16 logits."""
    import torch
    from transformers import AutoModelForCausalLM

    fp16_manifest_path = artifact_manifest_path(model_key, "fp16")
    if not fp16_manifest_path.is_file():
        raise RuntimeError(f"Audit GGUF-FP16 before conversion gate: {fp16_manifest_path}")
    fp16_manifest = json.loads(fp16_manifest_path.read_text(encoding="utf-8"))
    if fp16_manifest.get("gate_passed") is not True:
        raise RuntimeError(f"GGUF-FP16 artifact gate failed: {fp16_manifest_path}")
    path = STATUS_DIR / "gates" / model_key / "conversion.json"
    _archive_previous_gate(path, "conversion_attempts")
    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    prompts, identities, tokenizer = _gate_prompts(model_key)
    if len(prompts) != len(TRAIN_GATE_INDICES) or len(identities) != len(prompts):
        raise RuntimeError("Conversion gate did not resolve all frozen train prompts")
    binaries = binary_paths(llama_cpp_dir)
    hf_tokenizations = [
        [int(token) for token in tokenizer.encode(prompt, add_special_tokens=False)]
        for prompt in prompts
    ]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_SPECS[model_key]["path"],
        local_files_only=True,
        torch_dtype=torch.float16,
    ).to(f"cuda:{gpu}")
    model.eval()
    hf_effective_inputs = []
    hf_tokens = []
    hf_teacher_logprobs = []
    try:
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(
                model.device
            )
            input_length = int(encoded["input_ids"].shape[1])
            hf_effective_inputs.append(
                [int(token) for token in encoded["input_ids"][0].detach().cpu().tolist()]
            )
            with torch.inference_mode():
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
                reference_tokens = [
                    int(token)
                    for token in output[0, input_length:].detach().cpu().tolist()
                ]
                if len(reference_tokens) != TOKENS_PER_PROMPT:
                    raise RuntimeError(
                        "HF reference generation did not return the frozen token count"
                    )
                teacher_input_ids = output[:, :-1]
                teacher_logits = model(
                    input_ids=teacher_input_ids,
                    attention_mask=torch.ones_like(teacher_input_ids),
                ).logits[0]
            hf_tokens.append(reference_tokens)
            prompt_rows = []
            for step in range(TOKENS_PER_PROMPT):
                next_logprobs = torch.log_softmax(
                    teacher_logits[input_length - 1 + step].float(), dim=-1
                )
                values, indices = torch.topk(next_logprobs, k=LOGPROB_TOP_K)
                prompt_rows.append(
                    {
                        int(token): float(value)
                        for token, value in zip(
                            indices.detach().cpu().tolist(),
                            values.detach().cpu().tolist(),
                        )
                    }
                )
            hf_teacher_logprobs.append(prompt_rows)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_path = (
        STATUS_DIR
        / "gate_logs"
        / "conversion"
        / model_key
        / f"{attempt_id}.jsonl"
    )
    gguf_teacher_logprobs = []
    with LlamaServer(
        binaries["server"],
        artifact_path(model_key, "fp16"),
        log_path,
        gpu=gpu,
        slots=1,
        cuda=True,
    ) as server:
        server_tokenizations = [server.tokenize(prompt) for prompt in prompts]
        gguf_tokens = _server_tokens(server, hf_effective_inputs)
        for prompt_offset, (input_ids, reference_tokens) in enumerate(
            zip(hf_effective_inputs, hf_tokens)
        ):
            prompt_rows = []
            for step in range(TOKENS_PER_PROMPT):
                history = input_ids + reference_tokens[:step]
                prompt_rows.append(
                    _server_next_token_logprobs(
                        server,
                        history,
                        seed=20260918 + prompt_offset * TOKENS_PER_PROMPT + step,
                    )
                )
            gguf_teacher_logprobs.append(prompt_rows)
    server_resources = server.resource_record()

    if not (
        len(server_tokenizations)
        == len(gguf_tokens)
        == len(gguf_teacher_logprobs)
        == len(prompts)
    ):
        raise RuntimeError("GGUF conversion gate returned incomplete prompt records")
    if any(len(rows) != TOKENS_PER_PROMPT for rows in gguf_teacher_logprobs):
        raise RuntimeError("GGUF conversion gate returned incomplete position records")

    tokenizer_matches = [
        left == right for left, right in zip(hf_tokenizations, server_tokenizations)
    ]
    for identity, hf_tokens_raw, gguf_tokens_raw, input_ids in zip(
        identities,
        hf_tokenizations,
        server_tokenizations,
        hf_effective_inputs,
    ):
        identity["hf_prompt_token_ids"] = hf_tokens_raw
        identity["gguf_prompt_token_ids"] = gguf_tokens_raw
        identity["effective_input_token_ids"] = input_ids
        identity["effective_input_token_count"] = len(input_ids)
        identity["effective_input_tokens_sha256"] = hashlib.sha256(
            json.dumps(input_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    free_running = _continuation_agreement(hf_tokens, gguf_tokens)
    teacher_rows = []
    for prompt_offset, (hf_prompt_rows, gguf_prompt_rows) in enumerate(
        zip(hf_teacher_logprobs, gguf_teacher_logprobs)
    ):
        for step, (hf_row, gguf_row) in enumerate(
            zip(hf_prompt_rows, gguf_prompt_rows)
        ):
            row = _reference_logprob_agreement(hf_row, gguf_row)
            row["train_index"] = identities[prompt_offset]["train_index"]
            row["continuation_step"] = step
            teacher_rows.append(row)
    teacher_passed = (
        len(teacher_rows) == len(prompts) * TOKENS_PER_PROMPT
        and all(row["passed"] for row in teacher_rows)
    )
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "attempt_id": attempt_id,
        "gate": "conversion",
        "conversion_gate_policy_sha256": conversion_gate_policy_sha256(),
        "model_key": model_key,
        "test_data_used": False,
        "train_prompts": identities,
        "tokenizer_exact_matches": sum(tokenizer_matches),
        "tokenizer_total": len(tokenizer_matches),
        "tokenizer_passed": all(tokenizer_matches),
        "free_running_continuation_diagnostic": {
            **free_running,
            "gate_role": "diagnostic only; divergence can amplify a near-tied decision",
        },
        "teacher_forced_reference_logprob_check": {
            "top_k": LOGPROB_TOP_K,
            "required_overlap_per_prompt": MIN_LOGPROB_TOP_K_OVERLAP,
            "single_token_logprob_absolute_error_diagnostic_reference": (
                MAX_COMMON_LOGPROB_ABS_ERROR
            ),
            "maximum_critical_logprob_gap_absolute_error": (
                MAX_CRITICAL_LOGPROB_GAP_ERROR
            ),
            "near_tie_max_top1_margin": NEAR_TIE_MAX_MARGIN,
            "conditioning": (
                "identical HF-greedy reference history supplied as token IDs "
                "to both backends"
            ),
            "positions": len(teacher_rows),
            "rows": teacher_rows,
            "passed": teacher_passed,
        },
        "fp16_artifact_sha256": sha256_file(artifact_path(model_key, "fp16")),
        "fp16_manifest_sha256": sha256_file(fp16_manifest_path),
        "server_binary_sha256": sha256_file(binaries["server"]),
        "server_resources": server_resources,
        "gate_passed": (
            all(tokenizer_matches) and teacher_passed
        ),
    }
    atomic_write_json(path, record)
    if not record["gate_passed"]:
        raise RuntimeError(f"Conversion gate failed; inspect {path}")
    return record


def _run_logged_backend_case(
    command: list[str], log_path: Path, *, gpu: int | None
) -> dict:
    env = os.environ.copy()
    env.pop("GGML_CUDA_DISABLE_GRAPHS", None)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    timed_out = False
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            env=env,
            timeout=BACKEND_TEST_TIMEOUT_SECONDS,
        )
        returncode = result.returncode
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = None
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        output = stdout + stderr + "\nTIMEOUT\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    failure_marker = re.search(r"(?:^|\s)FAIL(?:\s|$)|compare failed", output)
    return {
        "command": command,
        "physical_gpu": gpu,
        "cuda_visible_devices": str(gpu) if gpu is not None else None,
        "cuda_device_order": "PCI_BUS_ID",
        "cuda_graph_disable_variable": "unset",
        "returncode": returncode,
        "timeout_seconds": BACKEND_TEST_TIMEOUT_SECONDS,
        "timed_out": timed_out,
        "output_bytes": len(output.encode("utf-8")),
        "log_path": str(log_path),
        "log_sha256": sha256_file(log_path),
        "failure_marker_found": failure_marker is not None,
        "passed": (
            returncode == 0
            and bool(output.strip())
            and failure_marker is None
        ),
    }


def _archive_previous_backend_gate(path: Path) -> None:
    _archive_previous_gate(path, "official_backend_test_attempts")


def run_official_backend_tests(llama_cpp_dir: Path) -> dict:
    root = llama_cpp_dir.resolve()
    build_dir = root / "build"
    path = STATUS_DIR / "gates" / "official_backend_tests.json"
    _archive_previous_backend_gate(path)
    commit_result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    commit = commit_result.stdout.strip()
    cmake_provenance = cmake_build_provenance(root)
    binaries = binary_paths(root)
    quantize_binary = build_dir / "bin" / "test-quantize-fns"
    missing = [str(quantize_binary)] if not quantize_binary.is_file() else []
    if missing:
        raise FileNotFoundError(f"Missing pinned backend test binaries: {missing}")

    attempt_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = STATUS_DIR / "gate_logs" / "official_backend_tests" / attempt_id
    quantize = _run_logged_backend_case(
        [str(quantize_binary)], log_dir / "quantize_fns.log", gpu=None
    )
    passed = (
        commit_result.returncode == 0
        and commit == LLAMA_CPP_COMMIT
        and quantize["passed"]
    )
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "attempt_id": attempt_id,
        "llama_cpp_commit": commit,
        "cmake_build_provenance": cmake_provenance,
        "generic_backend_ops": {
            "executed": False,
            "gate_role": "none",
            "reason": (
                "The upstream randomized generic-op suite is retained only in "
                "historical qualification logs; deployment eligibility is tested "
                "on every actual packed artifact by the mandatory CPU-versus-CUDA "
                "continuation gate."
            ),
        },
        "quantize_fns": quantize,
        "strict_checks_required": 1,
        "strict_checks_passed": int(quantize["passed"]),
        "binary_sha256": {
            name: sha256_file(path) for name, path in binaries.items()
        },
        "test_binary_sha256": {
            "quantize_fns": sha256_file(quantize_binary),
        },
        "gate_passed": passed,
    }
    attempt_path = (
        STATUS_DIR / "gates" / "official_backend_test_attempts" / f"{attempt_id}.json"
    )
    atomic_write_json(attempt_path, record)
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
    conversion_path = STATUS_DIR / "gates" / model_key / "conversion.json"
    conversion_record = (
        json.loads(conversion_path.read_text(encoding="utf-8"))
        if conversion_path.is_file()
        else {}
    )
    if (
        conversion_record.get("gate_passed") is not True
        or conversion_record.get("model_key") != model_key
        or conversion_record.get("conversion_gate_policy_sha256")
        != conversion_gate_policy_sha256()
    ):
        raise RuntimeError("Run and pass `conversion-gate` before packed gates")
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
    if conversion_record.get("server_binary_sha256") != sha256_file(
        binaries["server"]
    ):
        raise RuntimeError("llama-server binary changed after conversion gate")
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

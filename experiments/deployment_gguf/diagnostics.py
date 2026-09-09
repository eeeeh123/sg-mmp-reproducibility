"""Read-only gate diagnostics on saved CPU reference histories."""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from experiments.deployment_gguf.gates import (
    TRAIN_GATE_INDICES, TOKENS_PER_PROMPT, _packed_teacher_agreement,
    _teacher_forced_rows,
)
from experiments.deployment_gguf.llama_server import LlamaServer
from experiments.deployment_gguf.protocol import (
    STATUS_DIR, artifact_path, atomic_write_json, binary_paths,
    packed_gate_policy_sha256, sha256_file,
)


def _saved_reference(record):
    identities = record["train_prompts"]
    if [row["train_index"] for row in identities] != list(TRAIN_GATE_INDICES):
        raise RuntimeError("Saved gate does not contain the fixed train prompts")
    inputs = [row["effective_input_token_ids"] for row in identities]
    references = [row["cpu_token_ids"] for row in identities]
    if any(not row or any(type(t) is not int or t < 0 for t in row)
           for row in inputs + references):
        raise RuntimeError("Saved gate contains invalid token IDs")
    if any(len(row) != TOKENS_PER_PROMPT for row in references):
        raise RuntimeError("Saved reference continuation is incomplete")
    rows = record["teacher_forced_cpu_vs_cuda"]["rows"]
    expected = [(i, step) for i in TRAIN_GATE_INDICES for step in range(TOKENS_PER_PROMPT)]
    if [(r["train_index"], r["continuation_step"]) for r in rows] != expected:
        raise RuntimeError("Saved distribution positions are incomplete or reordered")
    cpu = []
    for offset in range(len(identities)):
        cpu.append([
            {entry["id"]: entry["logprob"] for entry in row["cpu_top_logprobs"]}
            for row in rows[offset * TOKENS_PER_PROMPT:(offset + 1) * TOKENS_PER_PROMPT]
        ])
    _packed_teacher_agreement(cpu, cpu, identities)  # Validate finite, complete rows.
    return identities, inputs, references, cpu


def _compare(left, right, identities):
    check = _packed_teacher_agreement(left, right, identities)
    rows = check["rows"]
    # The shared checker uses CPU/CUDA field names; relabel for CUDA/CUDA pairs.
    check["rows"] = [
        {key.replace("cpu_", "left_").replace("cuda_", "right_"): value
         for key, value in row.items()} for row in rows
    ]
    errors = [r["max_critical_logprob_gap_abs_error"] for r in rows
              if r["max_critical_logprob_gap_abs_error"] is not None]
    summary = {
        "positions": len(rows),
        "failed_under_existing_policy": sum(not r["passed"] for r in rows),
        "top1_disagreements": sum(not r["same_top1"] for r in rows),
        "overlap_below_7": sum(r["top_k_overlap"] < 7 for r in rows),
        "critical_candidates_missing": len(rows) - len(errors),
        "gap_error_median": statistics.median(errors) if errors else None,
        "gap_error_max": max(errors) if errors else None,
        "exact_top8_rows": sum(a == b for aa, bb in zip(left, right) for a, b in zip(aa, bb)),
    }
    return summary, check


def packed_diagnostic(model_key: str, method: str, llama_cpp_dir: Path, *, gpu: int):
    source = STATUS_DIR / "gates" / model_key / f"packed__{method}.json"
    source_hash = sha256_file(source)
    record = json.loads(source.read_text(encoding="utf-8"))
    binary = binary_paths(llama_cpp_dir)["server"]
    artifact = artifact_path(model_key, method)
    if (record.get("model_key") != model_key or record.get("method") != method
            or record.get("tokenizer_passed") is not True
            or record.get("packed_gate_policy_sha256") != packed_gate_policy_sha256()):
        raise RuntimeError("Diagnostic requires a current, tokenizer-matched packed record")
    if (record.get("artifact_sha256") != sha256_file(artifact)
            or record.get("server_binary_sha256") != sha256_file(binary)):
        raise RuntimeError("Artifact or server differs from the saved gate")
    identities, inputs, references, cpu = _saved_reference(record)
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = STATUS_DIR / "diagnostics" / "packed" / model_key / method / attempt
    report = {
        "diagnostic_only": True, "complete": False, "attempt_id": attempt,
        "model_key": model_key, "method": method, "gpu": gpu,
        "source_gate_sha256": source_hash,
        "source_gate": str(source), "artifact_sha256": record["artifact_sha256"],
        "server_binary_sha256": record["server_binary_sha256"],
        "packed_gate_policy_sha256": packed_gate_policy_sha256(),
        "scope": "same saved CPU token histories, one-step full-prefix re-prefill; not incremental KV-cache equivalence",
        "runs": {}, "comparisons": {},
    }
    atomic_write_json(output / "source_gate.json", record)
    atomic_write_json(output / "summary.json", report)
    distributions = {"saved_cpu": cpu}
    for label, fa in (("cuda_on_1", "on"), ("cuda_on_2", "on"), ("cuda_off", "off")):
        print(f"Running {label}: 128 saved-history positions", flush=True)
        with LlamaServer(binary, artifact, output / f"{label}.jsonl",
                         gpu=gpu, slots=1, cuda=True, flash_attn=fa) as server:
            rows = _teacher_forced_rows(server, inputs, references)
        distributions[label] = rows
        run = {"flash_attn": fa, "resources": server.resource_record(), "rows": rows}
        atomic_write_json(output / f"{label}.json", run)
        report["runs"][label] = {"file": str(output / f"{label}.json"), "flash_attn": fa}
        atomic_write_json(output / "summary.json", report)
    for left, right in (("saved_cpu", "cuda_on_1"), ("saved_cpu", "cuda_on_2"),
                        ("saved_cpu", "cuda_off"), ("cuda_on_1", "cuda_on_2"),
                        ("cuda_on_1", "cuda_off")):
        name = f"{left}__vs__{right}"
        summary, check = _compare(distributions[left], distributions[right], identities)
        check.update(left_backend=left, right_backend=right, diagnostic_only=True)
        atomic_write_json(output / f"{name}.json", check)
        report["comparisons"][name] = summary
    report["complete"] = True
    atomic_write_json(output / "summary.json", report)
    return {"output_directory": str(output), **report}

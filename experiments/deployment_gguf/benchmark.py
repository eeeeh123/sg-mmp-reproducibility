"""Process-blocked llama-bench and real streaming request benchmarks."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path

from experiments.deployment_gguf.llama_server import (
    GpuMemorySampler,
    LlamaServer,
    gpu_memory_mib,
    process_rss_mib,
)
from experiments.deployment_gguf.protocol import (
    BENCH_DIR,
    CONTEXT_TOKENS_PER_SLOT,
    CPU_THREADS,
    LLAMA_CPP_COMMIT,
    MICRO_PROMPT_TOKENS,
    PERFORMANCE_GENERATED_TOKENS,
    PHASE_MODELS,
    PILOT_REPETITIONS,
    PROTOCOL_VERSION,
    SERVICE_CONCURRENCY,
    STATUS_DIR,
    artifact_manifest_path,
    artifact_path,
    atomic_write_json,
    binary_paths,
    conversion_gate_policy_sha256,
    packed_gate_policy_sha256,
    sha256_file,
)
from experiments.deployment_gguf.quality import load_prompts


MAX_PRELAUNCH_GPU_MIB = 1024.0


def _validate_bench_json(parsed: object) -> list[dict]:
    if not isinstance(parsed, list) or not parsed or any(
        not isinstance(row, dict) for row in parsed
    ):
        raise RuntimeError("llama-bench JSON must be a non-empty array of records")
    for row in parsed:
        reported = str(row.get("build_commit", ""))
        if not reported or not LLAMA_CPP_COMMIT.startswith(reported):
            raise RuntimeError(
                f"llama-bench record is not from pinned commit {LLAMA_CPP_COMMIT}: "
                f"{reported!r}"
            )
    return parsed


def workload_path(model_key: str) -> Path:
    return BENCH_DIR / "workloads" / f"{model_key}.json"


def prepare_workload(model_key: str) -> dict:
    rows, prompts, tokenizer = load_prompts(model_key, "train")
    token_counts = [
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ]
    ordered = sorted(token_counts)
    median = int(statistics.median(ordered))
    representative_indices = sorted(
        range(len(prompts)), key=lambda index: (abs(token_counts[index] - median), index)
    )[:4]
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "model_key": model_key,
        "source": "GSM8K train only",
        "test_data_used": False,
        "train_rows": len(rows),
        "prompt_length_min": min(token_counts),
        "prompt_length_median": median,
        "prompt_length_max": max(token_counts),
        "representative_train_indices": representative_indices,
        "representative_prompt_token_counts_hf": [
            token_counts[index] for index in representative_indices
        ],
        "representative_prompt_sha256": [
            hashlib.sha256(prompts[index].encode()).hexdigest()
            for index in representative_indices
        ],
    }
    atomic_write_json(workload_path(model_key), record)
    return record


def _require_benchmark_gate(model_key: str, method: str) -> tuple[dict, dict]:
    conversion = STATUS_DIR / "gates" / model_key / "conversion.json"
    gate = STATUS_DIR / "gates" / model_key / f"packed__{method}.json"
    manifest = artifact_manifest_path(model_key, method)
    records = []
    for path in (conversion, gate, manifest):
        if not path.is_file():
            raise RuntimeError(f"Benchmark locked until gate passes: {path}")
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("gate_passed") is not True:
            raise RuntimeError(f"Benchmark locked until gate passes: {path}")
        records.append(record)
    conversion_record, gate_record, manifest_record = records
    if gate_record.get("packed_gate_policy_sha256") != packed_gate_policy_sha256():
        raise RuntimeError("Packed gate belongs to an obsolete gate policy")
    if conversion_record.get("model_key") != model_key:
        raise RuntimeError("Conversion gate belongs to another model")
    if (
        conversion_record.get("conversion_gate_policy_sha256")
        != conversion_gate_policy_sha256()
    ):
        raise RuntimeError("Conversion gate belongs to an obsolete gate policy")
    if (
        gate_record.get("model_key") != model_key
        or gate_record.get("method") != method
    ):
        raise RuntimeError("Packed gate belongs to another model or method")
    if (
        manifest_record.get("model_key") != model_key
        or manifest_record.get("method") != method
    ):
        raise RuntimeError("Artifact manifest belongs to another model or method")
    if gate_record.get("artifact_sha256") != manifest_record.get("artifact_sha256"):
        raise RuntimeError("Packed gate and artifact manifest hashes disagree")
    return manifest_record, gate_record


def _require_idle_gpu(gpu: int) -> float:
    used = gpu_memory_mib(gpu)
    if used is None:
        raise RuntimeError("Cannot measure GPU memory with nvidia-smi")
    if used > MAX_PRELAUNCH_GPU_MIB:
        raise RuntimeError(
            f"Timing GPU {gpu} already uses {used:.0f} MiB; frozen limit is "
            f"{MAX_PRELAUNCH_GPU_MIB:.0f} MiB. Stop other GPU work first."
        )
    return used


def _run_bench_process(
    command: list[str], raw_stdout: Path, raw_stderr: Path, *, gpu: int
) -> dict:
    raw_stdout.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    baseline = _require_idle_gpu(gpu)
    sampler = GpuMemorySampler(gpu)
    peak_rss = None
    started = time.perf_counter()
    process = None
    sampler_started = False
    with raw_stdout.open("w", encoding="utf-8") as stdout, raw_stderr.open(
        "w", encoding="utf-8"
    ) as stderr:
        try:
            process = subprocess.Popen(
                command, stdout=stdout, stderr=stderr, text=True, env=env
            )
            sampler.start()
            sampler_started = True
            while process.poll() is None:
                rss = process_rss_mib(process.pid)
                if rss is not None:
                    peak_rss = rss if peak_rss is None else max(peak_rss, rss)
                time.sleep(0.1)
            returncode = process.returncode
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if sampler_started:
                sampler.stop()
    elapsed = time.perf_counter() - started
    if returncode:
        raise RuntimeError(f"llama-bench failed; inspect {raw_stderr}")
    try:
        parsed = json.loads(raw_stdout.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"llama-bench did not emit valid JSON: {raw_stdout}") from exc
    parsed = _validate_bench_json(parsed)
    return {
        "command": command,
        "elapsed_seconds": elapsed,
        "baseline_gpu_memory_mib": baseline,
        "peak_gpu_memory_mib": max(sampler.values) if sampler.values else None,
        "peak_gpu_delta_mib": (
            None if not sampler.values else max(sampler.values) - baseline
        ),
        "peak_host_rss_mib": peak_rss,
        "raw_stdout": str(raw_stdout),
        "raw_stderr": str(raw_stderr),
        "llama_bench_json": parsed,
    }


def benchmark_micro(
    model_key: str,
    method: str,
    llama_cpp_dir: Path,
    *,
    gpu: int,
    block: int,
    phase: str,
    repetitions: int = PILOT_REPETITIONS,
) -> dict:
    artifact_record, _ = _require_benchmark_gate(model_key, method)
    if phase not in {"engineering", "value", "formal"}:
        raise ValueError(f"Unknown run phase {phase}")
    if model_key not in PHASE_MODELS[phase]:
        raise ValueError(f"{model_key} is not registered for {phase} phase")
    if block < 0 or repetitions <= 0:
        raise ValueError("block must be non-negative and repetitions positive")
    binaries = binary_paths(llama_cpp_dir)
    benchmark_binary_sha256 = sha256_file(binaries["bench"])
    output = BENCH_DIR / "blocks" / "micro" / phase / model_key / method / f"block_{block:03d}.json"
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        identity = {
            "protocol_version": PROTOCOL_VERSION,
            "model_key": model_key,
            "method": method,
            "run_phase": phase,
            "process_block": block,
            "within_process_repetitions": repetitions,
            "benchmark_binary_sha256": benchmark_binary_sha256,
            "artifact_sha256": artifact_record["artifact_sha256"],
            "complete": True,
        }
        if any(existing.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"Existing process block belongs to another artifact: {output}")
        print(f"[skip] immutable completed process block: {output}", flush=True)
        return existing
    workload = (
        json.loads(workload_path(model_key).read_text(encoding="utf-8"))
        if workload_path(model_key).is_file()
        else prepare_workload(model_key)
    )
    median = int(workload["prompt_length_median"])
    prompt_lengths = list(dict.fromkeys((*MICRO_PROMPT_TOKENS, median)))
    depths = list(dict.fromkeys((*MICRO_PROMPT_TOKENS, median)))
    base = [
        str(binaries["bench"]),
        "--model",
        str(artifact_path(model_key, method)),
        "--repetitions",
        str(repetitions),
        "--threads",
        str(CPU_THREADS),
        "--output",
        "json",
        "--n-gpu-layers",
        "all",
        "--device",
        "CUDA0",
        "--cache-type-k",
        "f16",
        "--cache-type-v",
        "f16",
        "--flash-attn",
        "on",
        "--load-mode",
        "none",
        "--lazy-mode",
        "off",
        "--batch-size",
        "512",
        "--ubatch-size",
        "512",
    ]
    pp_command = base + [
        "--n-prompt",
        ",".join(map(str, prompt_lengths)),
        "--n-gen",
        "0",
    ]
    tg_command = base + [
        "--n-prompt",
        "0",
        "--n-gen",
        str(PERFORMANCE_GENERATED_TOKENS),
        "--n-depth",
        ",".join(map(str, depths)),
    ]
    raw_dir = BENCH_DIR / "raw" / "micro" / phase / model_key / method
    pp = _run_bench_process(
        pp_command,
        raw_dir / f"block_{block:03d}__pp.json",
        raw_dir / f"block_{block:03d}__pp.stderr.log",
        gpu=gpu,
    )
    tg = _run_bench_process(
        tg_command,
        raw_dir / f"block_{block:03d}__tg.json",
        raw_dir / f"block_{block:03d}__tg.stderr.log",
        gpu=gpu,
    )
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark": "llama-bench backend compute only",
        "explicitly_excluded": ["tokenization", "sampling", "request scheduling"],
        "model_key": model_key,
        "method": method,
        "run_phase": phase,
        "process_block": block,
        "within_process_repetitions": repetitions,
        "benchmark_binary_sha256": benchmark_binary_sha256,
        "artifact_sha256": artifact_record["artifact_sha256"],
        "workload": workload,
        "prompt_processing": pp,
        "text_generation": tg,
        "complete": True,
    }
    atomic_write_json(output, record)
    return record


def _request_group(server: LlamaServer, prompts: list[str], seed_base: int) -> list[dict]:
    import threading

    barrier = threading.Barrier(len(prompts))

    def run(index: int, prompt: str):
        barrier.wait()
        return server.stream_complete(
            prompt,
            n_predict=PERFORMANCE_GENERATED_TOKENS,
            seed=seed_base + index,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = [pool.submit(run, index, prompt) for index, prompt in enumerate(prompts)]
        return [future.result() for future in futures]


def benchmark_service(
    model_key: str,
    method: str,
    llama_cpp_dir: Path,
    *,
    gpu: int,
    block: int,
    phase: str,
    repetitions: int = PILOT_REPETITIONS,
) -> dict:
    artifact_record, gate_record = _require_benchmark_gate(model_key, method)
    if phase not in {"engineering", "value", "formal"}:
        raise ValueError(f"Unknown run phase {phase}")
    if model_key not in PHASE_MODELS[phase]:
        raise ValueError(f"{model_key} is not registered for {phase} phase")
    if block < 0 or repetitions <= 0:
        raise ValueError("block must be non-negative and repetitions positive")
    binaries = binary_paths(llama_cpp_dir)
    server_binary_sha256 = sha256_file(binaries["server"])
    if gate_record.get("server_binary_sha256") != server_binary_sha256:
        raise RuntimeError(
            "Benchmark server binary differs from the binary that passed the packed gate"
        )
    output = BENCH_DIR / "blocks" / "service" / phase / model_key / method / f"block_{block:03d}.json"
    if output.is_file():
        existing = json.loads(output.read_text(encoding="utf-8"))
        identity = {
            "protocol_version": PROTOCOL_VERSION,
            "model_key": model_key,
            "method": method,
            "run_phase": phase,
            "process_block": block,
            "within_process_repetitions_per_concurrency": repetitions,
            "server_binary_sha256": server_binary_sha256,
            "artifact_sha256": artifact_record["artifact_sha256"],
            "complete": True,
        }
        if any(existing.get(key) != value for key, value in identity.items()):
            raise RuntimeError(f"Existing process block belongs to another artifact: {output}")
        print(f"[skip] immutable completed process block: {output}", flush=True)
        return existing
    workload = (
        json.loads(workload_path(model_key).read_text(encoding="utf-8"))
        if workload_path(model_key).is_file()
        else prepare_workload(model_key)
    )
    _, all_prompts, _ = load_prompts(model_key, "train")
    prompts = [all_prompts[index] for index in workload["representative_train_indices"]]
    baseline = _require_idle_gpu(gpu)
    order = list(SERVICE_CONCURRENCY) * repetitions
    random.Random(20260908 + block).shuffle(order)
    groups = []
    server_log = BENCH_DIR / "raw" / "service" / phase / model_key / method / f"block_{block:03d}.server.jsonl"
    server = LlamaServer(
        binaries["server"],
        artifact_path(model_key, method),
        server_log,
        gpu=gpu,
        slots=max(SERVICE_CONCURRENCY),
        cuda=True,
    )
    with server:
        server_prompt_token_counts = [len(server.tokenize(prompt)) for prompt in prompts]
        if any(
            count + PERFORMANCE_GENERATED_TOKENS > CONTEXT_TOKENS_PER_SLOT
            for count in server_prompt_token_counts
        ):
            raise RuntimeError("Representative service prompt exceeds the frozen slot context")
        server.complete(prompts[0], n_predict=16, ignore_eos=True, seed=20260908)
        for group_index, concurrency in enumerate(order):
            chosen = [prompts[index % len(prompts)] for index in range(concurrency)]
            requests = _request_group(
                server, chosen, 2026090800 + block * 1000 + group_index * 10
            )
            if any(
                row["tokens_predicted"] != PERFORMANCE_GENERATED_TOKENS
                or row["stop_type"] != "limit"
                for row in requests
            ):
                raise RuntimeError(
                    "Fixed-token service benchmark stopped before 128 generated tokens"
                )
            group_start = min(row["submitted_monotonic"] for row in requests)
            group_end = max(row["ended_monotonic"] for row in requests)
            if group_end <= group_start:
                raise RuntimeError("Non-positive concurrent request wall time")
            groups.append(
                {
                    "condition_order": group_index,
                    "concurrency": concurrency,
                    "requests": requests,
                    "aggregate_tokens": sum(row["tokens_predicted"] for row in requests),
                    "aggregate_wall_seconds": group_end - group_start,
                    "aggregate_tokens_per_second": sum(
                        row["tokens_predicted"] for row in requests
                    )
                    / (group_end - group_start),
                }
            )
    resource_record = server.resource_record()
    record = {
        "protocol_version": PROTOCOL_VERSION,
        "benchmark": "llama-server native streaming request",
        "model_key": model_key,
        "method": method,
        "run_phase": phase,
        "process_block": block,
        "within_process_repetitions_per_concurrency": repetitions,
        "server_binary_sha256": server_binary_sha256,
        "concurrency_order": order,
        "artifact_sha256": artifact_record["artifact_sha256"],
        "prelaunch_gpu_memory_mib": baseline,
        "workload": workload,
        "representative_prompt_token_counts_server": server_prompt_token_counts,
        "generation": {
            "tokens": PERFORMANCE_GENERATED_TOKENS,
            "ignore_eos": True,
            "cache_prompt": False,
            "quality_interpretation_forbidden": True,
        },
        "groups": groups,
        "resources": resource_record,
        "complete": True,
    }
    atomic_write_json(output, record)
    return record

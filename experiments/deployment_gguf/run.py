"""CLI for the stage-gated deployment-gguf-v1 extension."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.deployment_gguf.analyze import analyze_model
from experiments.deployment_gguf.artifacts import (
    build_imatrix,
    convert_fp16,
    prepare_calibration_corpus,
    quantize_artifact,
    require_llama_cpp,
)
from experiments.deployment_gguf.benchmark import (
    benchmark_micro,
    benchmark_service,
    prepare_workload,
)
from experiments.deployment_gguf.gates import (
    conversion_gate,
    packed_backend_gate,
    run_official_backend_tests,
)
from experiments.deployment_gguf.gguf_manifest import (
    audit_artifact,
    audit_model_set,
    manifest_set_path,
)
from experiments.deployment_gguf.protocol import (
    BENCH_DIR,
    METHODS,
    MODEL_SPECS,
    FORMAL_MIN_BLOCKS,
    PHASE_MODELS,
    PILOT_BLOCKS,
    PROTOCOL_VERSION,
    QUALITY_DIR,
    STATUS_DIR,
    OUT,
    atomic_write_json,
    conversion_gate_policy_sha256,
    protocol_lock,
    sha256_file,
)
from experiments.deployment_gguf.quality import evaluate_quality


def emit(value) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def readiness(stage: str) -> dict:
    errors = []
    preflight = STATUS_DIR.parent / "manifests" / "llama_cpp_preflight.json"
    if not preflight.is_file() or json.loads(
        preflight.read_text(encoding="utf-8")
    ).get("gate_passed") is not True:
        errors.append(f"missing or failed {preflight}")
    official = STATUS_DIR / "gates" / "official_backend_tests.json"
    if not official.is_file() or json.loads(
        official.read_text(encoding="utf-8")
    ).get("gate_passed") is not True:
        errors.append(f"missing or failed {official}")
    for model in PHASE_MODELS[stage]:
        model_set = manifest_set_path(model)
        if not model_set.is_file() or json.loads(
            model_set.read_text(encoding="utf-8")
        ).get("gate_passed") is not True:
            errors.append(f"missing or failed {model_set}")
        for method in METHODS:
            manifest = STATUS_DIR.parent / "manifests" / "artifacts" / model / f"{method}.json"
            gate = STATUS_DIR / "gates" / model / f"packed__{method}.json"
            for path in (manifest, gate):
                if not path.is_file():
                    errors.append(f"missing {path}")
                elif json.loads(path.read_text(encoding="utf-8")).get("gate_passed") is not True:
                    errors.append(f"failed {path}")
            process_blocks = FORMAL_MIN_BLOCKS if stage == "formal" else PILOT_BLOCKS
            for block in range(process_blocks):
                for kind in ("micro", "service"):
                    path = BENCH_DIR / "blocks" / kind / stage / model / method / f"block_{block:03d}.json"
                    if not path.is_file():
                        errors.append(f"missing {path}")
                    else:
                        block_record = json.loads(path.read_text(encoding="utf-8"))
                        if (
                            block_record.get("complete") is not True
                            or block_record.get("run_phase") != stage
                            or block_record.get("model_key") != model
                            or block_record.get("method") != method
                            or block_record.get("process_block") != block
                        ):
                            errors.append(f"invalid {path}")
            if stage != "engineering":
                summary = QUALITY_DIR / "summaries" / f"{model}__{method}.json"
                if not summary.is_file():
                    errors.append(f"missing {summary}")
                elif json.loads(summary.read_text(encoding="utf-8")).get("complete") is not True:
                    errors.append(f"incomplete {summary}")
                else:
                    summary_record = json.loads(summary.read_text(encoding="utf-8"))
                    sample = QUALITY_DIR / "samples" / f"{model}__{method}__gsm8k1319.jsonl"
                    if (
                        not sample.is_file()
                        or summary_record.get("sample_file_sha256") != sha256_file(sample)
                    ):
                        errors.append(f"stale quality summary {summary}")
        conversion = STATUS_DIR / "gates" / model / "conversion.json"
        conversion_record = (
            json.loads(conversion.read_text(encoding="utf-8"))
            if conversion.is_file()
            else {}
        )
        if (
            conversion_record.get("gate_passed") is not True
            or conversion_record.get("model_key") != model
            or conversion_record.get("conversion_gate_policy_sha256")
            != conversion_gate_policy_sha256()
        ):
            errors.append(f"missing or failed {conversion}")
        analysis = OUT / "analysis" / stage / f"{model}.json"
        if not analysis.is_file():
            errors.append(f"missing {analysis}")
        else:
            analysis_record = json.loads(analysis.read_text(encoding="utf-8"))
            if (
                analysis_record.get("protocol_version") != PROTOCOL_VERSION
                or analysis_record.get("model_key") != model
                or analysis_record.get("run_phase") != stage
                or analysis_record.get("complete") is not True
            ):
                errors.append(f"invalid {analysis}")
            if analysis_record.get("formal_precision", {}).get(
                "additional_blocks_required"
            ) is True:
                errors.append(
                    f"formal precision target not reached; add paired blocks: {analysis}"
                )
            registered_inputs = analysis_record.get("input_file_sha256", {})
            current_inputs = {
                str(path)
                for method in METHODS
                for kind in ("micro", "service")
                for path in (
                    BENCH_DIR / "blocks" / kind / stage / model / method
                ).glob("block_*.json")
            }
            current_inputs.update(
                str(STATUS_DIR.parent / "manifests" / "artifacts" / model / f"{method}.json")
                for method in METHODS
            )
            if stage != "engineering":
                current_inputs.update(
                    str(QUALITY_DIR / "summaries" / f"{model}__{method}.json")
                    for method in METHODS
                )
                current_inputs.update(
                    str(QUALITY_DIR / "samples" / f"{model}__{method}__gsm8k1319.jsonl")
                    for method in METHODS
                )
            if not isinstance(registered_inputs, dict) or set(
                registered_inputs
            ) != current_inputs:
                errors.append(f"stale analysis input set {analysis}")
                registered_inputs = {}
            for raw_path, registered_sha256 in registered_inputs.items():
                path = Path(raw_path)
                if not path.is_file() or sha256_file(path) != registered_sha256:
                    errors.append(f"stale analysis input {path}")
    record = {"stage": stage, "ready": not errors, "errors": errors}
    atomic_write_json(STATUS_DIR / f"readiness__{stage}.json", record)
    return record


def parser() -> argparse.ArgumentParser:
    model_choices = tuple(MODEL_SPECS)
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)

    sub.add_parser("show-protocol")
    p = sub.add_parser("preflight")
    p.add_argument("--llama-cpp-dir", type=Path, required=True)
    p = sub.add_parser("verify-backend")
    p.add_argument("--llama-cpp-dir", type=Path, required=True)
    p = sub.add_parser("prepare-calibration")
    p.add_argument("--model", choices=model_choices, required=True)
    p = sub.add_parser("convert")
    p.add_argument("--model", choices=model_choices, required=True)
    p.add_argument("--llama-cpp-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("imatrix")
    p.add_argument("--model", choices=model_choices, required=True)
    p.add_argument("--llama-cpp-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("quantize")
    p.add_argument("--model", choices=model_choices, required=True)
    p.add_argument("--method", choices=("q4", "q5", "sg"), required=True)
    p.add_argument("--llama-cpp-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("audit")
    p.add_argument("--model", choices=model_choices, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p = sub.add_parser("audit-set")
    p.add_argument("--model", choices=model_choices, required=True)
    p = sub.add_parser("backend-tests")
    p.add_argument("--llama-cpp-dir", type=Path, required=True)
    p = sub.add_parser("conversion-gate")
    p.add_argument("--model", choices=model_choices, required=True)
    p.add_argument("--llama-cpp-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p = sub.add_parser("packed-gate")
    p.add_argument("--model", choices=model_choices, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--llama-cpp-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p = sub.add_parser("prepare-workload")
    p.add_argument("--model", choices=model_choices, required=True)
    for name in ("benchmark-micro", "benchmark-service"):
        p = sub.add_parser(name)
        p.add_argument("--model", choices=model_choices, required=True)
        p.add_argument("--method", choices=METHODS, required=True)
        p.add_argument("--llama-cpp-dir", type=Path, required=True)
        p.add_argument("--gpu", type=int, required=True)
        p.add_argument("--block", type=int, required=True)
        p.add_argument("--run-phase", choices=PHASE_MODELS, required=True)
        p.add_argument("--repetitions", type=int, default=5)
    p = sub.add_parser("quality")
    p.add_argument("--model", choices=model_choices, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--llama-cpp-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p = sub.add_parser("analyze")
    p.add_argument("--model", choices=model_choices, required=True)
    p.add_argument("--run-phase", choices=PHASE_MODELS, required=True)
    p = sub.add_parser("readiness")
    p.add_argument("--stage", choices=PHASE_MODELS, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "show-protocol":
        result = protocol_lock()
    elif args.command == "preflight":
        result = require_llama_cpp(args.llama_cpp_dir, minimum_free_disk_gib=40)
    elif args.command == "verify-backend":
        result = require_llama_cpp(args.llama_cpp_dir)
    elif args.command == "prepare-calibration":
        result = prepare_calibration_corpus(args.model)
    elif args.command == "convert":
        result = convert_fp16(args.model, args.llama_cpp_dir, force=args.force)
    elif args.command == "imatrix":
        result = build_imatrix(args.model, args.llama_cpp_dir, gpu=args.gpu, force=args.force)
    elif args.command == "quantize":
        result = quantize_artifact(args.model, args.method, args.llama_cpp_dir, force=args.force)
    elif args.command == "audit":
        result = audit_artifact(args.model, args.method)
    elif args.command == "audit-set":
        result = audit_model_set(args.model)
    elif args.command == "backend-tests":
        result = run_official_backend_tests(args.llama_cpp_dir)
    elif args.command == "conversion-gate":
        result = conversion_gate(args.model, args.llama_cpp_dir, gpu=args.gpu)
    elif args.command == "packed-gate":
        result = packed_backend_gate(args.model, args.method, args.llama_cpp_dir, gpu=args.gpu)
    elif args.command == "prepare-workload":
        result = prepare_workload(args.model)
    elif args.command == "benchmark-micro":
        result = benchmark_micro(args.model, args.method, args.llama_cpp_dir, gpu=args.gpu, block=args.block, phase=args.run_phase, repetitions=args.repetitions)
    elif args.command == "benchmark-service":
        result = benchmark_service(args.model, args.method, args.llama_cpp_dir, gpu=args.gpu, block=args.block, phase=args.run_phase, repetitions=args.repetitions)
    elif args.command == "quality":
        result = evaluate_quality(args.model, args.method, args.llama_cpp_dir, gpu=args.gpu)
    elif args.command == "analyze":
        result = analyze_model(args.model, args.run_phase)
    elif args.command == "readiness":
        result = readiness(args.stage)
    else:
        raise AssertionError(args.command)
    emit(result)


if __name__ == "__main__":
    main()

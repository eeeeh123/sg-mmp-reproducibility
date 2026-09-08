"""Generate fail-fast, stage-separated server plans."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

from experiments.deployment_gguf.protocol import (
    FORMAL_MAX_BLOCKS,
    FORMAL_MIN_BLOCKS,
    METHODS,
    PHASE_MODELS,
    PILOT_BLOCKS,
)


MODULE = "experiments.deployment_gguf.run"


def quote(value: object) -> str:
    return shlex.quote(str(value))


def command(*parts: object) -> str:
    return " ".join(quote(part) for part in ("python", "-m", MODULE, *parts))


def balanced_methods(block: int) -> list[str]:
    methods = list(METHODS)
    offset = block % len(methods)
    rotated = methods[offset:] + methods[:offset]
    return list(reversed(rotated)) if (block // len(methods)) % 2 else rotated


def build_plan(
    stage: str, llama_cpp_dir: str, timing_gpu: int, quality_gpu: int
) -> str:
    if stage not in PHASE_MODELS:
        raise ValueError(stage)
    if timing_gpu == quality_gpu:
        raise ValueError("Timing and quality/gate GPUs must be distinct")
    models = PHASE_MODELS[stage]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "unset REVISION_FULL_ONLINE_STAGING",
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY",
        "unset CUDA_VISIBLE_DEVICES GGML_CUDA_DISABLE_GRAPHS",
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"LLAMA_CPP_DIR={quote(llama_cpp_dir)}",
        "export DEPLOYMENT_LLAMA_CPP_PYTHON=\"${LLAMA_CPP_DIR}-convert-venv/bin/python\"",
        command("preflight", "--llama-cpp-dir", "$LLAMA_CPP_DIR"),
        command("backend-tests", "--llama-cpp-dir", "$LLAMA_CPP_DIR"),
    ]
    # shlex deliberately quoted the variable above; replace only exact safe token.
    lines = [line.replace("'$LLAMA_CPP_DIR'", '"$LLAMA_CPP_DIR"') for line in lines]
    for model in models:
        lines.extend(
            [
                "",
                f"# {stage}: {model}",
                command("prepare-calibration", "--model", model),
                command("convert", "--model", model, "--llama-cpp-dir", "$LLAMA_CPP_DIR"),
                command(
                    "imatrix",
                    "--model",
                    model,
                    "--llama-cpp-dir",
                    "$LLAMA_CPP_DIR",
                    "--gpu",
                    quality_gpu,
                ),
            ]
        )
        for method in ("q4", "q5", "sg"):
            lines.append(
                command(
                    "quantize",
                    "--model",
                    model,
                    "--method",
                    method,
                    "--llama-cpp-dir",
                    "$LLAMA_CPP_DIR",
                )
            )
        lines.append(command("audit-set", "--model", model))
        lines.append(
            command(
                "conversion-gate",
                "--model",
                model,
                "--llama-cpp-dir",
                "$LLAMA_CPP_DIR",
                "--gpu",
                quality_gpu,
            )
        )
        for method in METHODS:
            lines.append(
                command(
                    "packed-gate",
                    "--model",
                    model,
                    "--method",
                    method,
                    "--llama-cpp-dir",
                    "$LLAMA_CPP_DIR",
                    "--gpu",
                    quality_gpu,
                )
            )
        lines.append(command("prepare-workload", "--model", model))
        process_blocks = FORMAL_MIN_BLOCKS if stage == "formal" else PILOT_BLOCKS
        for block in range(process_blocks):
            for method in balanced_methods(block):
                common = (
                    "--model",
                    model,
                    "--method",
                    method,
                    "--llama-cpp-dir",
                    "$LLAMA_CPP_DIR",
                    "--gpu",
                    timing_gpu,
                    "--block",
                    block,
                    "--run-phase",
                    stage,
                )
                lines.append(command("benchmark-micro", *common))
                lines.append(command("benchmark-service", *common))
    if stage != "engineering":
        lines.extend(
            [
                "",
                "# Test firewall: every registered model has passed all pre-test work.",
            ]
        )
        for model in models:
            for method in METHODS:
                lines.append(
                    command(
                        "quality",
                        "--model",
                        model,
                        "--method",
                        method,
                        "--llama-cpp-dir",
                        "$LLAMA_CPP_DIR",
                        "--gpu",
                        quality_gpu,
                    )
                )
    for model in models:
        lines.append(command("analyze", "--model", model, "--run-phase", stage))
    lines.append("")
    lines.append(command("readiness", "--stage", stage))
    return "\n".join(
        line.replace("'$LLAMA_CPP_DIR'", '"$LLAMA_CPP_DIR"') for line in lines
    ) + "\n"


def build_formal_extension_plan(
    llama_cpp_dir: str,
    timing_gpu: int,
    quality_gpu: int,
    start_block: int,
    end_block: int,
) -> str:
    if timing_gpu == quality_gpu:
        raise ValueError("Timing and quality/gate GPUs must be distinct")
    if not FORMAL_MIN_BLOCKS <= start_block < end_block <= FORMAL_MAX_BLOCKS:
        raise ValueError(
            f"Formal extension must satisfy {FORMAL_MIN_BLOCKS} <= start < end "
            f"<= {FORMAL_MAX_BLOCKS}"
        )
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "unset REVISION_FULL_ONLINE_STAGING",
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY",
        "unset CUDA_VISIBLE_DEVICES GGML_CUDA_DISABLE_GRAPHS",
        "export CUDA_DEVICE_ORDER=PCI_BUS_ID",
        f"LLAMA_CPP_DIR={quote(llama_cpp_dir)}",
        "export DEPLOYMENT_LLAMA_CPP_PYTHON=\"${LLAMA_CPP_DIR}-convert-venv/bin/python\"",
        command("verify-backend", "--llama-cpp-dir", "$LLAMA_CPP_DIR"),
    ]
    for model in PHASE_MODELS["formal"]:
        lines.extend(["", f"# formal extension: {model}"])
        for block in range(start_block, end_block):
            for method in balanced_methods(block):
                common = (
                    "--model",
                    model,
                    "--method",
                    method,
                    "--llama-cpp-dir",
                    "$LLAMA_CPP_DIR",
                    "--gpu",
                    timing_gpu,
                    "--block",
                    block,
                    "--run-phase",
                    "formal",
                )
                lines.append(command("benchmark-micro", *common))
                lines.append(command("benchmark-service", *common))
        lines.append(command("analyze", "--model", model, "--run-phase", "formal"))
    lines.extend(["", command("readiness", "--stage", "formal")])
    return "\n".join(
        line.replace("'$LLAMA_CPP_DIR'", '"$LLAMA_CPP_DIR"') for line in lines
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=PHASE_MODELS, required=True)
    parser.add_argument("--llama-cpp-dir", required=True)
    parser.add_argument("--timing-gpu", type=int, default=1)
    parser.add_argument("--quality-gpu", type=int, default=0)
    parser.add_argument("--start-block", type=int)
    parser.add_argument("--end-block", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.start_block is None and args.end_block is None:
        value = build_plan(
            args.stage, args.llama_cpp_dir, args.timing_gpu, args.quality_gpu
        )
    else:
        if args.stage != "formal" or args.start_block is None or args.end_block is None:
            parser.error(
                "--start-block and --end-block must be supplied together for --stage formal"
            )
        value = build_formal_extension_plan(
            args.llama_cpp_dir,
            args.timing_gpu,
            args.quality_gpu,
            args.start_block,
            args.end_block,
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value, encoding="utf-8", newline="\n")
        print(args.output)
    else:
        print(value, end="")


if __name__ == "__main__":
    main()

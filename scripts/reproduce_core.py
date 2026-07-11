"""Canonical entry point for the released SG-MMP reproduction workflow.

This wrapper deliberately separates checkpoint download, quantization, private
evaluation logs, and public-result verification.  It never downloads model
weights during evaluation and never copies private GSM8K prompts or generations
into the public ``data/processed`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_MODELS = ("qwen05", "qwen15", "smollm")


def run(command: list[str], dry_run: bool) -> None:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(f"[run] {printable}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def parse_models(value: str) -> list[str]:
    if value == "primary":
        return list(PRIMARY_MODELS)
    values = [item.strip() for item in value.split(",") if item.strip()]
    allowed = set(PRIMARY_MODELS)
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unsupported primary-model keys: {sorted(unknown)}")
    return values


def download_primary(dry_run: bool) -> None:
    run([sys.executable, "scripts/01_download_models.py"], dry_run)


def prepare_data(cache_dir: Path | None, dry_run: bool) -> None:
    command = [
        sys.executable,
        "-c",
        (
            "from datasets import load_dataset; "
            "load_dataset('wikitext', 'wikitext-2-raw-v1', split='train'" 
            + (f", cache_dir={str(cache_dir)!r}" if cache_dir else "")
            + "); "
            "load_dataset('openai/gsm8k', 'main', split='train'"
            + (f", cache_dir={str(cache_dir)!r}" if cache_dir else "")
            + "); "
            "load_dataset('openai/gsm8k', 'main', split='test'"
            + (f", cache_dir={str(cache_dir)!r}" if cache_dir else "")
            + "); print('Cached WikiText-2 and GSM8K')"
        ),
    ]
    run(command, dry_run)


def quantize(models: list[str], dry_run: bool) -> None:
    for model in models:
        if model == "qwen05":
            run(
                [
                    sys.executable,
                    "scripts/quantize.py",
                    "--model",
                    "Qwen2.5-0.5B",
                    "--method",
                    "gptq",
                ],
                dry_run,
            )
            run(
                [
                    sys.executable,
                    "experiments/exp07_mixed_ablation/run.py",
                    "--configs",
                    "config_b",
                    "--quantize-only",
                ],
                dry_run,
            )
        elif model == "qwen15":
            run(
                [
                    sys.executable,
                    "scripts/quantize.py",
                    "--model",
                    "Qwen2.5-1.5B",
                    "--method",
                    "gptq",
                ],
                dry_run,
            )
            run(
                [sys.executable, "experiments/exp12_qwen15b_config_b/run.py", "quantize_2a"],
                dry_run,
            )
        elif model == "smollm":
            run(
                [
                    sys.executable,
                    "experiments/fix_gsm8k_500/requantize_gptq_compact.py",
                    "--model",
                    "smollm",
                ],
                dry_run,
            )
            run([sys.executable, "experiments/exp13_smollm_config_b/run.py", "quantize"], dry_run)


def evaluate(
    models: list[str],
    cache_dir: Path | None,
    batch_size: int,
    max_new_tokens: int,
    offline: bool,
    force: bool,
    dry_run: bool,
) -> None:
    command = [
        sys.executable,
        "experiments/fix_gsm8k_500/direct_eval.py",
        "run",
        "--models",
        ",".join(models),
        "--methods",
        "fp16,gptq,sg",
        "--n",
        "500",
        "--batch-size",
        str(batch_size),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if cache_dir is not None:
        command.extend(["--dataset-cache-dir", str(cache_dir)])
    if offline:
        command.append("--offline")
    if force:
        command.append("--force")
    run(command, dry_run)


def analyze(dry_run: bool) -> None:
    run([sys.executable, "experiments/fix_gsm8k_500/direct_eval.py", "analyze", "--n", "500"], dry_run)


def figures(dry_run: bool) -> None:
    run([sys.executable, "scripts/generate_figures.py"], dry_run)
    run([sys.executable, "scripts/generate_concept_figures.py"], dry_run)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_public_release(dry_run: bool) -> None:
    checksum_file = ROOT / "SHA256SUMS"
    failures = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        candidate = ROOT / relative
        if not candidate.exists() or sha256(candidate) != expected:
            failures.append(relative)
    if failures:
        raise RuntimeError(f"Checksum validation failed: {failures}")
    print("[ok] SHA256SUMS validates every released source file", flush=True)

    audit_dir = ROOT / ".release-audit"
    audit_dir.mkdir(exist_ok=True)
    output = audit_dir / "recomputed_paired_stats.json"
    try:
        run(
            [
                sys.executable,
                "scripts/analyze_released_gsm8k500.py",
                "--output",
                str(output),
            ],
            dry_run,
        )
        if not dry_run:
            expected = (ROOT / "data/processed/gsm8k500/recomputed_paired_stats.json").read_bytes()
            if output.read_bytes() != expected:
                raise RuntimeError("Recomputed paired statistics do not match the released reference file.")
    finally:
        try:
            output.unlink(missing_ok=True)
        except PermissionError:
            # Some Windows desktop environments retain a brief handle after a
            # subprocess exits. The ignored audit file is harmless and must
            # not turn a successful verification into a false failure.
            print(f"[note] Could not remove ignored audit file: {output}", flush=True)
        try:
            audit_dir.rmdir()
        except OSError:
            pass
    print("[ok] Redacted outcomes reproduce the released paired statistics", flush=True)


def print_plan() -> None:
    print(
        "\n".join(
            [
                "Primary reproduction path:",
                "  1. python scripts/reproduce_core.py download-primary",
                "  2. python scripts/reproduce_core.py prepare-data",
                "  3. python scripts/reproduce_core.py quantize",
                "  4. python scripts/reproduce_core.py evaluate",
                "  5. python scripts/reproduce_core.py analyze",
                "  6. python scripts/reproduce_core.py figures",
                "",
                "Public-artifact verification without models or a GPU:",
                "  python scripts/reproduce_core.py verify-public",
            ]
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical SG-MMP reproduction workflow")
    parser.add_argument(
        "command",
        choices=[
            "download-primary",
            "prepare-data",
            "quantize",
            "evaluate",
            "analyze",
            "figures",
            "verify-public",
            "print-plan",
        ],
    )
    parser.add_argument("--models", default="primary", help="primary or qwen05,qwen15,smollm")
    parser.add_argument("--dataset-cache-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    models = parse_models(args.models)
    if args.command == "download-primary":
        download_primary(args.dry_run)
    elif args.command == "prepare-data":
        prepare_data(args.dataset_cache_dir, args.dry_run)
    elif args.command == "quantize":
        quantize(models, args.dry_run)
    elif args.command == "evaluate":
        evaluate(
            models,
            args.dataset_cache_dir,
            args.batch_size,
            args.max_new_tokens,
            args.offline,
            args.force,
            args.dry_run,
        )
    elif args.command == "analyze":
        analyze(args.dry_run)
    elif args.command == "figures":
        figures(args.dry_run)
    elif args.command == "verify-public":
        verify_public_release(args.dry_run)
    else:
        print_plan()


if __name__ == "__main__":
    main()

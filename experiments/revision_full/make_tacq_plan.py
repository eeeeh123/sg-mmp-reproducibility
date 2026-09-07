"""Print a fail-closed Shadow -> TaCQ server execution plan."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from experiments.revision_full.protocol import CALIB_SEEDS
from experiments.revision_full.tacq import OFFICIAL_SOURCE_COMMIT, TACQ_MODELS


def _registration(model: str, seed: int) -> str:
    return (
        "experiments/revision_full/outputs/external_baselines/"
        f"{model}__tacq__c{seed}.json"
    )


def shadow_commands(gpu: int = 0) -> list[str]:
    run = "python experiments/revision_full/run.py"
    shadow = "python experiments/revision_full/shadow_gate.py"
    result = [
        "python -m unittest discover -s experiments/revision_full -p 'test_*.py'",
        "python experiments/revision_full/server_preflight.py --expected-gpus 2 --concurrent-models 1",
        f"{shadow} prepare",
    ]
    for model in TACQ_MODELS:
        result.extend(
            [
                f"CUDA_VISIBLE_DEVICES={gpu} {run} build-bank --model {model} --calib-seed 41",
                f"{run} materialize --model {model} --calib-seed 41 --variant gptq_w4",
                f"{run} materialize --model {model} --calib-seed 41 --variant sg_mmp",
                f"CUDA_VISIBLE_DEVICES={gpu} {shadow} run --model {model} --variant gptq_w4",
                f"CUDA_VISIBLE_DEVICES={gpu} {shadow} run --model {model} --variant sg_mmp",
            ]
        )
    result.extend(
        [
            f"{shadow} verify",
            "python experiments/revision_full/readiness.py --stage shadow",
        ]
    )
    return result


def tacq_commands(gpu: int = 0) -> list[str]:
    run = "python experiments/revision_full/run.py"
    tacq = "python experiments/revision_full/tacq.py"
    result = [
        # This check is intentionally repeated at the TaCQ boundary.  The
        # TaCQ phase cannot start from a copied/stale or failed Shadow receipt.
        "python experiments/revision_full/server_preflight.py --expected-gpus 2 --concurrent-models 1",
        "python experiments/revision_full/readiness.py --stage shadow",
        f"{tacq} freeze --source-commit {OFFICIAL_SOURCE_COMMIT}",
    ]
    for model in TACQ_MODELS:
        result.extend(
            [
                f"{run} cleanup-state --model {model} --calib-seed 41 --variant gptq_w4",
                f"{run} cleanup-state --model {model} --calib-seed 41 --variant sg_mmp",
            ]
        )
    for model in TACQ_MODELS:
        registrations = [_registration(model, seed) for seed in CALIB_SEEDS]
        missing_registration = " || ".join(f"[[ ! -f {path} ]]" for path in registrations)
        result.append(
            f"if {missing_registration}; then "
            f"CUDA_VISIBLE_DEVICES={gpu} {tacq} capture-importance --model {model}; "
            "fi"
        )
        for seed in CALIB_SEEDS:
            registration = _registration(model, seed)
            # A registered seed is an atomic completed unit.  A missing record
            # means that every command below may resume its own validated
            # checkpoint; the bank is rebuilt if an earlier cleanup removed it.
            result.append(
                f"if [[ ! -f {registration} ]]; then "
                f"CUDA_VISIBLE_DEVICES={gpu} {run} build-bank --model {model} --calib-seed {seed}; "
                f"{tacq} build --model {model} --calib-seed {seed}; "
                f"CUDA_VISIBLE_DEVICES={gpu} {tacq} smoke --model {model} --calib-seed {seed}; "
                f"CUDA_VISIBLE_DEVICES={gpu} {tacq} evaluate --model {model} --calib-seed {seed}; "
                "fi"
            )
            result.extend(
                (
                    f"{tacq} cleanup --model {model} --calib-seed {seed}",
                    f"{run} cleanup-bank --model {model} --calib-seed {seed}",
                )
            )
        result.append(f"{tacq} cleanup --model {model}")
    result.extend(
        [
            "python experiments/revision_full/analyze.py",
            "python experiments/revision_full/readiness.py --stage tacq",
            "python experiments/revision_full/readiness.py --stage resubmission",
        ]
    )
    return result


def commands(gpu: int = 0, phase: str = "all") -> list[str]:
    if phase == "shadow":
        return shadow_commands(gpu)
    if phase == "tacq":
        return tacq_commands(gpu)
    if phase == "all":
        return shadow_commands(gpu) + tacq_commands(gpu)
    raise ValueError(f"unknown phase: {phase}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--phase", choices=("shadow", "tacq", "all"), default="all")
    args = parser.parse_args()
    print("#!/usr/bin/env bash")
    print("set -euo pipefail")
    for command in commands(args.gpu, args.phase):
        print(command)


if __name__ == "__main__":
    main()

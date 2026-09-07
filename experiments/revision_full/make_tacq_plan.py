"""Print the fail-closed TaCQ + contemporaneous SG server plan."""

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


def _control_registration(model: str, seed: int) -> str:
    return (
        "experiments/revision_full/outputs/tacq/controls/"
        f"{model}__sg_contemporary__c{seed}.json"
    )


def shadow_commands(gpu: int = 0) -> list[str]:
    del gpu
    return [
        "echo 'Protocol v2 online-stop Shadow was rejected; do not rerun or tune it.' >&2",
        "exit 2",
    ]


def tacq_commands(gpu: int = 0) -> list[str]:
    run = "python experiments/revision_full/run.py"
    tacq = "python experiments/revision_full/tacq.py"
    result = [
        "python -m unittest discover -s experiments/revision_full -p 'test_*.py'",
        "python experiments/revision_full/server_preflight.py --expected-gpus 2 --concurrent-models 1",
    ]
    # A failed Shadow run may have reconstructed these c41 artifacts after the
    # core cleanup receipts were written.  Restore lifecycle consistency while
    # preserving every Shadow row and receipt as historical evidence.
    for model in TACQ_MODELS:
        result.extend(
            [
                f"{run} cleanup-state --model {model} --calib-seed 41 --variant gptq_w4",
                f"{run} cleanup-state --model {model} --calib-seed 41 --variant sg_mmp",
                f"{run} cleanup-bank --model {model} --calib-seed 41",
            ]
        )
    result.extend(
        [
            "python experiments/revision_full/readiness.py --stage core",
            f"{tacq} freeze --source-commit {OFFICIAL_SOURCE_COMMIT}",
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
            control = _control_registration(model, seed)
            # A registered seed is an atomic completed unit.  A missing record
            # means that every command below resumes its validated artifacts.
            # Both arms are generated from the same seed-specific bank before
            # either reconstructible state is cleaned.
            result.append(
                f"if [[ ! -f {registration} || ! -f {control} ]]; then "
                f"CUDA_VISIBLE_DEVICES={gpu} {run} build-bank --model {model} --calib-seed {seed} --require-output; "
                f"{run} materialize --model {model} --calib-seed {seed} --variant sg_mmp --require-output; "
                f"if [[ ! -f {control} ]]; then "
                f"CUDA_VISIBLE_DEVICES={gpu} {tacq} evaluate-control --model {model} --calib-seed {seed}; "
                "fi; "
                f"if [[ ! -f {registration} ]]; then "
                f"{tacq} build --model {model} --calib-seed {seed}; "
                f"CUDA_VISIBLE_DEVICES={gpu} {tacq} smoke --model {model} --calib-seed {seed}; "
                f"CUDA_VISIBLE_DEVICES={gpu} {tacq} evaluate --model {model} --calib-seed {seed}; "
                "fi; "
                "fi"
            )
            result.extend(
                (
                    f"{tacq} cleanup --model {model} --calib-seed {seed}",
                    f"{run} cleanup-state --model {model} --calib-seed {seed} --variant sg_mmp",
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
        return tacq_commands(gpu)
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

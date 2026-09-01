"""Print the serial server command plan without launching experiments."""

import sys

sys.path.insert(0, ".")

from experiments.revision_full.protocol import (
    CALIB_SEEDS,
    MODEL_SPECS,
    RANDOM_ALLOCATIONS,
    RANDOM_CALIB_SEED,
    SCREEN_SEEDS,
)


CONTROL_VARIANTS = (
    "qkv_only",
    "o_only",
    "ffn_only",
    "qkv_priority_matched",
    "o_priority_matched",
    "ffn_priority_matched",
    "hessian_diag_matched",
)


def state_cycle(
    model: str,
    seed: int,
    variant: str,
    *,
    panels: bool = False,
    causal: bool = False,
):
    if variant in {"gptq_w5", "gptq_w6"}:
        yield (
            "python experiments/revision_full/run.py quantize-uniform "
            f"--model {model} --calib-seed {seed} --bits {variant[-1]}"
        )
    else:
        yield (
            "python experiments/revision_full/run.py materialize "
            f"--model {model} --calib-seed {seed} --variant {variant}"
        )
    yield (
        "python experiments/revision_full/run.py evaluate-full "
        f"--model {model} --variant {variant} --calib-seed {seed}"
    )
    if panels:
        yield (
            "python experiments/revision_full/run.py evaluate-broad "
            f"--model {model} --variant {variant} --calib-seed {seed}"
        )
        yield (
            "python experiments/revision_full/run.py evaluate-extra "
            f"--model {model} --variant {variant} --calib-seed {seed}"
        )
        yield (
            "python experiments/revision_full/format_control.py "
            f"--model {model} --variant {variant} --calib-seed {seed}"
        )
    if causal:
        yield (
            "python experiments/revision_full/causal_patch.py run "
            f"--model {model} --calib-seed {seed}"
        )
    yield (
        "python experiments/revision_full/run.py cleanup-state "
        f"--model {model} --calib-seed {seed} --variant {variant}"
    )


def commands():
    yield 'python -m unittest discover -s experiments/revision_full -p "test*.py" -v'
    yield "python experiments/revision_full/run.py prepare"
    yield "python experiments/revision_full/format_control.py --prepare-only"
    yield "python experiments/revision_full/readiness.py --stage preflight"

    for model in MODEL_SPECS:
        for split_id in range(len(SCREEN_SEEDS)):
            yield (
                "python experiments/revision_full/run.py screen "
                f"--model {model} --split-id {split_id}"
            )
        yield f"python experiments/revision_full/run.py select --model {model}"

        yield (
            "python experiments/revision_full/run.py evaluate-full "
            f"--model {model} --variant fp16"
        )
        for command in [
            "evaluate-broad",
            "evaluate-extra",
        ]:
            yield (
                f"python experiments/revision_full/run.py {command} "
                f"--model {model} --variant fp16"
            )
        yield (
            "python experiments/revision_full/format_control.py "
            f"--model {model} --variant fp16"
        )

        for seed in [value for value in CALIB_SEEDS if value != RANDOM_CALIB_SEED]:
            yield (
                "python experiments/revision_full/run.py build-bank "
                f"--model {model} --calib-seed {seed}"
            )
            for variant in ["gptq_w4", "gptq_w5", "gptq_w6", "sg_mmp"]:
                yield from state_cycle(model, seed, variant)
            yield (
                "python experiments/revision_full/run.py cleanup-bank "
                f"--model {model} --calib-seed {seed}"
            )

        yield (
            "python experiments/revision_full/run.py build-bank "
            f"--model {model} --calib-seed {RANDOM_CALIB_SEED}"
        )
        for variant in ["gptq_w5", "gptq_w6", *CONTROL_VARIANTS]:
            yield from state_cycle(model, RANDOM_CALIB_SEED, variant)

        if MODEL_SPECS[model]["role"] == "primary":
            for allocation_id in range(RANDOM_ALLOCATIONS):
                for prefix in ["random", "random_modules"]:
                    variant = f"{prefix}_{allocation_id}"
                    yield (
                        "python experiments/revision_full/run.py evaluate-allocation "
                        f"--model {model} --variant {variant} --calib-seed {RANDOM_CALIB_SEED}"
                    )

        yield from state_cycle(
            model,
            RANDOM_CALIB_SEED,
            "gptq_w4",
            panels=True,
            causal=model == "qwen05",
        )
        yield from state_cycle(
            model, RANDOM_CALIB_SEED, "sg_mmp", panels=True
        )
        yield (
            "python experiments/revision_full/run.py cleanup-bank "
            f"--model {model} --calib-seed {RANDOM_CALIB_SEED}"
        )

        if MODEL_SPECS[model]["role"] == "primary":
            yield (
                "python experiments/revision_full/error_analysis.py prepare "
                f"--model {model} --calib-seed {RANDOM_CALIB_SEED} --sample-size 200"
            )

    yield "python experiments/revision_full/analyze.py"
    yield "python experiments/revision_full/readiness.py --stage core"


if __name__ == "__main__":
    print("#!/usr/bin/env bash")
    print("set -euo pipefail")
    for command in commands():
        print(command)

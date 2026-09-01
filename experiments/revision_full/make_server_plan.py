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
        for seed in CALIB_SEEDS:
            yield (
                "python experiments/revision_full/run.py build-bank "
                f"--model {model} --calib-seed {seed}"
            )
            for variant in ["gptq_w4", "sg_mmp"]:
                yield (
                    "python experiments/revision_full/run.py materialize "
                    f"--model {model} --calib-seed {seed} --variant {variant}"
                )
            for bits in [5, 6]:
                yield (
                    "python experiments/revision_full/run.py quantize-uniform "
                    f"--model {model} --calib-seed {seed} --bits {bits}"
                )
            for variant in ["gptq_w4", "gptq_w5", "gptq_w6", "sg_mmp"]:
                yield (
                    "python experiments/revision_full/run.py evaluate-full "
                    f"--model {model} --variant {variant} --calib-seed {seed}"
                )

        for variant in [
            "qkv_only",
            "o_only",
            "ffn_only",
            "qkv_priority_matched",
            "o_priority_matched",
            "ffn_priority_matched",
            "hessian_diag_matched",
        ]:
            yield (
                "python experiments/revision_full/run.py materialize "
                f"--model {model} --calib-seed {RANDOM_CALIB_SEED} --variant {variant}"
            )
            yield (
                "python experiments/revision_full/run.py evaluate-full "
                f"--model {model} --variant {variant} --calib-seed {RANDOM_CALIB_SEED}"
            )

        if MODEL_SPECS[model]["role"] == "primary":
            for allocation_id in range(RANDOM_ALLOCATIONS):
                for prefix in ["random", "random_modules"]:
                    variant = f"{prefix}_{allocation_id}"
                    yield (
                        "python experiments/revision_full/run.py evaluate-allocation "
                        f"--model {model} --variant {variant} --calib-seed {RANDOM_CALIB_SEED}"
                    )

        for variant, seed_arg in [
            ("fp16", ""),
            ("gptq_w4", f" --calib-seed {RANDOM_CALIB_SEED}"),
            ("sg_mmp", f" --calib-seed {RANDOM_CALIB_SEED}"),
        ]:
            yield (
                "python experiments/revision_full/run.py evaluate-broad "
                f"--model {model} --variant {variant}{seed_arg}"
            )
            yield (
                "python experiments/revision_full/run.py evaluate-extra "
                f"--model {model} --variant {variant}{seed_arg}"
            )
            yield (
                "python experiments/revision_full/format_control.py "
                f"--model {model} --variant {variant}{seed_arg}"
            )

        if MODEL_SPECS[model]["role"] == "primary":
            yield (
                "python experiments/revision_full/error_analysis.py prepare "
                f"--model {model} --calib-seed {RANDOM_CALIB_SEED} --sample-size 200"
            )

        if model == "qwen05":
            yield (
                "python experiments/revision_full/causal_patch.py run "
                f"--model {model} --calib-seed {RANDOM_CALIB_SEED}"
            )

    yield "python experiments/revision_full/analyze.py"
    yield "python experiments/revision_full/readiness.py --stage core"


if __name__ == "__main__":
    for command in commands():
        print(command)

"""Print the commands belonging to one model for one-GPU execution."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from experiments.revision_full.make_server_plan import commands
from experiments.revision_full.protocol import MODEL_SPECS


SETUP_COMMANDS = 4


def shard_commands(models: list[str], include_setup: bool = False) -> list[str]:
    if not models or any(model not in MODEL_SPECS for model in models):
        raise ValueError("A shard needs one or more known models")
    if len(models) != len(set(models)):
        raise ValueError("Shard models must be unique")
    all_commands = list(commands())
    selected = list(all_commands[:SETUP_COMMANDS]) if include_setup else []
    for model in models:
        marker = f"--model {model}"
        selected.extend(command for command in all_commands if marker in command)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=MODEL_SPECS)
    group.add_argument("--models", nargs="+", choices=MODEL_SPECS)
    parser.add_argument("--include-setup", action="store_true")
    args = parser.parse_args()
    print("#!/usr/bin/env bash")
    print("set -euo pipefail")
    selected = args.models or [args.model]
    try:
        selected_commands = shard_commands(selected, args.include_setup)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    for command in selected_commands:
        print(command)


if __name__ == "__main__":
    main()

"""Print the commands belonging to one model for one-GPU execution."""

from __future__ import annotations

import argparse
import sys

sys.path.insert(0, ".")

from experiments.revision_full.make_server_plan import commands
from experiments.revision_full.protocol import MODEL_SPECS


SETUP_COMMANDS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_SPECS, required=True)
    parser.add_argument("--include-setup", action="store_true")
    args = parser.parse_args()
    all_commands = list(commands())
    if args.include_setup:
        for command in all_commands[:SETUP_COMMANDS]:
            print(command)
    marker = f"--model {args.model}"
    for command in all_commands:
        if marker in command:
            print(command)


if __name__ == "__main__":
    main()

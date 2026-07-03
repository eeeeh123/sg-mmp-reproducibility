# Reproduction environment

The released analyses and figure generation were checked with the following
software stack on Windows:

| Component | Version |
|---|---:|
| Python | 3.13.12 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| Transformers | 5.8.0 |
| Datasets | 4.8.5 |
| LM Evaluation Harness | 0.4.11 |

`requirements.txt` records the top-level package pins. The quantization and
generation runs require a CUDA-capable GPU with enough memory for the selected
FP16 checkpoint plus temporary quantization state. Wall-clock time and memory
use are environment-dependent and are not reproducibility targets.

## Data and checkpoint setup

1. Create a virtual environment and install `requirements.txt`.
2. Download the three primary checkpoints with
   `python scripts/reproduce_core.py download-primary`.
3. Cache public WikiText-2 and GSM8K with
   `python scripts/reproduce_core.py prepare-data`.

The original run did not preserve upstream Hugging Face commit hashes or a
dataset fingerprint. This is a provenance limitation, documented in
`configs/reproduction_manifest.json`; do not describe a new rerun as
byte-identical to the original unless those revisions are recorded.

## Windows note

Some historical large non-compact `.pt` state files caused native Windows /
PyTorch access violations. The released evaluator refuses those legacy state
files and prefers compact states where they are available. Keep the default
separate-process quantization and evaluation stages in `reproduce_core.py`.

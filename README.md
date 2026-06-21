# SG-MMP Reproducibility Package

This repository contains the code, fixed evaluation indices, and derived
results supporting *Reasoning Fragility in Quantized Small Language Models:
Diagnosis and Sensitivity-Guided Mixed-Precision Repair*.

## What is included

- Implementations of GPTQ-style quantization, the in-house AWQ baseline, and
  sensitivity-guided module-level mixed precision (SG-MMP).
- Broad benchmark scripts and the direct paired GSM8K-500 evaluator.
- The fixed GSM8K-500 index set, redacted per-example correctness outcomes,
  paired statistics, and figure-generation code.
- A figure-generation script that reads only the released derived results.

Model weights, quantized state files, benchmark prompts and solutions, and
generated reasoning traces are intentionally not redistributed. They are either
large, governed by upstream licenses, or not needed to inspect the reported
statistics.

## Main direct GSM8K-500 results

| Model | GPTQ-W4 | SG-MMP | Difference | Paired bootstrap 95% CI |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B | 16.80 | 26.80 | +10.00 | [+6.20, +14.00] |
| Qwen2.5-1.5B | 46.00 | 56.20 | +10.20 | [+6.00, +14.40] |
| SmolLM2-1.7B | 18.80 | 25.80 | +7.00 | [+3.20, +10.80] |
| Gemma-2-2B-it | 47.20 | 50.40 | +3.20 | [-0.40, +6.80] |

The Gemma result is a boundary check whose confidence interval crosses zero;
it is not presented as confirmatory evidence for SG-MMP.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the original checkpoints under their upstream licenses. The model
identifiers are recorded in `docs/model_provenance.md`. Gemma access requires
accepting its upstream terms. The public evaluator expects local models under
`models/` and quantized states under `results/`; these states are regenerated,
not distributed.

```powershell
# Broad benchmark pipeline
python scripts/01_download_models.py
python experiments/exp01_baseline/run.py

# Recompute paired statistics from the released redacted outcomes
python scripts/analyze_released_gsm8k500.py

# Regenerate analysis figures from released results
python scripts/generate_figures.py
```

For exact protocols, data boundaries, expected outputs, and model identity,
read [docs/reproducibility.md](docs/reproducibility.md) and
[docs/model_provenance.md](docs/model_provenance.md).

## Artifact boundaries

`data/processed/gsm8k500/per_example_correctness.csv` contains only model key,
method, test-document identifier, normalized prediction, and correctness. It
does not include GSM8K prompts, reference answers, or model generations. To
rerun evaluation from scratch, obtain GSM8K from its original distribution.


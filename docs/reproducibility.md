# Reproducibility protocol

## Scope

The paper reports two distinct evaluation settings:

1. **Broad benchmark setting**: ARC-Challenge, HellaSwag, MMLU, and a fixed
   GSM8K-300 subset, evaluated through the local LM Evaluation Harness stack.
2. **Direct paired setting**: a fixed GSM8K-500 test subset using a common
   five-shot prompt, greedy generation, exact-match numeric extraction, exact
   McNemar tests, and 10,000 paired-bootstrap resamples.

These settings must not be conflated. The direct GSM8K-500 comparison is the
primary confirmatory SG-MMP repair result.

## Deterministic inputs

- GSM8K-500 seed: `20260615`.
- Selection: shuffle `range(1319)` with the seed, retain the first 500, then
  sort indices into dataset order.
- Calibration: 128 WikiText-2 samples, sequence length 2048 unless an
  experiment explicitly says otherwise.
- GPTQ and weight-only methods: 4-bit weights, group size 128.
- Direct generation: greedy decoding, five in-context examples, maximum 256
  generated tokens.

## Reproduce derived analyses without checkpoints

The released redacted data is sufficient to audit paired outcomes and the
reported direct scores. The statistical summaries are in:

```text
data/processed/source_artifacts/experiments/fix_gsm8k_500/results_direct/
data/processed/gsm8k500/
```

## Recompute model outputs

1. Install `requirements.txt` in a CUDA-enabled Python environment matching
   the recorded package versions where possible.
2. Download upstream checkpoints under their original licenses.
3. Run the required sensitivity screen, GPTQ quantization, and SG-MMP
   quantization scripts for a model family.
4. Run `experiments/fix_gsm8k_500/direct_eval.py` on identical local model and
   state paths, then run its `analyze` command.
5. Rebuild analysis figures with `python scripts/generate_figures.py`.

Quantized `.pt` states are omitted because they are multi-gigabyte,
implementation-specific intermediate artifacts. Their absence does not alter
the fixed inputs or release of all derived outcomes.

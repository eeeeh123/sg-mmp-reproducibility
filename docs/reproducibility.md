# Reproducibility protocol

## Evaluation settings

The paper uses two distinct settings that must not be conflated:

1. **Broad benchmark setting:** ARC-Challenge, HellaSwag, MMLU, and a fixed
   GSM8K-300 subset through the local LM Evaluation Harness stack.
2. **Direct paired setting:** a fixed GSM8K-500 test subset with a common
   five-shot prompt, greedy generation, numeric exact match, exact McNemar
   tests, and 10,000 paired-bootstrap resamples.

The direct setting is reproducible through
`experiments/fix_gsm8k_500/direct_eval.py`. It loads public GSM8K online by
default; pass `--offline` only when reproducing from an existing Arrow cache.

## Fixed protocol

- GSM8K-500 seed: `20260615`.
- Selection: shuffle `range(1319)` with the seed, keep the first 500, then
  sort indices into test-set order.
- Calibration: 128 WikiText-2 training samples, seed 42, sequence length 2048.
- GPTQ: 4-bit weights and group size 128.
- Direct generation: five in-context examples, greedy decoding, maximum 256
  generated tokens.

`configs/reproduction_manifest.json` is the machine-readable source for this
protocol and for the published sensitive-layer sets.

## Public verification path

The release contains enough redacted data to audit the reported direct paired
statistics without model checkpoints:

```powershell
python scripts/reproduce_core.py verify-public
python scripts/reproduce_core.py figures
```

The first command validates `SHA256SUMS` and recomputes
`data/processed/gsm8k500/recomputed_paired_stats.json` from
`per_example_correctness.csv`. The second command creates ignored local figure
files from released summaries only.

## End-to-end rerun path

```powershell
python scripts/reproduce_core.py download-primary
python scripts/reproduce_core.py prepare-data
python scripts/reproduce_core.py quantize
python scripts/reproduce_core.py evaluate
python scripts/reproduce_core.py analyze
```

The wrapper is intentionally explicit about stages. It does not publish or
copy intermediate `.pt` states, raw GSM8K records, or raw generations. The
private evaluator writes raw logs under ignored `samples/` directories; use
`scripts/export_public_gsm8k_results.py` if a future release needs redacted
per-example outcomes.

## Selection/evaluation separation

The original Qwen2.5-0.5B single-layer sensitivity screen used the first 300
GSM8K test documents. Consequently, 119 documents in the fixed direct-500
subset overlap that historical screen. This release includes a non-overlap
robustness analysis on the remaining 381 documents in:

```text
data/processed/source_artifacts/experiments/fix_gsm8k_500/
results_direct/selection_eval_split_gsm8k500.json
```

For Qwen2.5-0.5B, the non-overlap slice reports GPTQ-W4 16.01, SG-MMP 28.08,
and a +12.07 point paired difference (95% bootstrap CI [7.61, 16.54]; exact
McNemar p = 3.32e-7). This is a robustness check, not a substitute for a
fresh validation-based layer-selection study.

## Provenance limitations

The original runs did not preserve Hugging Face checkpoint commit hashes or a
dataset fingerprint. The release records the canonical identifiers and exact
input-selection algorithm, but it does not claim that a new upstream download
will be byte-identical. Record those revisions before any future rerun.

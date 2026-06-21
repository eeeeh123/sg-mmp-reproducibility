# Processed data

This directory contains derived results needed to inspect the figures and
reported statistics. It does not redistribute raw benchmark records, reference
answers, calibration text, model checkpoints, or generated reasoning traces.

- `gsm8k500/gsm8k_500_indices.json` fixes the 500 test-document identifiers
  selected with seed `20260615`.
- `gsm8k500/per_example_correctness.csv` contains only derived predictions and
  correctness flags. It has 18 complete model/method groups of 500 rows.
- `gsm8k500/per_example_correctness_manifest.json` documents the fields and
  explicitly records raw fields excluded during export.
- `source_artifacts/` holds compact summary tables and diagnostic outputs used
  in the paper.

The exporter at `scripts/export_public_gsm8k_results.py` is the exact
redaction procedure used to create the per-example CSV from private evaluator
logs.

# SG-MMP Reproducibility Package

Version `1.2.0` of the code and derived results supporting *Reasoning
Fragility in Quantized Small Language Models: Diagnosis and
Sensitivity-Guided Mixed-Precision Repair*.

## What this release can reproduce

Two reproduction paths are deliberately separated:

1. **Public-artifact verification, no GPU or checkpoints required.** Recompute
   the paired GSM8K-500 statistics from redacted per-example outcomes and
   regenerate all manuscript figures backed by the released numerical data.
2. **End-to-end model rerun.** Download the three primary checkpoints, cache
   public datasets, regenerate GPTQ and SG-MMP states, and run direct
   GSM8K-500 evaluation.

The package does not redistribute model weights, quantized states, GSM8K
prompts or answers, or generated reasoning traces. See
`docs/reproducibility.md` for the protocol and `docs/environment.md` for the
tested software stack.

## Main direct GSM8K-500 results

| Model | GPTQ-W4 | SG-MMP | Difference | Paired bootstrap 95% CI |
|---|---:|---:|---:|---|
| Qwen2.5-0.5B | 16.80 | 26.80 | +10.00 | [+6.20, +14.00] |
| Qwen2.5-1.5B | 46.00 | 56.20 | +10.20 | [+6.00, +14.40] |
| SmolLM2-1.7B | 18.80 | 25.80 | +7.00 | [+3.20, +10.80] |
| Gemma-2-2B-it | 47.20 | 50.40 | +3.20 | [-0.40, +6.80] |

Gemma-2-2B-it is a boundary-family check: its confidence interval crosses
zero and is not confirmatory evidence for SG-MMP.

## Quick start: verify the public release

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/reproduce_core.py verify-public
python scripts/reproduce_core.py figures
```

`verify-public` validates every source-file checksum and confirms that the
released redacted outcomes reproduce the paired-statistics JSON byte-for-byte.
Generated figures are written to the ignored local `figures/` directory. The
quantitative result plots are produced by `scripts/generate_figures.py`; the
study-overview, precision-policy, and error-propagation figures are produced by
`scripts/generate_concept_figures.py` from the same released JSON/CSV sources.

## End-to-end rerun

```powershell
python scripts/reproduce_core.py download-primary
python scripts/reproduce_core.py prepare-data
python scripts/reproduce_core.py quantize
python scripts/reproduce_core.py evaluate
python scripts/reproduce_core.py analyze
```

The main quantization steps are GPU-intensive. The wrapper runs each model
family in separate Python processes and writes intermediate states under the
ignored local `results/` directory. Use `--dry-run` with any command to inspect
the exact commands before running them.

## Important provenance note

The original local downloads did not preserve Hugging Face checkpoint commit
hashes or the original dataset fingerprint. Canonical model identifiers,
protocol, fixed test indices, and this limitation are recorded in
`configs/reproduction_manifest.json`. A future rerun should record its own
checkpoint revisions before claiming byte-identical reproduction.

For model identity, artifact-to-claim mapping, and archive boundaries, see
`docs/model_provenance.md`, `docs/artifact_manifest.md`, and
`docs/zenodo_release.md`.


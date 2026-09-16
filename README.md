# SG-MMP Reproducibility Package

Version `2.0.0` of the code supporting *Reasoning Fragility under Low-Bit
Quantization: Full-Test, Seed-Aware Evidence for Sensitivity-Guided Mixed
Precision*.

The immutable source release is tagged
[`v2.0.0`](https://github.com/eeeeh123/sg-mmp-reproducibility/tree/v2.0.0).
Versioned result archives are deposited under the stable Zenodo concept DOI
[`10.5281/zenodo.21096006`](https://doi.org/10.5281/zenodo.21096006). See
[`docs/zenodo_release.md`](docs/zenodo_release.md) for the exact v2.0.0 file
inventory and integrity procedure.

## What v2.0.0 contains

The release separates source code from large result evidence:

1. **Source archive.** This repository, frozen protocol definitions, tests,
   environment pins, and analysis code.
2. **`revision-full-v4` evidence.** Full 1,319-item GSM8K results, three
   calibration seeds, matched-budget controls, cross-task summaries, TaCQ
   registrations and contemporaneous SG-MMP controls, frozen manifests, state
   metadata, and analysis outputs.
3. **Packed deployment evidence.** Four-model GGUF/CUDA quality and performance
   records for FP16, Q4, Q5, and the frozen SG allocation, including ten
   process blocks per model and method, gate history, artifact manifests, and
   final analyses.

Pretrained weights, reconstructible PyTorch quantized states, GGUF weight
files, and dataset caches are not redistributed. Their immutable identities,
hashes where available, and reconstruction commands are preserved in the
released manifests.

## Main `revision-full-v4` result

The primary endpoint is direct five-shot greedy generation on all 1,319 GSM8K
test items. Calibration seeds 41, 97, and 193 are repeated calibration
realizations, not independent scientific claims. Model-level intervals use the
pre-specified two-stage seed/item bootstrap.

| Model | FP16 | W4 mean | SG-MMP mean | SG-MMP minus W4 | 95% CI | SG average bits |
|---|---:|---:|---:|---:|---|---:|
| Qwen2.5-0.5B | 35.56 | 12.36 | 22.04 | +9.68 | [8.24, 11.09] | 4.897 |
| Qwen2.5-1.5B | 60.65 | 47.97 | 54.26 | +6.29 | [4.27, 8.29] | 4.935 |
| SmolLM2-1.7B | 28.89 | 14.18 | 25.12 | +10.94 | [8.57, 13.34] | 4.885 |
| Gemma-2-2B-it | 52.77 | 45.69 | 49.07 | +3.39 | [2.00, 4.83] | 4.890 |

SG-MMP is a consistent partial recovery from W4 in this setting. It does not
outperform uniform W5/W6. The TaCQ shared-backend comparison is model dependent:
both model-level seed/item intervals include zero. These boundaries are part of
the result, not omitted failure cases.

## Packed deployment extension

The deployment extension asks whether the already frozen allocation transfers
to a useful quality-memory-speed trade-off on one pinned `llama.cpp` CUDA
backend. It uses newly quantized GGUF artifacts and is reported separately from
the PyTorch GPTQ experiment. Across the tested workloads, SG is slower than Q4,
usually faster than Q5 at concurrency four, and does not have a universal
quality advantage over Q5. See
[`experiments/deployment_gguf/README.md`](experiments/deployment_gguf/README.md)
for the frozen design and limitations.

## Verify the source release

Create an environment from `requirements.txt`, then run:

```powershell
python scripts/reproduce_core.py verify-public
python -m unittest `
  experiments.revision_full.test_protocol `
  experiments.revision_full.test_resume `
  experiments.revision_full.test_lifecycle `
  experiments.revision_full.test_diagnostics `
  experiments.revision_full.test_quantization_pipeline `
  experiments.revision_full.test_tacq_protocol `
  experiments.deployment_gguf.test_deployment_gguf -v
```

`verify-public` validates `SHA256SUMS` and the legacy v1.2 redacted
GSM8K-500 statistics retained for provenance. The v2.0.0 Zenodo delivery
manifest separately validates the large revision and deployment archives.

## Reproduce the current experiments

- Full revision protocol: [`experiments/revision_full/README.md`](experiments/revision_full/README.md)
- TaCQ extension: [`experiments/revision_full/TACQ_INTEGRATION.md`](experiments/revision_full/TACQ_INTEGRATION.md)
- GGUF deployment extension: [`experiments/deployment_gguf/README.md`](experiments/deployment_gguf/README.md)
- Artifact-to-claim map: [`docs/artifact_manifest.md`](docs/artifact_manifest.md)
- Environment and provenance: [`docs/environment.md`](docs/environment.md) and
  [`docs/model_provenance.md`](docs/model_provenance.md)

The experiment scripts write all large local outputs beneath ignored
`outputs/`, `results/`, `samples/`, and `logs/` paths. Do not add weights or
server caches to Git.

## Legacy v1.2 material

The historical v1.2.0 GSM8K-500 analysis remains in `data/processed/` so that
earlier claims and figures stay auditable. It is exploratory provenance only
and must not be pooled with or substituted for the complete v2.0.0
`revision-full-v4` evidence.

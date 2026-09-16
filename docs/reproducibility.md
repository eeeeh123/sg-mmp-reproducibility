# Reproducibility protocol

## Current confirmatory protocol (`revision-full-v4`)

The current paper uses all 1,319 official GSM8K test items for four models:
Qwen2.5-0.5B, Qwen2.5-1.5B, SmolLM2-1.7B, and Gemma-2-2B-it. Precision
selection uses training data only. Seeds 41, 97, and 193 represent calibration
variability and are aggregated at the model level with a two-stage seed/item
bootstrap; they are not treated as independent scientific claims.

The fixed generation protocol is direct five-shot greedy decoding with
`max_new_tokens=256`. The flexible numeric extractor is primary. Strict
`#### number` extraction and delimiter coverage are sensitivity analyses. A
candidate online `Question:` stopping rule failed its pre-specified Shadow gate
and was rejected before the TaCQ extension; all reported new baselines retain
the original generation semantics.

The complete execution order, lifecycle rules, and readiness gates are in:

- `experiments/revision_full/EXPERIMENT_PLAN.md`
- `experiments/revision_full/README.md`
- `experiments/revision_full/SERVER_MIGRATION.md`
- `experiments/revision_full/TACQ_INTEGRATION.md`

The Zenodo v2.0.0 revision archive supplies the generated `protocol_lock.json`,
resolved model and dataset manifests, selections, screens, per-run/sample
records, state metadata, external-baseline registrations, and
`analysis_full.json`. Reconstructible `.pt` states are intentionally excluded.

## TaCQ extension

TaCQ is a disclosed shared-backend adaptation, not an unmodified upstream
reproduction. Its source commit, train-only importance settings, rounding rule,
logical-bit ledger, and mask hashes are frozen before test evaluation. Each of
the six TaCQ cells is paired with a contemporaneously regenerated SG-MMP
control. Primary inference is performed for two model-level effects; seed-level
McNemar and bootstrap results are diagnostics.

## Packed GGUF/CUDA extension

The deployment study starts from the frozen SG allocation but requantizes the
original high-precision checkpoints into backend-specific GGUF artifacts. It
therefore evaluates allocation transfer, not the runtime of the Python
`GPTQLinear` implementation. FP16, Q4, Q5, and SG artifacts use one pinned
`llama.cpp` commit and a shared train-only importance matrix within each model.

Quality uses the full 1,319-item protocol. Performance uses fixed 128-token
generation and ten process blocks per model and method. `llama-bench` compute
measurements are kept separate from streaming `llama-server` TTFT, latency,
memory, and throughput. The Zenodo deployment archive preserves every retained
block, quality row, artifact/gate manifest, historical failure record, and the
final four-model analysis. It does not contain GGUF weights.

## Public verification

The source snapshot can be verified without model weights:

```powershell
python scripts/reproduce_core.py verify-public
```

This checks `SHA256SUMS` and recomputes the historical redacted v1.2 paired
statistics byte-for-byte. The large v2.0.0 evidence is validated independently
with `MANIFEST_v2.0.0.json` and `SHA256SUMS_v2.0.0.txt` from the Zenodo upload.
End-to-end reruns require downloading the model checkpoints and public datasets
identified in the archived manifests.

## Historical v1.2 protocol

The repository retains the fixed GSM8K-500 analysis and broad benchmark
artifacts released in v1.2.0. That material is exploratory provenance only. It
must not be pooled with, averaged into, or substituted for missing
`revision-full-v4` evidence. Its original selection seed, redacted outcomes,
and recomputed paired statistics remain under `data/processed/`.

## Availability boundary

Authored code is released under MIT. Public benchmark records and model outputs
inside the evidence archives are included for result auditing and remain
subject to applicable upstream terms. Pretrained weights, dataset caches,
reconstructible quantized states, and GGUF weight files are not redistributed;
the archive records resolved identities, hashes, commands, and environment
details needed to reconstruct them.

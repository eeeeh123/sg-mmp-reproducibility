# Artifact-to-claim manifest

Paths beginning with `revision/` or `deployment/` refer to the correspondingly
named Zenodo v2.0.0 result archive. Repository paths are relative to the source
root.

| Reported claim or analysis | Released evidence |
|---|---|
| Full-test FP16/W4/W5/W6/SG-MMP accuracies and model-level seed/item inference | `revision/experiments/revision_full/outputs/analysis_full.json`, `analysis_full.md`, and `results/samples/` |
| Same-item generation-versus-MCQ interaction | `revision/.../results/format_control/`, canonical generation sample records, and `experiments/revision_full/analyze.py` |
| Train-only layer selection and matched logical budgets | `revision/.../screens/`, `selections/`, `state_metadata/`, and `protocol_lock.json` |
| Random, structured, and module-placement controls | Canonical records in `revision/.../results/samples/` plus `analysis_full.json` |
| TaCQ shared-backend adaptation | `revision/.../tacq/frozen_manifest.json`, `external_baselines/`, contemporaneous SG/TaCQ sample records, the separate run-era TaCQ archive, and `experiments/revision_full/TACQ_INTEGRATION.md` |
| Generation integrity and rejected online-stop candidate | `sg-mmp-integrity-audits-v2.0.0.zip`, TaCQ readiness logs, and `experiments/revision_full/shadow_gate.py` |
| Cross-task descriptive transfer | `revision/.../results/broad/`, `results/extra/`, and `analysis_full.json` |
| Packed GGUF quality results | `deployment/quality/samples/`, `quality/summaries/`, and the phase analysis JSON files |
| Packed memory, latency, TTFT, and throughput | `deployment/benchmarks/raw/`, `benchmarks/blocks/`, artifact manifests, and `unified_10_block_review.json` |
| Deployment caveats and gate history | `deployment/status/`, `provenance/`, `README_FINAL.md`, and `experiments/deployment_gguf/methodology_review.md` |
| Historical v1.2 GSM8K-500 results | `data/processed/`; provenance only, not v2.0.0 revision evidence |

The Zenodo delivery-level `MANIFEST_v2.0.0.json` and
`SHA256SUMS_v2.0.0.txt` bind each uploaded archive to its exact bytes. Internal
model, dataset, state, artifact, and block hashes provide the next level of
provenance without redistributing large reconstructible weights.

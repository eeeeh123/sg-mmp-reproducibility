# SG-MMP Reproducibility Package

Version `1.2.0` of the code and derived results supporting *Reasoning
Fragility in Quantized Small Language Models: Diagnosis and
Sensitivity-Guided Mixed-Precision Repair*.

The current `main` branch additionally includes the fail-closed
`revision-full-v4` rejection-revision pipeline. Its new full-test GPU results
are intentionally marked pending; the published v1.2 GSM8K-500 results below
remain exploratory provenance and are forbidden as v4 evidence.

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

## Historical v1.2 direct GSM8K-500 results (not revision evidence)

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

## Rejection-revision full experiment

The resource-aware v4 protocol uses all 1,319 official GSM8K test items,
native train-only sensitivity screens for every model, three calibration
seeds, W4/W5/W6 and matched-placement controls, two 30-allocation null
families for every primary model, selection/bootstrap uncertainty, explicit
format and task controls, and fail-closed external-baseline gates.

Model weights are not stored in GitHub or Git LFS. On the laboratory server,
download and pin them directly from Hugging Face:

```bash
python experiments/revision_full/download_models.py --models qwen05 qwen15 smollm
hf auth login  # required after accepting the Gemma license
python experiments/revision_full/download_models.py --models gemma2
python experiments/revision_full/download_core_datasets.py
export REVISION_FULL_STATE_DIR=/scratch/$USER/sg-mmp-revision-states  # optional
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export REVISION_FULL_EVAL_BATCH_SIZE=4 REVISION_FULL_FORMAT_BATCH_SIZE=2
# Smoke all four architectures on GSM8K train before freezing batch settings.
CUDA_VISIBLE_DEVICES=0 python experiments/revision_full/run.py smoke-eval --model gemma2 --batch-size 4 --format-batch-size 2
CUDA_VISIBLE_DEVICES=0 python experiments/revision_full/run.py smoke-eval --model qwen05 --batch-size 4 --format-batch-size 2
CUDA_VISIBLE_DEVICES=1 python experiments/revision_full/run.py smoke-eval --model smollm --batch-size 4 --format-batch-size 2
CUDA_VISIBLE_DEVICES=1 python experiments/revision_full/run.py smoke-eval --model qwen15 --batch-size 4 --format-batch-size 2
python experiments/revision_full/run.py prepare --force
python experiments/revision_full/format_control.py --prepare-only --force
python experiments/revision_full/server_preflight.py --expected-gpus 2 --concurrent-models 2
mkdir -p server_plans logs
python experiments/revision_full/make_server_shard.py --models gemma2 qwen05 > server_plans/gpu0.sh
python experiments/revision_full/make_server_shard.py --models smollm qwen15 > server_plans/gpu1.sh
# Launch one shard per GPU only when preflight reports ready=true.
```

The downloader records immutable upstream commit SHAs and resumes interrupted
files. The generated server plan keeps only one materialized quantized state at
a time, validates and hashes persistent evidence before cleanup, and safely
skips already-complete work even after reconstructible `.pt` files are removed.
The largest single-process transient state peak is about 9.9 GiB; two
concurrent model processes are estimated at 17.78 GiB before safety and
persistent-result reserves. See
`experiments/revision_full/SERVER_MIGRATION.md` for the complete
two-RTX-3090 workflow and `experiments/revision_full/EXPERIMENT_PLAN.md`
for the reviewer-facing evidence gates.

## Important provenance note

The original local downloads did not preserve Hugging Face checkpoint commit
hashes or the original dataset fingerprint. Canonical model identifiers,
protocol, fixed test indices, and this limitation are recorded in
`configs/reproduction_manifest.json`. A future rerun should record its own
checkpoint revisions before claiming byte-identical reproduction.

For model identity, artifact-to-claim mapping, and archive boundaries, see
`docs/model_provenance.md`, `docs/artifact_manifest.md`, and
`docs/zenodo_release.md`.


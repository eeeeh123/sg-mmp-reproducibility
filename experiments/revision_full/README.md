# Rejection-revision resource-aware full experiment pipeline (v4)

This pipeline is the only source for revised headline results. Historical GSM8K-300/500 scripts and outputs are preserved for provenance, but `claim_policy.json` forbids using them as revised-paper evidence.

## What is fixed before GPU execution

- Canonical GSM8K evaluation is the direct 5-shot greedy evaluator on all 1,319 official test examples. The lm-evaluation-harness broad table intentionally excludes GSM8K, so the revision cannot silently produce two headline GSM8K numbers.
- Sensitivity selection uses three disjoint 256-item train-only screens (768 unique development items in total); every model gets a native selection.
- Calibration robustness uses three fixed seeds: `41, 97, 193`. This is the preregistered lower bound for “several” independent calibration runs; run-level and example-level uncertainty remain mandatory.
- Core baselines include GPTQ-W4, uniform GPTQ-W5, uniform GPTQ-W6, SG-MMP, up to 30 matched random-layer allocations, and 30 matched random-module allocations for every primary model. If fewer than 30 distinct whole-layer placements are mathematically feasible, the exhaustive feasible set is locked and reported without duplicates.
- Module controls include pure q/k/v, o, and FFN curves; budget-matched role-priority controls; and a calibration-weighted diagonal-Hessian reconstruction control.
- Statistical analysis includes paired example inference, two-stage seed/example bootstrap intervals, Holm correction, random-allocation percentiles, and a paired format-by-method interaction.
- Selection stability includes a fixed-seed 2,000-replicate bootstrap over the three disjoint train-screen units, with per-layer inclusion rates and set-level Jaccard; its three-unit limitation must be reported.
- The format control uses the same 1,319 questions in deterministic multiple-choice form.
- The error-analysis tool records every automatic transition and prepares a fixed-seed, blinded 200-case annotation sheet.
- A fixed 200-example teacher-forced activation-patching diagnostic is implemented for Qwen2.5-0.5B.
- Deployment speed/memory claims are disabled until a real packed-kernel backend is measured.

## Server preflight

After downloading and fingerprinting every model and dataset on the server, run:

```bash
python -m unittest discover -s experiments/revision_full -p "test*.py" -v
python experiments/revision_full/run.py prepare --force
python experiments/revision_full/format_control.py --prepare-only --force
python experiments/revision_full/server_preflight.py --expected-gpus 2 --concurrent-models 2
```

Preflight also verifies immutable checkpoint/cache hashes, both GPUs, offline mode, RAM, disk, and the batch/token lock. A 32-GiB two-GPU host must use `REVISION_FULL_MAX_CONCURRENT_RAM_BUILDERS=1`: high-host-RAM screen/precision builders are serialized by a cross-process lock, while the two GPU workers may still evaluate concurrently. Run the four train-only smoke tests in `SERVER_MIGRATION.md` before creating the protocol lock.

## Server run order

For first-time upload, environment creation, multi-GPU sharding, monitoring, and result download, follow `SERVER_MIGRATION.md`.

Print the complete serial command matrix:

```bash
python experiments/revision_full/make_server_plan.py
```

The generated plan contains tests, protocol preparation, preflight, all model runs, analysis, and the final core-results gate. Do not mix commands from `fix_gsm8k_300` or `fix_gsm8k_500` into this run.

The core loop is lifecycle-aware. It materializes, consumes, verifies, and
cleans one state before creating the next:

```bash
python experiments/revision_full/run.py build-screen-bank --model qwen05 --split-id 0 --calib-seed 41
python experiments/revision_full/run.py screen --model qwen05 --split-id 0
python experiments/revision_full/run.py cleanup-screen-bank --model qwen05 --split-id 0 --calib-seed 41
# Repeat build/screen/cleanup for split 1/seed 97 and split 2/seed 193.
python experiments/revision_full/run.py select --model qwen05
python experiments/revision_full/run.py build-bank --model qwen05 --calib-seed 97
python experiments/revision_full/run.py quantize-uniform --model qwen05 --calib-seed 97 --bits 5
python experiments/revision_full/run.py evaluate-full --model qwen05 --variant gptq_w5 --calib-seed 97
python experiments/revision_full/run.py cleanup-state --model qwen05 --calib-seed 97 --variant gptq_w5
# Repeat the materialize/evaluate/cleanup cycle for W4, W6, and SG-MMP.
python experiments/revision_full/run.py cleanup-bank --model qwen05 --calib-seed 97
```

The full plan also runs FP16, all controls, ARC-Challenge/HellaSwag/MMLU, an explicitly reported MMLU high-school-mathematics multiple-choice score, and the full SVAMP/ASDiv/MATH-500/TruthfulQA-generation panel. Per-example logs are mandatory where available.

Set `REVISION_FULL_STATE_DIR` to node-local scratch when available. Quantized
`.pt` files live there; persistent metadata and SHA256 cleanup receipts remain
under `outputs/state_metadata/` and `outputs/lifecycle_receipts/`. Deletion is
fail-closed: incomplete GSM8K IDs, missing panels, or a missing causal diagnostic
keeps the required state. A normal rerun checks complete evidence before looking
for a state, so already-cleaned work is not recomputed.

### Resuming an existing run

Keep `outputs/` intact, including selections, screens, state metadata, cleanup
receipts, and samples. Repeating `select` validates and reuses the saved selection
byte-for-byte; it does not redraw or renumber random allocations. A changed
screen, provenance, or invalid manifest stops the run for inspection instead of
overwriting the locked experiment. Missing or invalid random manifests also
block precision-bank cleanup.

After updating code, regenerate the GPU shard scripts before restarting a stopped
worker. The allocation-id lookup and every random evaluation must succeed; an
empty or failed lookup now stops the shard instead of silently skipping a family.
Do not edit scripts or update code under active workers, and do not launch a
second worker for the same model. Check the current run's log filename, timestamp,
and Python processes first: an exception in an older log does not identify a new
failure. For example, the September 3 repaired run used `gpu0_bf674b8.log` and
`gpu1_bf674b8.log`, not the earlier `gpu0.log` and `gpu1.log`.

Without `--force`, compatible complete results are skipped; canonical GSM8K
generation resumes missing document IDs after validating existing rows. This is
artifact/sample-level resumption, not a universal in-memory checkpoint: an
interrupted builder or an evaluation panel without sample-level resume may need
to rerun that stage. Do not use `prepare --force`, delete selections/results, or
relax bit-budget gates to resume an existing run.

## Error analysis and causal diagnostic

After W4 and SG-MMP seed-41 generations exist for a primary model:

```bash
python experiments/revision_full/error_analysis.py prepare --model qwen05 --calib-seed 41 --sample-size 200
```

Give output A and output B separate labels. Complete both consensus output fields for all 200 rows; for the 40 rows marked `double_code_required=1`, both raters must label both outputs. Then run:

```bash
python experiments/revision_full/error_analysis.py summarize --annotations <annotation.csv>
```

The server plan also runs:

```bash
python experiments/revision_full/causal_patch.py run --model qwen05 --calib-seed 41
```

This diagnostic may support a mechanistic statement only if its corrected inferential results support it; otherwise remove the causal wording.

## External baselines and final gates

TaCQ and HAWQ-V2 must run from their official implementations in isolated environments and be imported through the canonical result contract described in `TACQ_INTEGRATION.md`. Registration rejects incomplete samples, missing provenance, and bit budgets differing from SG-MMP by more than 0.05 bit. The resubmission gate requires both methods on Qwen2.5-0.5B and Qwen2.5-1.5B; they are reviewer-mandated and are not removed by the resource-aware internal-matrix reduction.

```bash
python experiments/revision_full/external_baselines.py validate
python experiments/revision_full/analyze.py
python experiments/revision_full/readiness.py --stage core
python experiments/revision_full/readiness.py --stage resubmission
```

- `core` requires all preregistered internal numerical experiments.
- `resubmission` additionally requires registered official TaCQ and HAWQ-V2 results and completed blinded error annotation.

## Result replacement policy

Do not overwrite old 300/500 files. New results live only under `experiments/revision_full/outputs/` and replace old numerical claims in the revised manuscript. Old results may remain as exploratory provenance, but cannot be pooled with, averaged into, or used to repair missing v4 runs.

## Resource changes that do not relax the reviewer-facing evidence

Version 4 changes execution cost, provenance, and failure detection without relaxing the estimand or final-test evidence. It uses calibration-repeated GPTQ-W4 screens, preregisters unique random sets, stages and hashes every dataset/model file, locks batch and decoding settings into resumable rows, isolates dual-GPU writes, and analyzes relative error increase and normalized recovery. The complete 1,319-item test, native screens, all primary models, both random-allocation families, uniform controls, task/format/error controls, and block/attention/MLP causal diagnostic remain required.

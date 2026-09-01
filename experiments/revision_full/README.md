# Rejection-revision resource-aware full experiment pipeline (v3)

This pipeline is the only source for revised headline results. Historical GSM8K-300/500 scripts and outputs are preserved for provenance, but `claim_policy.json` forbids using them as revised-paper evidence.

## What is fixed before GPU execution

- Canonical GSM8K evaluation is the direct 5-shot greedy evaluator on all 1,319 official test examples. The lm-evaluation-harness broad table intentionally excludes GSM8K, so the revision cannot silently produce two headline GSM8K numbers.
- Sensitivity selection uses three disjoint 256-item train-only screens (768 unique development items in total); every model gets a native selection.
- Calibration robustness uses three fixed seeds: `41, 97, 193`. This is the preregistered lower bound for “several” independent calibration runs; run-level and example-level uncertainty remain mandatory.
- Core baselines include GPTQ-W4, uniform GPTQ-W5, uniform GPTQ-W6, SG-MMP, 30 matched random-layer allocations, and 30 matched random-module allocations for every primary model.
- Module controls include pure q/k/v, o, and FFN curves; budget-matched role-priority controls; and a calibration-weighted diagonal-Hessian reconstruction control.
- Statistical analysis includes paired example inference, two-stage seed/example bootstrap intervals, Holm correction, random-allocation percentiles, and a paired format-by-method interaction.
- Selection stability includes a fixed-seed 2,000-replicate bootstrap over the three disjoint train-screen units, with per-layer inclusion rates and set-level Jaccard; its three-unit limitation must be reported.
- The format control uses the same 1,319 questions in deterministic multiple-choice form.
- The error-analysis tool records every automatic transition and prepares a fixed-seed, blinded 200-case annotation sheet.
- A fixed 200-example teacher-forced activation-patching diagnostic is implemented for Qwen2.5-0.5B.
- Deployment speed/memory claims are disabled until a real packed-kernel backend is measured.

## Local preflight

Run from the repository root before copying the project to the server:

```bash
python -m unittest experiments.revision_full.test_protocol
python experiments/revision_full/run.py prepare --force
python experiments/revision_full/format_control.py --prepare-only --force
python experiments/revision_full/readiness.py --stage preflight
```

`preflight` fails closed if the protocol version, three seeds, full test set, fixed format manifest, or existing state metadata disagree.

## Server run order

For first-time upload, environment creation, multi-GPU sharding, monitoring, and result download, follow `SERVER_MIGRATION.md`.

Print the complete serial command matrix:

```bash
python experiments/revision_full/make_server_plan.py
```

The generated plan contains tests, protocol preparation, preflight, all model runs, analysis, and the final core-results gate. Do not mix commands from `fix_gsm8k_300` or `fix_gsm8k_500` into this run.

The core loop for one model and seed is:

```bash
python experiments/revision_full/run.py screen --model qwen05 --split-id 0
python experiments/revision_full/run.py screen --model qwen05 --split-id 1
python experiments/revision_full/run.py screen --model qwen05 --split-id 2
python experiments/revision_full/run.py select --model qwen05
python experiments/revision_full/run.py build-bank --model qwen05 --calib-seed 41
python experiments/revision_full/run.py materialize --model qwen05 --calib-seed 41 --variant gptq_w4
python experiments/revision_full/run.py materialize --model qwen05 --calib-seed 41 --variant sg_mmp
python experiments/revision_full/run.py quantize-uniform --model qwen05 --calib-seed 41 --bits 5
python experiments/revision_full/run.py quantize-uniform --model qwen05 --calib-seed 41 --bits 6
python experiments/revision_full/run.py evaluate-full --model qwen05 --variant gptq_w4 --calib-seed 41
python experiments/revision_full/run.py evaluate-full --model qwen05 --variant gptq_w5 --calib-seed 41
python experiments/revision_full/run.py evaluate-full --model qwen05 --variant gptq_w6 --calib-seed 41
python experiments/revision_full/run.py evaluate-full --model qwen05 --variant sg_mmp --calib-seed 41
```

The full plan also runs FP16, all controls, ARC-Challenge/HellaSwag/MMLU, an explicitly reported MMLU high-school-mathematics multiple-choice score, and the full SVAMP/ASDiv/MATH-500/TruthfulQA-generation panel. Per-example logs are mandatory where available.

## Error analysis and causal diagnostic

After W4 and SG-MMP seed-41 generations exist for a primary model:

```bash
python experiments/revision_full/error_analysis.py prepare --model qwen05 --calib-seed 41 --sample-size 200
```

Complete `rater1_label`, `rater2_label` for at least 40 rows, and `consensus_label` for all 200 rows. Then run:

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

Do not overwrite old 300/500 files. New results live only under `experiments/revision_full/outputs/` and replace old numerical claims in the revised manuscript. Old results may remain as exploratory provenance, but cannot be pooled with, averaged into, or used to repair missing v3 runs.

## Resource changes that do not relax the reviewer-facing evidence

Version 3 changes execution cost, not the estimand or the final test evidence. It packs real WikiText tokens without synthetic zero padding; balances a 4,096-token Hessian reservoir across all 128 calibration sequences; captures module activations once per model/seed; derives W4/W5/W6 from the same Hessian; evaluates generation at a conservative RTX-3090 default batch size of four; and avoids rereading the growing JSONL file after every batch. The complete 1,319-item test, native screen for every family, all primary models, both random-allocation families, uniform controls, task/format/error controls, and causal diagnostic remain required.

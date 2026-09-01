# Code Experiment Plan

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: implementation and validation
- Origin Date: 2026-08-31
- Verification Status: CODE-VERIFIED; GPU RESULTS PENDING
- Version Label: revision_full_v3_resource_aware

## Objective and fixed hypotheses

The primary question is whether train-selected SG-MMP improves GPTQ-W4 on the complete GSM8K test set under a matched parameter-weighted bit budget. Secondary questions test whether the gain survives calibration variability, exceeds uniform-bit and allocation controls, generalizes across models/tasks, changes with answer format, and has a measurable layer-level causal diagnostic.

## Hard protocol gates

1. Screening uses three disjoint 256-example subsets from GSM8K train (768 unique development items); the fixed demonstrations and all test examples are excluded.
2. Every headline GSM8K result uses all 1,319 official test examples through one direct 5-shot greedy evaluator.
3. The broad lm-evaluation-harness table contains ARC-Challenge, HellaSwag, MMLU, and the explicit MMLU high-school-mathematics subset only; it is forbidden from generating a second GSM8K headline value.
4. Every model receives a native screen. Transferring Qwen-selected layers to another architecture is forbidden.
5. The allocation target is a parameter-weighted average precision of at most 5.0 bits. Every matched control must be within 0.01 bit internally; external baselines must be within 0.05 bit.
6. SG-MMP and internal mixed allocations use the same calibration-specific W4/W8 precision bank.
7. Calibration robustness uses three fixed WikiText seeds: 41, 97, and 193. All 128 packed sequences contribute to a fixed per-module activation reservoir.
8. Required uniform baselines are GPTQ-W4, W5, and W6 for every model and seed.
9. Required allocation controls are 30 matched random-layer and 30 matched random-module allocations for every primary model at seed 41. Thirty is fixed in advance and supports the reviewer-requested “dozens” and empirical-percentile analysis.
10. Required placement controls are pure q/k/v, o, and FFN curves plus budget-matched q/k/v-priority, o-priority, FFN-priority, and diagonal-Hessian-reconstruction allocations.
11. All inferential families use multiplicity correction. Calibration uncertainty uses a two-stage seed/example bootstrap, not only a pooled per-example interval.
12. Historical 300/500 results are provenance only and cannot enter revised headline estimates.
13. Layer-selection stability is recomputed in 2,000 fixed-seed bootstrap resamples of the three disjoint train-screen units; report inclusion probabilities, exact-set rate, Jaccard interval, and the three-unit limitation.

## Models and tasks

Primary models are Qwen2.5-0.5B, Qwen2.5-1.5B, and SmolLM2-1.7B. Gemma-2-2B-it is a predeclared family/boundary check.

- Canonical task: full GSM8K test, direct free generation.
- Same-item format control: deterministic four-choice GSM8K for the same 1,319 item IDs.
- Broad discriminative panel: ARC-Challenge, HellaSwag, MMLU, plus an explicitly reported MMLU high-school-mathematics multiple-choice score.
- Generative transfer panel: SVAMP, ASDiv, MATH-500, TruthfulQA-generation.

## Decision rules

| Claim | Required evidence |
|---|---|
| SG-MMP improves W4 | Positive full-test delta, paired-bootstrap CI excluding zero, and Holm-adjusted exact McNemar p below 0.05 |
| Gain is stable across calibration | Three run deltas plus two-stage seed/example 95% CI; report every run, mean, SD, minimum, and maximum; do not claim broad seed invariance from only three runs |
| Gain is not merely extra precision | Accuracy-versus-bit comparison against uniform W5 and W6; wording must follow the observed uncertainty |
| Placement is informative | Percentile against both 30-allocation null families and comparison with budget-matched role/Hessian controls |
| Format matters | Paired difference-in-differences interaction over identical item IDs, with corrected inference |
| Mechanistic propagation | Fixed activation-patching result with corrected inference; otherwise delete causal/mechanistic wording |
| External competitiveness | Official TaCQ plus HAWQ-V2 runs on Qwen 0.5B/1.5B, canonical evaluator, complete per-item generations, pinned commits/configs, matched bit budget |
| Deployment efficiency | Disabled unless packed serialized size, peak memory, latency, and throughput are measured with real packed kernels |

## Error analysis

Automatic error analysis covers all 1,319 seed-41 W4/SG-MMP item transitions, parse failures, EOS status, generated-token count, and truncation flags. For each primary model, a fixed-seed sample of 200 non-both-correct items is blinded and annotated using arithmetic, reasoning setup, state tracking, extraction/format, truncation, hallucination, or other. At least 40 cases are double coded; all 200 require consensus labels. Cohen's kappa and category counts are reported.

## Expected artifacts and gates

| Artifact | Success condition |
|---|---|
| `outputs/protocol_lock.json` | `revision-full-v3`, full 1,319 test, three seeds, packed calibration, single evaluator |
| `outputs/screens/` and `outputs/selections/` | Native train-only selection for every model |
| `outputs/results/samples/` | Every required method contains exactly IDs 0-1318 |
| `outputs/results/format_control/` | Same complete item IDs for FP16/W4/SG-MMP |
| `outputs/results/broad/` | Exactly ARC-Challenge, HellaSwag, MMLU, and MMLU high-school mathematics; no GSM8K |
| `outputs/results/extra/` | Four complete task records with logged samples |
| `outputs/results/causal_patch/` | 200 preregistered items and every decoder layer |
| `outputs/external_baselines/` | Hash-verified official baseline provenance and canonical samples |
| `outputs/state_metadata/` | Persistent protocol, budget, seed, and source-bank metadata for reconstructible states |
| `outputs/lifecycle_receipts/` | SHA256-identified evidence that passed before each transient `.pt` deletion |
| `outputs/analysis_full.md` | Paired, hierarchical, random-control, format-interaction, and external comparisons |
| `readiness.py --stage core` | No missing internal run |
| `readiness.py --stage resubmission` | TaCQ, HAWQ-V2, and human annotation also complete |

## What must wait for the server

The protocol, code paths, output schemas, analysis, and fail-closed gates are fixed now. Model screening, quantization, full inference, causal patching, and third-party TaCQ/HAWQ-V2 execution require server compute. Human annotation must wait until the new generations exist. These pending computations must not be replaced with historical subset results.

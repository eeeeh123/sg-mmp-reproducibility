# Rejection-revision experiment plan (revision-full-v4)

Status: core complete; shadow-gated TaCQ add-on code-verified locally; server execution pending.

## Confirmatory question and separation rules

The primary question is whether train-selected SG-MMP improves GPTQ-W4 on the complete GSM8K test set under the same parameter-weighted precision budget. GSM8K train is used for demonstrations and layer-selection development only. The official 1,319-example test split is untouched until final evaluation and is never used to select layers, hyperparameters, random allocations, or calibration settings.

Historical GSM8K-300/500 runs are exploratory provenance. They must not be pooled with, substituted for, or used to fill missing v4 results. Old files remain intact; revised manuscript tables and conclusions must be regenerated only from `experiments/revision_full/outputs/`.

## Frozen internal matrix

1. Models: Qwen2.5-0.5B, Qwen2.5-1.5B, and SmolLM2-1.7B are primary; Gemma-2-2B-it is a predeclared family/boundary check. Every model gets a native selection.
2. Selection: three disjoint 256-example GSM8K-train screens, excluding the five demonstrations. Each split is paired with a different WikiText calibration seed (`41`, `97`, `193`) and a real GPTQ-W4 state. Layer ranking is aggregated across splits.
3. Selection stability: 2,000 fixed-seed bootstrap resamples of the three split-level units. Report inclusion probabilities, exact-set rate, Jaccard interval, and the limitation of having only three split units.
4. Canonical accuracy: direct five-shot greedy generation on all 1,319 official GSM8K test examples, with `max_new_tokens=256` and one locked per-GPU batch size.
5. Calibration robustness: every model runs GPTQ-W4, uniform GPTQ-W5, uniform GPTQ-W6, and SG-MMP for all three calibration seeds.
6. Same-budget controls: each primary model requests 30 unique preregistered random-layer and 30 unique preregistered random-module allocations at seed 41. If a model admits fewer than 30 distinct whole-layer placements at the locked budget, all feasible placements are exhaustively enumerated and the model-specific count and feasibility certificate are stored. Allocation lists and hashes are stored before evaluation; allocations are never duplicated to reach 30.
7. Placement controls: every model runs pure q/k/v, output-projection, and FFN precision curves plus budget-matched q/k/v-priority, output-priority, FFN-priority, and diagonal-Hessian reconstruction allocations.
8. Format/task controls at seed 41: FP16, GPTQ-W4, and SG-MMP run the identical 1,319 GSM8K items as both free generation and deterministic four-choice scoring. They also run ARC-Challenge, HellaSwag, MMLU, MMLU high-school mathematics, generative SVAMP, generative ASDiv, MATH-500, and TruthfulQA generation.
9. Causal diagnostic: Qwen2.5-0.5B uses a fixed, model-output-independent 200-item test subset. Aligned FP16 outputs replace GPTQ-W4 block, self-attention, or MLP outputs at every layer. The primary outcome is gold final-answer-token NLL; full reasoning-trace NLL/logit similarity/KL are secondary. Holm correction is applied separately to the primary and secondary NLL families. This is diagnostic, not generated-answer accuracy.
10. Error audit: all 1,319 W4/SG transitions, parse failures, EOS/truncation status, and generated lengths are automatic. For every primary model, 200 fixed-seed non-both-correct cases are blinded. Each output receives its own label; all cases require two consensus output labels and a fixed preregistered set of 40 cases is double-coded.

## Statistical reporting

- Core SG-W4 paired comparison follows the locked v4 analysis. For the TaCQ add-on, per-seed paired bootstrap/McNemar results are diagnostics only. The primary TaCQ inference aggregates all three seeds within each Qwen model using a two-stage seed/example bootstrap and a paired-item cluster sign-flip test; Holm adjustment covers exactly the two model-level hypotheses.
- Calibration uncertainty: all three run deltas plus two-stage bootstrap over calibration seeds and paired examples; report mean, SD, minimum, maximum, and interval. Three seeds do not justify a universal invariance claim.
- Quantization severity: absolute accuracy degradation, relative error increase, and normalized recovery against FP16. Cross-task values are descriptive unless item-level pairing exists.
- Allocation evidence: SG percentile and empirical one-sided p-value against both 30-member null families. Missing or duplicate allocations disable the claim.
- Format confounding: paired difference-in-differences on the same GSM8K IDs with multiplicity correction.
- Negative or inconsistent results are retained. No post-result model, seed, task, or layer-set deletion is allowed.

## Reviewer-facing decision gates

| Claim | Required evidence |
|---|---|
| SG-MMP improves W4 | Positive full-test paired delta, CI excluding zero, and adjusted McNemar p below 0.05 |
| Robust to calibration | Three complete seed runs and the hierarchical interval; wording follows observed heterogeneity |
| Not merely extra precision | Direct W5/W6 accuracy-versus-bit comparisons and matched internal placement controls |
| Learned placement matters | Complete the locked random controls (up to 30 exhaustive layer placements plus 30 module placements) and role/Hessian matched controls |
| Format contributes | Corrected same-item generation-versus-MCQ interaction |
| Transfers beyond one benchmark | Full locked task panels with FP16/W4/SG and relative-error reporting |
| Mechanistic propagation | Complete block/attention/MLP patching with corrected final-answer outcome; otherwise remove causal language |
| Competitive with automated mixed precision | Shadow-validated TaCQ shared-backend adaptation for Qwen 0.5B/1.5B, three calibration seeds, complete canonical samples, pinned code/config, and non-exceeding budget within 0.01 bit. HAWQ-V2 is not claimed. |
| Deployment efficiency | Disabled until real packed-kernel size, peak memory, latency, and throughput exist |

## Required gates and artifacts

| Gate/artifact | Exact success condition |
|---|---|
| `protocol_lock.json` | `revision-full-v4`, frozen dataset hash, full test, three calibration-repeated GPTQ screens, execution batch/token lock |
| dataset/model manifests | Immutable model revisions plus weight hashes; every core/panel dataset cache file fingerprinted and available offline |
| screens/selections | Exactly one baseline and every native layer per split; requested/actual allocation counts, uniqueness, hashes, and any exhaustive layer-feasibility certificate |
| canonical/format samples | Exactly IDs 0-1318 once, valid correctness fields, v4 provenance, locked batch settings |
| panel records | Exact task set, no second GSM8K evaluator, full logged samples for generative tasks |
| causal record | Exactly 200 fixed IDs and every block/attention/MLP-by-layer pair |
| lifecycle receipts | Evidence validated and hashed before each reconstructible state deletion |
| `readiness.py --stage core` | Every preregistered internal computation complete |
| `readiness.py --stage shadow` | 200/200 historical-prefix, prediction, and correctness equivalence |
| `readiness.py --stage tacq` | Six seed registrations, two model-level primary effects, and two-hypothesis Holm family |
| `readiness.py --stage resubmission` | TaCQ gate plus explicit HAWQ-V2 and human-error-taxonomy claim waivers |

The pipeline is fail closed: partial files, duplicate IDs, stale protocol/data/model provenance, changed batch settings, missing task samples, or an incomplete downstream consumer stop execution and preserve the needed state.

## What is not solved before server execution

GPU memory behavior for the TaCQ adaptation must still pass save/reload and 32-generation train-only smoke gates on the actual RTX 3090 server. TaCQ importance checkpoints are large and are deleted only after all registered evidence is validated. HAWQ-V2 and human error-taxonomy claims are explicitly omitted. These add-on results do not rerun, pool with, or rewrite valid v4 core outputs.

# Shadow-gated TaCQ shared-backend adaptation

This add-on does not modify or rerun the 52 immutable core sample files. It
addresses two separate questions in a fail-closed order:

1. Does generated-only early stopping preserve the historical evaluator's
   answer prefix, prediction, and correctness exactly?
2. Under the same eligible projection scope and a non-exceeding ±0.01-bit
   logical budget, does a TaCQ allocation outperform or underperform SG-MMP?

The comparison must be called **TaCQ shared-backend adaptation**. The TaCQ
importance formula is retained, while the GPTQ-W4 group-128 backend and direct
GSM8K evaluator are shared with SG-MMP for controlled comparison. The actual
serialized state includes redundant W4 values beneath the FP16 exception mask;
therefore the experiment supports allocation-quality claims, not deployment
speed or physical-memory claims.

## Frozen source and adaptation

- Official source: `https://github.com/The-Inscrutable-X/TACQ`
- Pinned commit: `cfc4cccfb6b7d6f7d184c9fbc8f9373c3e74569a`
- Models: Qwen2.5-0.5B and Qwen2.5-1.5B
- Calibration seeds: 41, 97, 193
- Importance: 128 deterministically selected GSM8K-train examples (the official
  default scale), excluding the five fixed demonstration rows, batch 1, full
  causal loss over the five-shot prompt plus worked answer, float16 gradient
  computation under the locked loader, per-example absolute gradients, float32
  sum, no normalization, and a 2,048-token limit
- The clean gradient accumulator is computed once per model. Each seed uses its
  own locked GPTQ-W4 perturbation and therefore its own score/mask.
- Allocation: global element-level W4/FP16 mask. The FP16 count is the largest
  integer count that does not exceed the model's frozen SG-MMP logical budget.
  Equal scores are resolved by module name and row-major index.
- No importance count, loss, normalization, mask rule, bit rounding, or other
  TaCQ setting may change after the manifest is written or after inspecting a
  TaCQ test output.

## Gates

`shadow_gate.py prepare` freezes 50 archived IDs per model. W4 and SG each
replay those IDs, for 200 formal shadow generations. `verify` writes `PASS.json`
only when all 200 canonical answer prefixes, extracted predictions, and
correctness labels match the immutable old outputs exactly. The prompt is never
scanned by the stop processor. A failed gate forbids TaCQ test evaluation.
It also ends this preregistered add-on: do not tune the stop rule using failed
Shadow rows and retry it under the same protocol identity.

`tacq.py freeze` then records every adaptation degree, input identity, model and
dataset provenance, Shadow receipt, budget rule, and statistical plan. Gradient
capture is checkpointed every 32 train examples. Each seed must subsequently
pass formula/module/finite-score/mask/budget checks and a state save-reload plus
32-generation train-only smoke test before the 1,319-item test command is
available. The manifest hashes every implementation file that can affect the
Shadow comparison or TaCQ evaluation. Each final sample row is bound to the
exact state, deterministic mask, frozen manifest, and train-only smoke receipt;
readiness revalidates those hashes before accepting a registration or analysis.
Once any TaCQ test output exists, missing importance chunks, states, masks, or
smoke receipts cannot be reconstructed in place.

Generate two conservative one-GPU plans. Run the TaCQ plan only after the
Shadow plan exits successfully and `readiness --stage shadow` reports
`"ready": true`:

```bash
python experiments/revision_full/make_tacq_plan.py --phase shadow > server_plans/shadow_gate.sh
python experiments/revision_full/make_tacq_plan.py --phase tacq > server_plans/tacq_serial.sh
bash -n server_plans/shadow_gate.sh
bash -n server_plans/tacq_serial.sh
```

Each phase begins with the two-GPU/server-RAM preflight. The TaCQ plan rechecks
the Shadow receipt at its boundary. It skips a seed only when its validated
registration exists; otherwise it resumes the seed from its validated artifacts.
Do not use `--force` after any downstream test output exists.

## Statistical contract

Each seed retains paired bootstrap and exact McNemar results as diagnostics.
They are not six independent scientific hypotheses. The primary inference is
one SG-minus-TaCQ effect per model using the same two-stage calibration-seed /
paired-example bootstrap as the core SG-minus-W4 analysis. A paired-item
cluster sign-flip test supplies the model-level p-value, and Holm correction is
applied to exactly the two model-level hypotheses.

`readiness.py --stage tacq` checks all six registrations, both model-level
effects, the three diagnostic seeds per model, the two-test Holm family, the
manifest, the Shadow receipt, and both bit ledgers. HAWQ-V2 and human error
taxonomy are explicitly not claimed; no internal surrogate is relabelled as an
external method.

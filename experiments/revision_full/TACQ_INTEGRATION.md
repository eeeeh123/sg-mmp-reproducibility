# TaCQ external-baseline extension with contemporaneous SG controls

This extension does not modify, replace, or pool the 52 immutable core sample
files. It asks one additional question: under the same eligible projection
scope, seed-specific precision bank, original evaluator, and non-exceeding
matched logical budget, how does SG-MMP compare with a disclosed TaCQ
shared-backend adaptation?

## Protocol history

- **v1:** all completed core experiments used direct five-shot greedy generation
  with `max_new_tokens=256` and no online stopping rule.
- **v2 candidate:** a generated-only `Question:` stopping rule was tested on 200
  formal Shadow generations. It failed the pre-specified exact-equivalence gate
  (186/200 canonical prefixes, 192/200 extracted predictions, and 198/200
  correctness labels matched) and was rejected before any TaCQ test evaluation.
  The failed receipt and rows remain evidence; they must not be tuned against or
  rerun under the same protocol identity.
- **v3 external-baseline extension:** returns to v1 generation semantics. It adds
  six TaCQ cells and six contemporaneously regenerated SG-MMP controls for two
  Qwen models and three calibration seeds.

The archived-output audit may be reported only as evidence that offline
truncation at a subsequent `Question:` marker left archived flexible predictions
unchanged. It is not evidence that online stopping and regeneration are exactly
equivalent.

## Frozen source and adaptation

- Official source: `https://github.com/The-Inscrutable-X/TACQ`
- Pinned commit: `cfc4cccfb6b7d6f7d184c9fbc8f9373c3e74569a`
- Models: Qwen2.5-0.5B and Qwen2.5-1.5B
- Calibration seeds: 41, 97, 193
- Importance: 128 deterministically selected GSM8K-train examples, excluding
  the five fixed demonstrations; batch 1; full causal loss over the five-shot
  prompt plus worked answer; float16 gradient computation; per-example absolute
  gradients accumulated in float32; no normalization; 2,048-token limit.
- The clean gradient accumulator is computed once per model. Each calibration
  seed uses its own locked GPTQ-W4 perturbation and therefore its own score/mask.
- Allocation: global element-level W4/FP16 mask. The FP16 count is the largest
  integer count that does not exceed the frozen SG-MMP logical budget. Equal
  scores are resolved by module name and row-major index.
- No importance count, loss, normalization, mask rule, bit rounding, or other
  TaCQ setting may change after the manifest is written or after inspecting a
  TaCQ or contemporaneous-SG test output.

## Paired control contract

Every `(model, calibration_seed)` cell uses one reconstructed precision bank to
materialize both the frozen SG-MMP state and the TaCQ state. Both arms use:

- the same repository commit and server environment;
- the same complete 1,319-item GSM8K test set and direct five-shot prompt;
- greedy generation with `max_new_tokens=256` and no online stop;
- the same tokenizer, generation implementation, batch size, and flexible
  extraction;
- distinct sample files bound to their exact state hashes and the shared
  precision-bank hash.

TaCQ test evaluation is blocked until the corresponding contemporaneous SG
control has completed and validated. Cleanup is blocked until both registrations
exist. The old core SG files remain the source for the core paper tables; only
the new control files enter the SG-versus-TaCQ extension table.

## Gates and execution

`tacq.py freeze` records every adaptation degree, input identity, model and
dataset provenance, protocol history, paired-control matrix, generation
semantics, and statistical plan before new test access. Gradient capture is
checkpointed every 32 train examples. Each seed must pass formula, module,
finite-score, deterministic-mask, logical-budget, save/reload, and 32-generation
train-only smoke checks.

The smoke check and both formal arms use the original generation path; none uses
the rejected online stopping processor. Each sample row is bound to the frozen
manifest and state evidence. Readiness revalidates hashes and confirms that SG
and TaCQ used the same seed-specific precision bank.

Generate the conservative one-GPU extension plan:

```bash
python experiments/revision_full/make_tacq_plan.py --phase tacq > server_plans/tacq_serial.sh
bash -n server_plans/tacq_serial.sh
```

`--phase all` is an alias for this v3 extension. `--phase shadow` deliberately
produces a failing script so the rejected v2 candidate cannot be accidentally
rerun. The plan first runs local tests, server preflight, and core readiness. It
then freezes the manifest, creates paired cells, analyzes them, and checks TaCQ
and resubmission readiness. It never uses `--force`.

## Statistical contract

Each seed retains paired bootstrap and exact McNemar results as diagnostics;
the six seed runs are not six independent scientific hypotheses. The primary
inference is one contemporaneous-SG-minus-TaCQ effect per model, using the same
two-stage calibration-seed / paired-example bootstrap as the core analysis. A
paired-item cluster sign-flip test supplies each model-level p-value, and Holm
correction is applied to exactly the two model-level hypotheses.

The method must be called **TaCQ shared-backend adaptation**. The state contains
redundant W4 values beneath the FP16 exception mask, so the extension supports
allocation-quality claims, not deployment speed or physical-memory claims.
HAWQ-V2 and a human error taxonomy are not claimed.

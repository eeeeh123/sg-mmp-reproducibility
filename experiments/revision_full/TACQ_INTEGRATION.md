# TaCQ and established automated mixed-precision baseline integration

TaCQ is deliberately isolated from the main environment because its official repository specifies Python 3.12.2/CUDA and a separate dependency stack. The revision pipeline does not reimplement TaCQ or relabel the internal diagonal-Hessian heuristic as TaCQ/HAWQ-V2.

## Reproducibility lock

- Official repository: `https://github.com/The-Inscrutable-X/TACQ`
- Commit pinned on 2026-09-01: `cfc4cccfb6b7d6f7d184c9fbc8f9373c3e74569a`
- Run the official code in a separate environment on the laboratory server.
- Save the exact command, configuration, environment/package lock, model revision, calibration source/seed, selected precision allocation, and parameter-weighted average bit width.

The official scripts currently demonstrate larger Llama/Qwen models. Adapting a primary model is allowed only by changing model-specific loading/configuration while preserving the official TaCQ importance and allocation procedure. Record every adaptation in the imported config; do not describe an approximate local surrogate as TaCQ.

Use only GSM8K train and/or the locked WikiText calibration data for importance and allocation. Test questions, test labels, and v4 test generations are forbidden inputs. Tune the official preservation ratio without looking at test accuracy until the measured parameter-weighted budget is within 0.05 bit of that model's frozen SG-MMP budget. Record every tried ratio so budget matching cannot become hidden test-set tuning.

TaCQ importance arrays can be checkpoint-sized. Run one model at a time on scratch. After canonical samples, the full config/environment lock, measured bit accounting, and hashes are registered, the importance arrays and temporary checkpoints are reconstructible and may be removed.

## Canonical evaluation contract

After TaCQ produces a quantized model/state, evaluate it with the same direct 5-shot greedy GSM8K evaluator used by `run.py evaluate-full`. Export a JSONL file containing exactly one row for each `doc_id` 0-1318. Required fields match the auditable direct evaluator:

```json
{"doc_id": 0, "question": "...", "gold": "...", "prediction": null, "correct": 0, "generation": "...", "generated_token_count": 256, "truncated": true}
```

Use the same prompt construction, answer extractor, maximum generation length, official GSM8K test ordering, and model chat/raw prompt style as the internal methods. Do not import an accuracy number without per-example generations. Copy `external_baseline_config.template.json`, fill every provenance/adaptation/data/budget field, and record exact parameter counts per bit width; registration recomputes the reported average bit width.

## Registration

Once the model's SG-MMP selection exists, register the official result:

```bash
python experiments/revision_full/external_baselines.py register \
  --model qwen05 \
  --method tacq \
  --average-bits <measured_parameter_weighted_bits> \
  --samples <canonical_1319_jsonl> \
  --source-url https://github.com/The-Inscrutable-X/TACQ \
  --source-commit cfc4cccfb6b7d6f7d184c9fbc8f9373c3e74569a \
  --config <completed_external_config.json>
```

Registration fails unless all item IDs and required fields exist and the reported budget is within 0.05 bit of SG-MMP. It copies and hashes both samples and configuration. Validate before analysis:

```bash
python experiments/revision_full/external_baselines.py validate
```

The resubmission gate requires both TaCQ and HAWQ-V2 registrations for `qwen05` and `qwen15`. The official HAWQ repository is `https://github.com/zhen-dong/hawq`; clone it in an isolated environment and record the actual `git rev-parse HEAD`. Its released pipeline targets convolutional models and is not a drop-in causal-LM evaluator. An LLM adaptation may be registered only if it preserves HAWQ-V2's Hessian-aware automated precision-allocation rule, uses no GSM8K test information, records every model-specific change, reports actual bit accounting, and exports the canonical JSONL contract. Describe it as an adaptation where appropriate rather than implying an unmodified official LLM implementation. Register it with `--method hawq_v2`. Do not relabel the internal diagonal-Hessian placement control as HAWQ-V2.

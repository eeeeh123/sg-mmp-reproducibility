# TaCQ and established automated mixed-precision baseline integration

TaCQ is deliberately isolated from the main environment because its official repository specifies Python 3.12.2/CUDA and a separate dependency stack. The revision pipeline does not reimplement TaCQ or relabel the internal diagonal-Hessian heuristic as TaCQ/HAWQ-V2.

## Reproducibility lock

- Official repository: `https://github.com/The-Inscrutable-X/TACQ`
- Commit pinned on 2026-09-01: `cfc4cccfb6b7d6f7d184c9fbc8f9373c3e74569a`
- Run the official code in a separate environment on the laboratory server.
- Save the exact command, configuration, environment/package lock, model revision, calibration source/seed, selected precision allocation, and parameter-weighted average bit width.

The official scripts currently demonstrate larger Llama/Qwen models. Adapting a primary model is allowed only by changing model-specific loading/configuration while preserving the official TaCQ importance and allocation procedure. Record every adaptation in the imported config; do not describe an approximate local surrogate as TaCQ.

## Canonical evaluation contract

After TaCQ produces a quantized model/state, evaluate it with the same direct 5-shot greedy GSM8K evaluator used by `run.py evaluate-full`. Export a JSONL file containing exactly one row for each `doc_id` 0-1318, including at minimum:

```json
{"doc_id": 0, "correct": 0, "generation": "..."}
```

Use the same prompt construction, answer extractor, maximum generation length, official GSM8K test ordering, and model chat/raw prompt style as the internal methods. Do not import an accuracy number without per-example generations.

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
  --config <complete_run_config_or_environment_lock>
```

Registration fails unless all item IDs and required fields exist and the reported budget is within 0.05 bit of SG-MMP. It copies and hashes both samples and configuration. Validate before analysis:

```bash
python experiments/revision_full/external_baselines.py validate
```

The resubmission gate requires both official TaCQ and HAWQ-V2 registrations for `qwen05` and `qwen15`. Run HAWQ-V2 from a pinned official source revision in its own environment, preserve its automated sensitivity/allocation procedure, evaluate through the same canonical JSONL contract, and register it with `--method hawq_v2`. Do not relabel the internal diagonal-Hessian placement control as HAWQ-V2.

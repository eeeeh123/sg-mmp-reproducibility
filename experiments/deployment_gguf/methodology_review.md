# Deployment-cost methodology review — 2026-09-09

## Material Passport

- Task: review and implement cost-measurement eligibility appropriate to allocation transfer.
- Evidence: official public repositories inspected on 2026-09-09; local Q4 train-only diagnostics.
- Status: ANALYZED; server measurements of the revised operational check are pending.
- Scope: observed sources below, not an exhaustive survey of journals or every paper version.

## What went wrong

CUDA kernels are software functions executed on a GPU. No observed record proves
a hardware fault or a kernel bug. The immediate interruption was a Python gate
rejecting CPU/CUDA distribution differences under a 0.10 decision-gap rule.
The exact low-level origin of those differences remains unresolved. Treating
cross-backend equivalence as prerequisite to measuring a single target backend
was not justified for this research question. Repeatedly changing tolerances or
adding FA diagnostics does not establish the validity of that prerequisite.

## Official implementation evidence

1. [AWQ, MLSys 2024](https://github.com/mit-han-lab/llm-awq):
   [TinyChat benchmark](https://github.com/mit-han-lab/llm-awq/blob/main/tinychat/benchmark.py)
   chooses W4A16/W16A16, controls context length and FA, warms up the GPU/model,
   synchronizes CUDA and repeats timing. The text-model benchmark initializes
   weights rather than loading the real checkpoint, illustrating that synthetic
   performance measurement must not be confused with quality validation. We
   retain real artifacts for both measurements instead of copying this detail.
2. [GPTQ, ICLR 2023](https://github.com/IST-DASLab/gptq):
   [opt.py benchmark](https://github.com/IST-DASLab/gptq/blob/main/opt.py)
   feeds a fixed input token sequence incrementally with KV caching, synchronizes
   devices and reports median time. Its optional `check` computes perplexity;
   the inspected timing function does not demand CPU/CUDA token identity.
3. [Marlin](https://github.com/IST-DASLab/marlin): an additional kernel-engineering
   example, not presented here as a verified top-journal publication. The official
   repository separates [correctness tests](https://github.com/IST-DASLab/marlin/blob/master/test.py)
   from [benchmarks](https://github.com/IST-DASLab/marlin/blob/master/bench.py).
   Its tests build a dequantized reference for the quantized matrix computation.
   Its model evaluation reports perplexity/accuracy using deployed kernels.

The absence of the disputed gate in these inspected functions is not proof that
the authors performed no other correctness checks. Their examples support a
separation of numerical implementation validation, performance and task quality.
They do not validate our Q4 kernel or provide a transferable numerical tolerance.

## Implemented correction

- Keep immutable model/artifact hashes, tensor-type audits, FP16 conversion and
  pinned official backend checks. Use existing stock llama.cpp kernels.
- Add `deployment-check`: same target CUDA, FA on, f16 KV cache for every method;
  eight fixed train prompts, complete 16-token continuations, finite complete
  first-position top-8 distributions. Save provenance and operational result in
  `deployment__<method>.json`. This is deliberately only a runtime smoke check.
- Benchmark, quality and stage readiness require this new policy-bound record.
  Server plan generation uses the new command. Never auto-promote old failures.
- Keep CPU/CUDA/FA comparisons and their original records as optional diagnostics.
  No new equivalence assertion, no raised numerical threshold, no GPU modification.
- Retain the existing fixed-workload microbenchmark and fixed 128-output-token
  service measurement, warmup, repeated process blocks, idle-GPU check, paired
  comparisons, and separate model/storage/peak-memory accounting.
- Measure task quality with the actual GGUF artifacts on the same target backend.
  Accuracy is an outcome, not a condition to tune away. A successful operational
  check alone cannot support claims of correct kernels or retained task quality.

## Claim boundary

Report costs of FP16/Q4_0/Q5_0/SG-mixed GGUF on the specified backend, hardware and
workload. GGUF re-quantization transfers the frozen precision allocation; it is
not bit-exact deployment of the original Python GPTQ checkpoint. File size alone
is not GPU memory, and lower bit width does not guarantee lower latency.
Performance plus actual deployed quality supports a trade-off claim; operational
checks alone do not. Serious crashes, non-finite outputs or unexplained quality
collapse still warrant investigation rather than unqualified performance claims.

## Server execution

### Conversion acceptance amendment

The subsequent SmolLM train-only conversion record had identical tokenizer IDs,
all 128 free-running tokens and all 128 teacher-forced winners. One position
exceeded the legacy 0.10 score-gap rule (0.11655 drift; margins 6.8594/6.9759).
Following explicit user authorization, conversion acceptance now checks finite
complete top-8 rows, overlap >=7 and presence of both top-two candidate sets.
An unchanged winner is accepted without a score-drift veto. Flipped winners
retain the previous bounded near-tie exception. Numerical drift and the old
numerical-policy verdict remain in every row. This is not a claim of distribution
equivalence. The protocol records that this revision followed observed train
diagnostics; it is not presented as an original preregistered criterion.

Rerun conversion and deployment checks for both value-stage models after updating;
the new conversion policy hash rejects old conversion evidence downstream.
Previous conversion attempts are archived automatically. Existing unchanged
artifacts and matching completed cost blocks remain reusable. The generated
value plan performs these checks before reaching task quality.

```bash
python -m unittest experiments.deployment_gguf.test_deployment_gguf
for method in fp16 q4 q5 sg; do
  python -m experiments.deployment_gguf.run deployment-check \
    --model qwen05 --method "$method" --gpu 0 \
    --llama-cpp-dir /data/experiment/LQ/llama.cpp-deployment-gguf-v1 || break
done
```

Existing server plans generated before this correction should be regenerated;
their explicit `packed-gate` commands still run the old diagnostic and can fail.

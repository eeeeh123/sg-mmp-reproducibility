# Direct GSM8K-500 paired statistics

Fixed GSM8K test subset: n=500, seed=20260615. All reported base-model rows use the lm-eval-style `Question: ...\nAnswer:` 5-shot prompt.

| Model | GPTQ-W4 | SG-MMP | Delta | GPTQ wrong / SG correct | GPTQ correct / SG wrong | McNemar p | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-0.5B | 16.80 | 26.80 | +10.00 | 77 | 27 | 9.702e-07 | [+6.20, +14.00] |
| Qwen2.5-1.5B | 46.00 | 56.20 | +10.20 | 84 | 33 | 2.673e-06 | [+6.00, +14.40] |
| SmolLM-1.7B | 18.80 | 25.80 | +7.00 | 66 | 31 | 0.0004902 | [+3.20, +10.80] |
| TinyLlama-1.1B-intermediate-step-1431k-3T | 0.80 | 2.40 | +1.60 | 10 | 2 | 0.03857 | [+0.40, +3.00] |
| gemma-2-2b-it | 47.20 | 50.40 | +3.20 | 49 | 33 | 0.09703 | [-0.40, +6.80] |

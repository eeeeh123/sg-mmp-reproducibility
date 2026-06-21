# Llama-3.2-1B-Instruct Validation

This experiment adds a stronger LLaMA-family candidate for the SG-MMP paper.
Unlike TinyLlama, this checkpoint is expected to have non-trivial GSM8K ability,
so it is a better candidate for a fourth main-family validation if the local
FP16 screen passes.

## Current Blocker

`meta-llama/Llama-3.2-1B-Instruct` is a gated Hugging Face repository. The
current machine has no configured Hugging Face token, and `hf-mirror.com`
returns `GatedRepo`.

Do not paste a token into chat. Configure it locally instead:

```powershell
huggingface-cli login
```

or for the current PowerShell session:

```powershell
$env:HF_TOKEN = "<your-read-token>"
```

If direct Hugging Face access is blocked, keep the mirror endpoint:

```powershell
$env:PTQ_HF_DOWNLOAD_ENDPOINT = "https://hf-mirror.com"
```

## Execution Plan

Stage 0: download and inspect.

```powershell
python -B experiments\fix_llama32\run.py download
python -B experiments\fix_llama32\run.py inspect
```

Stage 1: FP16 candidate screen on the shared direct GSM8K protocol.

```powershell
python -B experiments\fix_gsm8k_500\direct_eval.py run --models llama32 --methods fp16 --n 200 --batch-size 2 --max-new-tokens 256 --force
```

Continue only if FP16 is clearly above the floor, preferably >= 25%.

Stage 2: sensitivity screen on GSM8K train examples.

```powershell
python -B experiments\fix_llama32\launch.py --name llama32_screen_train300 screen --screen-n 300 --batch-size 2 --max-new-tokens 256 --force
```

Stage 3: quantize GPTQ-W4 and SG-MMP.

```powershell
python -B experiments\fix_llama32\launch.py --name llama32_gptq_quant quantize-gptq --calib-samples 128 --calib-length 2048
python -B experiments\fix_llama32\launch.py --name llama32_sg_quant quantize-sg --calib-samples 128 --calib-length 2048 --top-k 4
```

Stage 4: smoke-load and GSM8K-500 paired validation.

```powershell
python -B experiments\fix_gsm8k_500\direct_eval.py smoke-load --models llama32 --methods gptq,sg
python -B experiments\fix_gsm8k_500\launch_direct.py --name llama32_gsm8k500 run --models llama32 --methods fp16,gptq,sg --n 500 --batch-size 2 --max-new-tokens 256
python -B experiments\fix_gsm8k_500\direct_eval.py analyze --n 500
```

## Interpretation Gate

Use Llama-3.2 as a main robustness model only if:

- FP16 GSM8K-500 is not near the floor.
- GPTQ-W4 shows a meaningful quantization drop.
- SG-MMP recovers a clear portion of that drop.
- Outputs do not show TinyLlama-like prompt repetition or continuation of
  few-shot examples.

Otherwise, report it as a boundary or appendix result.

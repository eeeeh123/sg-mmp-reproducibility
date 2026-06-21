# Gemma-2-2B-it Validation

This experiment adds `google/gemma-2-2b-it` as a candidate fourth model family
for the SG-MMP paper.

Local model path:

```text
models/gemma-2-2b-it
```

Required files:

```text
config.json
generation_config.json
model-00001-of-00002.safetensors
model-00002-of-00002.safetensors
model.safetensors.index.json
special_tokens_map.json
tokenizer.json
tokenizer_config.json
```

Execution gate:

1. Check local files and inspect the model.
2. Run FP16 GSM8K-200 first.
3. Continue to layer screening and quantization only if FP16 is not near the
   floor and outputs do not show TinyLlama-like repetition.

Commands:

```powershell
python -B experiments\fix_gemma2\run.py check-files
python -B experiments\fix_gemma2\run.py inspect
python -B experiments\fix_gsm8k_500\direct_eval.py run --models gemma2 --methods fp16 --n 200 --batch-size 1 --max-new-tokens 256 --force
```

If FP16 passes the gate:

```powershell
python -B experiments\fix_gemma2\launch.py --name gemma2_screen_train300 screen --screen-n 300 --batch-size 1 --max-new-tokens 256 --force
python -B experiments\fix_gemma2\launch.py --name gemma2_gptq_quant quantize-gptq --calib-samples 128 --calib-length 2048
python -B experiments\fix_gemma2\launch.py --name gemma2_sg_quant quantize-sg --calib-samples 128 --calib-length 2048 --top-k 4
python -B experiments\fix_gsm8k_500\direct_eval.py smoke-load --models gemma2 --methods gptq,sg
python -B experiments\fix_gsm8k_500\launch_direct.py --name gemma2_gsm8k500 run --models gemma2 --methods fp16,gptq,sg --n 500 --batch-size 1 --max-new-tokens 256
python -B experiments\fix_gsm8k_500\direct_eval.py analyze --n 500
```

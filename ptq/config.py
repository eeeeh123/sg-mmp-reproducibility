"""所有实验配置集中管理。"""

MODELS = [
    {"name": "Qwen2.5-0.5B", "hf_id": "Qwen/Qwen2.5-0.5B", "params_b": 0.5},
    {"name": "Qwen2.5-1.5B", "hf_id": "Qwen/Qwen2.5-1.5B", "params_b": 1.5},
    # Historical result files use the short ``SmolLM-1.7B`` prefix, but this
    # checkpoint was downloaded from the SmolLM2 model card below.
    {"name": "SmolLM-1.7B", "display_name": "SmolLM2-1.7B", "hf_id": "HuggingFaceTB/SmolLM2-1.7B", "params_b": 1.7},
]

QUANT_CONFIGS = [
    {"method": "fp16", "bits": 16, "group_size": None, "desc": "FP16 baseline"},
    {"method": "rtn", "bits": 4, "group_size": 128, "desc": "Round-to-Nearest 4-bit"},
    {"method": "gptq", "bits": 4, "group_size": 128, "desc": "GPTQ 4-bit"},
    {"method": "awq", "bits": 4, "group_size": 128, "desc": "AWQ 4-bit"},
    {"method": "smoothquant", "bits": 8, "group_size": None, "desc": "SmoothQuant W8A8"},
    {"method": "mixed_precision", "bits": "4+8", "group_size": 128, "desc": "Mixed: attn-INT8 + ffN-GPTQ-W4"},
]

DOWNSTREAM_TASKS = ["mmlu", "hellaswag", "arc_challenge", "gsm8k"]

TASK_FEWSHOT = {
    "mmlu": 5,
    "hellaswag": 10,
    "arc_challenge": 0,
    "gsm8k": 5,
}

# Headline benchmark policy: every task, including all 1,319 GSM8K test
# examples, is evaluated in full. Development/sensitivity screening must use
# GSM8K train data and is implemented separately under experiments/revision_full.
TASK_LIMIT = {
    "arc_challenge": None,
    "hellaswag": None,
    "mmlu": None,
    "gsm8k": None,
}

# 校准数据配置
CALIB_SAMPLES = 128
CALIB_MAX_LENGTH = 1024
CALIB_DATASET = "allenai/c4"

# 困惑度评测配置
WIKITEXT_DATASET = "wikitext-2-raw-v1"
EVAL_STRIDE = 512
EVAL_MAX_LENGTH = 2048

# 任务按耗时升序（先跑快的，快速积累结果）
TASKS_ORDER = ["arc_challenge", "hellaswag", "mmlu", "gsm8k"]

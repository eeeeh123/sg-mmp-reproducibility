"""下载所有 baseline 模型到本地。

用法: python scripts/01_download_models.py
"""

import sys
sys.path.insert(0, ".")

from ptq.config import MODELS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

SAVE_ROOT = "models"


def main():
    for model_cfg in MODELS:
        hf_id = model_cfg["hf_id"]
        name = model_cfg["name"]
        save_path = f"{SAVE_ROOT}/{name}"

        print(f"\n{'='*60}")
        print(f"Downloading {name} ({hf_id})")
        print(f"{'='*60}")

        print("  Downloading tokenizer...")
        tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        tok.save_pretrained(save_path)

        print("  Downloading model (FP16)...")
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.save_pretrained(save_path)

        print(f"  Saved to {save_path}")


if __name__ == "__main__":
    main()

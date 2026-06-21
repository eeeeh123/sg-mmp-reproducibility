# Model provenance

## Canonical checkpoint identifiers

| Paper label | Upstream identifier | Role |
|---|---|---|
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | Primary model |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | Primary model |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B` | Primary model |
| Gemma-2-2B-it | `google/gemma-2-2b-it` | Boundary-family check |
| TinyLlama-1.1B intermediate | `TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T` | Floor-effect check |

## SmolLM naming correction

The experiment download entry in `ptq/config.py` specifies
`HuggingFaceTB/SmolLM2-1.7B`. `scripts/01_download_models.py` loads that
identifier and saves it under `models/<name>`. Historical experiments used the
local directory and result-file prefix `SmolLM-1.7B`; this is a storage label,
not the checkpoint identity. The paper and this release therefore use the
canonical name **SmolLM2-1.7B**.

The released result tables preserve their historical keys for traceability.
Figure labels and manuscript text normalize them to SmolLM2-1.7B. Do not
substitute the original SmolLM-1.7B checkpoint when reproducing the results.

## Redistribution boundary

No model checkpoint is redistributed here. Users must retrieve each checkpoint
from the listed upstream source and comply with its license or access terms.

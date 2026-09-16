# Model provenance

## Revision-full-v4 primary checkpoints

The completed four-model revision generated a snapshot manifest before running
the experiments. The resolved upstream revisions were:

| Paper label | Upstream identifier | Resolved revision |
|---|---|---|
| Qwen2.5-0.5B | `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` |
| Qwen2.5-1.5B | `Qwen/Qwen2.5-1.5B` | `8faed761d45a263340a0528343f099c05c9a4323` |
| SmolLM2-1.7B | `HuggingFaceTB/SmolLM2-1.7B` | `effd688a12921b4cc83e3312b6feb579f70f9c71` |
| Gemma-2-2B-it | `google/gemma-2-2b-it` | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` |

The Zenodo v2.0.0 revision-results archive preserves the generated
`model_snapshot_manifest.json`, including local-directory information, weight
file sizes, and SHA-256 records. These four checkpoints are all primary models
in the completed revision analysis.

TinyLlama-1.1B intermediate
(`TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T`) remains a legacy
floor-effect check and is not part of the four-model primary revision protocol.

## SmolLM naming correction

The download entry in `ptq/config.py` specifies
`HuggingFaceTB/SmolLM2-1.7B`. Historical local directories and result keys use
the prefix `SmolLM-1.7B`; that is a storage label, not the checkpoint identity.
The paper and current release use the canonical name **SmolLM2-1.7B** while
retaining historical keys where needed for traceability. Do not substitute the
original SmolLM-1.7B checkpoint.

## Legacy evidence boundary

The older v1.2 compact release recorded canonical checkpoint identifiers but
not immutable upstream revisions. It must not be described as byte-identical
checkpoint provenance. The immutable revisions above apply specifically to the
completed `revision-full-v4` and its archived snapshot manifest.

## Redistribution boundary

No pretrained checkpoint, reconstructible quantized PyTorch state, or GGUF
weight artifact is redistributed. Users must retrieve checkpoints from the
listed upstream sources and comply with their licenses or access terms.

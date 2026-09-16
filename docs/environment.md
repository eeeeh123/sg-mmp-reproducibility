# Reproduction environments

This repository contains two evidence generations. They should not be described
as one byte-identical execution environment.

## Revision-full-v4 and TaCQ extension

The completed revision experiment used two NVIDIA GeForce RTX 3090 GPUs. Its
Zenodo v2.0.0 result archive records the generated model-snapshot manifest,
dataset-snapshot manifest, frozen protocol files, per-run metadata, sample
hashes, and analysis outputs. The current rerun dependencies are pinned in
`requirements-server.txt`.

The exact resolved Hugging Face revisions used by `revision-full-v4` are listed
in `docs/model_provenance.md` and preserved in
`experiments/revision_full/outputs/model_snapshot_manifest.json` inside the
result archive. The dataset cache fingerprints and source-file hashes are
likewise preserved in
`experiments/revision_full/outputs/dataset_snapshot_manifest.json`.

Quantized PyTorch states are reconstructible intermediates and are not release
artifacts. The archive instead retains their configuration and lifecycle
metadata, persistent results, and cryptographic hashes.

## Packed GGUF deployment extension

The deployment experiment used the same dual-RTX-3090 server and a pinned
`llama.cpp` checkout:

| Component | Recorded value |
|---|---|
| GPU | 2 x NVIDIA GeForce RTX 3090 (24 GiB each) |
| NVIDIA driver | 550.163.01 |
| CUDA toolkit used to build the backend | 12.4 |
| `llama.cpp` commit | `050dde50c9d70cf207db84f7224eedc491d817b2` |

The deployment archive preserves the build/gate manifests, toolchain evidence,
artifact type and byte-accounting manifests, raw benchmark blocks, request
records, quality samples, and final analyses. GGUF model files are excluded;
the manifests provide the information required to rebuild and audit them.

## Legacy v1.2 reproduction path

The earlier compact release and its figure-generation checks used this Windows
software stack:

| Component | Version |
|---|---:|
| Python | 3.13.12 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| Transformers | 5.8.0 |
| Datasets | 4.8.5 |
| LM Evaluation Harness | 0.4.11 |

`requirements.txt` records those top-level pins. The historical compact study
did not preserve immutable upstream checkpoint revisions or a dataset
fingerprint. That limitation applies to the legacy v1.2 path, not to the later
`revision-full-v4` archive.

## Practical notes

- Wall-clock time and peak memory are hardware- and backend-dependent; they are
  measured outcomes, not byte-identical reproduction targets.
- Retrieve checkpoints and public datasets from their upstream sources under
  the applicable licenses and access terms.
- Some historical non-compact `.pt` states triggered native Windows/PyTorch
  access violations. Keep quantization and evaluation in separate processes and
  prefer the serial server plan documented for `revision-full-v4`.

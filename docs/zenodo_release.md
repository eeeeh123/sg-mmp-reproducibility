# Zenodo release checklist

Create the Zenodo record from the immutable `v1.1.0` source snapshot after the
checksums have been regenerated and verified. The manual-upload asset is:

```text
sg-mmp-reproducibility-v1.1.0.zip
```

It contains source code, documentation, configuration, and released derived
results. It intentionally excludes `.git/`, model checkpoints, quantized
states, raw benchmark caches, prompts, answers, generated traces, private
logs, manuscript files, and generated figures.

Before uploading:

```powershell
python scripts/reproduce_core.py verify-public
```

Use Zenodo resource type **Software**, version `1.1.0`, the creators and
keywords in `zenodo.json`, and the MIT license included in this package. After
the matching GitHub tag is pushed, add its exact tag URL in the Zenodo record;
do not link to a moving `main` branch as the archival source.

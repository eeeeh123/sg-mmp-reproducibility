# Zenodo v2.0.0 release

Use **New version** on the existing Zenodo record. Do not edit or replace the
published v1.2.0 file. Zenodo assigns the new version its own DOI while keeping
all versions linked by the stable concept DOI `10.5281/zenodo.21096006`.

The published identifiers for this delivery are:

- Version 2.0.0 DOI: `10.5281/zenodo.22791181`
- All-versions DOI: `10.5281/zenodo.21096006`

## Upload files

The release builder creates one upload directory containing:

| File | Role |
|---|---|
| `sg-mmp-reproducibility-v2.0.0-source.zip` | Immutable source, documentation, tests, and legacy public derived data from Git tag `v2.0.0` |
| `sg-mmp-revision-full-v4-results-v2.0.0.tar.gz` | Complete revision evidence: result/sample records, frozen manifests, state metadata, screens, selections, TaCQ registrations, and final analyses |
| `sg-mmp-tacq-execution-evidence-v2.0.0.tar.gz` | Run-era TaCQ source snapshot, registrations, paired sample records, and final readiness logs |
| `sg-mmp-deployment-gguf-results-v2.0.0.zip` | Complete four-model packed-deployment evidence, including raw performance blocks, quality records, gate history, manifests, and final ten-block analyses |
| `sg-mmp-integrity-audits-v2.0.0.zip` | Independent generation-integrity and TaCQ archive audits |
| `README_ZENODO_v2.0.0.md` | Human-readable scope and exclusions |
| `MANIFEST_v2.0.0.json` | Machine-readable file roles, sizes, SHA-256 digests, source commit, and validation counts |
| `SHA256SUMS_v2.0.0.txt` | External checksum list for every other upload file |

The result archives contain public benchmark records and model-generated
outputs needed to audit the paper. Third-party benchmark material remains
subject to its original license. The repository's MIT license applies to the
authored code, not to third-party checkpoints or datasets.

## Build and verify locally

From the repository root:

```powershell
python scripts/write_manifest.py
python scripts/reproduce_core.py verify-public
python scripts/build_zenodo_release.py `
  --revision-archive "D:\Project\ptq-benchmark\incoming\revision_full_evidence_20260914_095656.tar.gz" `
  --tacq-archive "D:\Project\ptq-benchmark\incoming\revision_full_tacq_complete_0907__8cfddce.tar.gz" `
  --deployment-root "D:\Project\ptq-benchmark\incoming\deployment_value_10\20260910_184019" `
  --audit "D:\Project\ptq-benchmark\incoming\revision_full_generation_integrity_audit_0906__50e71dd.md" `
  --audit "D:\Project\ptq-benchmark\incoming\TaCQ_artifact_audit_2026-09-07.md" `
  --output-dir "D:\Project\ptq-benchmark\zenodo_upload_v2.0.0"
```

The builder refuses a dirty tracked worktree, unsafe tar member, missing final
analysis, missing result family, forbidden model-weight extension, or stale
source checksum manifest. It prints the final upload inventory and re-verifies
all SHA-256 entries.

## Zenodo metadata

- Resource type: **Software**
- Version: **2.0.0**
- Access: **Open**
- License: **MIT** for authored code; explain the third-party-content boundary
  in the description/notes
- Title, creators, description, and keywords: copy from `zenodo.json`
- Related software URL:
  `https://github.com/eeeeh123/sg-mmp-reproducibility/tree/v2.0.0`

After all files finish uploading, compare Zenodo's displayed sizes with
`MANIFEST_v2.0.0.json`, save the draft, preview the record, and publish only
after the file list is complete. Record the newly assigned version DOI in the
manuscript; use the concept DOI when the text should always resolve to the
latest version.

## Exclusions

Do not upload pretrained checkpoints, `.pt` quantized states, GGUF weight
files, Hugging Face caches, private credentials, or the unpublished manuscript.
The archives retain hashes, resolved model revisions, backend binaries/commits,
and reconstruction commands instead.

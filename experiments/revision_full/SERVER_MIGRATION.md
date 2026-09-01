# Laboratory-server migration and execution guide

## Hardware and time budget

The resource-aware v3 command plan has 410 resumable commands. Its known GSM8K workload alone contains:

- 81,408 train-only generation cases for native layer screening;
- 105,520 full-test cases across 80 FP16/core/control evaluations;
- 237,420 full-test cases across 180 random-allocation evaluations;
- 15,828 same-item format cases, each scoring four candidates;
- plus 12 broad panels, 12 generative transfer panels, 12 shared precision-bank builds, and the causal diagnostic.

This is still a throughput problem rather than a model-capacity problem. Historical local logs show that one Qwen2.5-0.5B 300-item layer screen took 9.0 hours on the RTX 5060 Ti machine. Linear item-count scaling gives about 23.0 hours for three 256-item screens on that local machine, before the faster RTX 3090 and batch-size increase are credited. Version 3 also reduces known GSM8K generation cases from 706,244 to 424,348 (39.9%) and reduces calibration-capture model forwards from roughly 172,800 to 1,536 by sharing one capture across W4/W5/W6. A conservative initial range for one RTX 3090 is roughly 120-300 GPU-hours, or about 5-13 continuous days; this is a planning range, not a promise. Run one Qwen-0.5B screen split and one seed-41 precision-bank build first, then replace the range with measured server throughput.

| Configuration | Feasibility | Planning wall time |
|---|---|---|
| 1× RTX 3090 24 GiB, 64 GiB RAM, at least 500 GiB free NVMe | Recommended plan for the available laboratory hardware | about 5-13 continuous days; calibrate with the pilot |
| 1× RTX 3090 24 GiB, 32 GiB RAM | GPU capacity is adequate, but shared-bank construction may pressure host RAM | use a reduced in-memory capture mode only after profiling; 64 GiB RAM is safer |
| 2× 24 GiB GPUs, 96 GiB RAM, 500 GiB free NVMe | Optional future acceleration | about 3-7 days if model shards run independently |

The current state format stores int8 code tensors rather than true packed 4/5/6-bit files. Each bank contains reusable W4/W5/W6/W8 entries, so banks, core states, and controls can still occupy roughly 200-250 GiB before caches and backups. Keep at least 500 GiB free; 1 TiB is preferable if periodic full backups are retained, even though the four source model folders total only about 11.9 GiB.

## 1. Clone code; do not put model weights in GitHub

The GitHub repository intentionally excludes `models/`, quantized states,
generated outputs, caches, archives, and secrets. Do not use Git LFS for the
four checkpoints: it adds quota and clone failure modes without improving the
experiment. Clone only the source code on the Linux server:

```bash
cd /data/$USER
git clone https://github.com/eeeeh123/sg-mmp-reproducibility.git ptq-benchmark
cd ptq-benchmark
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
```

The tested main environment uses PyTorch 2.11/CUDA 12.8. TaCQ and HAWQ-V2 must use isolated environments; do not install them into this one.

Set caches on a large persistent disk, not a small home partition:

```bash
export HF_HOME=/data/$USER/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE"
```

Persist these exports in the job script or shell profile used for the experiment.

## 2. Download models directly on the server

The model downloader resolves each upstream revision to an immutable Hugging
Face commit SHA, resumes interrupted files, and writes the resolved provenance
to `experiments/revision_full/outputs/model_snapshot_manifest.json`. Download the three public models
first:

```bash
python experiments/revision_full/download_models.py --models qwen05 qwen15 smollm
```

Gemma is gated. Sign in to Hugging Face, accept the Gemma license at
`https://huggingface.co/google/gemma-2-2b-it`, and authenticate on the server
without committing or printing the token:

```bash
hf auth login
python experiments/revision_full/download_models.py --models gemma2
```

Expected local directories are `models/Qwen2.5-0.5B`,
`models/Qwen2.5-1.5B`, `models/SmolLM-1.7B` (the checkpoint is
SmolLM2-1.7B), and `models/gemma-2-2b-it`. Their combined weights are about
11.9 GiB. Rerunning the same command resumes against the SHA recorded in the
manifest rather than silently moving to a newer checkpoint.

If the official Hugging Face endpoint is blocked, the public models may be
downloaded through a mirror:

```bash
python experiments/revision_full/download_models.py \
  --models qwen05 qwen15 smollm \
  --endpoint https://hf-mirror.com
```

Use the official authenticated endpoint for Gemma. If compute nodes have no
internet, run the downloader on a networked login node that shares `/data`, or
ask the administrator to pre-stage the four pinned snapshots. A physical-disk
fallback should use exFAT/NTFS or split archives; FAT32 cannot hold Gemma's
approximately 5-GB first shard.

## 3. Download data and run fail-fast checks

```bash
python experiments/revision_full/download_core_datasets.py
python -m unittest discover -s experiments/revision_full -p "test*.py" -v
python experiments/revision_full/run.py prepare --force
python experiments/revision_full/format_control.py --prepare-only --force
python experiments/revision_full/server_preflight.py
```

The preflight checks CUDA, package versions, all four local models, GSM8K/WikiText caches, protocol lock, RAM, and disk. Do not launch long runs until it prints `"ready": true`.

The remaining lm-evaluation-harness datasets are downloaded on first use. If compute nodes have no internet, run one short job on a networked login/download node with the same `HF_HOME`, or ask the administrator to pre-stage that cache.

## 4. Run on one GPU

For a single GPU, the safest complete sequence is:

```bash
python experiments/revision_full/make_server_plan.py > server_all.sh
CUDA_VISIBLE_DEVICES=0 bash server_all.sh 2>&1 | tee server_all.log
```

This is resumable but long. Completed screens, per-item generations, task panels, and atomic quantized states are skipped when rerun. Do not add `--force` when resuming a normal interruption.

## 5. Optional multi-GPU execution

Run the global preparation once, then create one model plan per GPU:

```bash
mkdir -p server_plans logs
python experiments/revision_full/make_server_shard.py --model qwen05 > server_plans/qwen05.sh
python experiments/revision_full/make_server_shard.py --model qwen15 > server_plans/qwen15.sh
python experiments/revision_full/make_server_shard.py --model smollm > server_plans/smollm.sh
python experiments/revision_full/make_server_shard.py --model gemma2 > server_plans/gemma2.sh
```

Start four persistent `tmux` sessions:

```bash
tmux new-session -d -s qwen05 "cd /data/$USER/ptq-benchmark && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=0 bash server_plans/qwen05.sh 2>&1 | tee logs/qwen05.log"
tmux new-session -d -s qwen15 "cd /data/$USER/ptq-benchmark && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=1 bash server_plans/qwen15.sh 2>&1 | tee logs/qwen15.log"
tmux new-session -d -s smollm "cd /data/$USER/ptq-benchmark && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=2 bash server_plans/smollm.sh 2>&1 | tee logs/smollm.log"
tmux new-session -d -s gemma2 "cd /data/$USER/ptq-benchmark && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=3 bash server_plans/gemma2.sh 2>&1 | tee logs/gemma2.log"
```

Monitor without interrupting jobs:

```bash
nvidia-smi
tmux ls
tail -f logs/qwen05.log
```

Each process sees its assigned physical GPU as `cuda:0`. All model-specific artifacts have distinct paths. The shared `outputs/status.json` is only a latest-status display and is not used as numerical evidence.

## 6. Final analysis and external work

After all four model sessions finish:

```bash
source .venv/bin/activate
python experiments/revision_full/analyze.py
python experiments/revision_full/readiness.py --stage core
```

Then complete the blinded annotation sheets and the isolated official TaCQ and HAWQ-V2 runs described in `TACQ_INTEGRATION.md`. Register both methods, rerun analysis, and require:

```bash
python experiments/revision_full/readiness.py --stage resubmission
```

## 7. Download results and back up

On the server:

```bash
tar -czf revision_full_outputs.tar.gz experiments/revision_full/outputs logs
```

On Windows PowerShell:

```powershell
scp username@server:/data/username/ptq-benchmark/revision_full_outputs.tar.gz .
```

Keep periodic copies of `experiments/revision_full/outputs/`. Precision banks and states are expensive to reproduce; sample JSONL files are the irreplaceable basis of paired statistical analysis.

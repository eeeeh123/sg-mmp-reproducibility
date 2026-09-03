# Two-RTX-3090 server migration and execution guide

This guide is for `revision-full-v4`. Run commands from the repository root. Do not upload checkpoints to GitHub; download and verify them directly on the server.

## Capacity and realistic duration

The generated plan contains 349 top-level shell steps; six model/family loops expand from each locked selection manifest so an exhaustively smaller layer-allocation family is handled without nonexistent jobs. State/bank/screen cleanup remains fail-closed. The expensive work is not the cleanup:

- 81,408 train-only generations for native layer screening;
- 105,520 complete-test generations for 80 FP16/core/placement runs;
- up to 237,420 complete-test generations for 180 requested random allocations; a model with fewer than 30 feasible whole-layer placements uses its exhaustive distinct set and records the smaller actual count;
- 15,828 same-item MCQ cases, each scoring four candidates;
- 12 broad panels, 12 generative transfer panels, 12 precision-bank builds;
- a 200-item teacher-forced diagnostic at every Qwen-0.5B layer for block, attention, and MLP interventions.

With two 24-GiB RTX 3090 cards, the internal core is expected to take roughly 3–7 continuous days. This is a planning range, not a guarantee; actual prompt lengths, generated lengths, host RAM, shared-disk speed, and quantization kernels dominate. The 32-GiB mode still runs two single-GPU workers, but serializes activation-heavy screen/precision builders. Evaluation can remain concurrent, so it should retain a material dual-GPU speedup, although not a full 2x speedup. Use the first screen and precision-bank timings to update the estimate. Official TaCQ/HAWQ-V2 adaptation is additional work.

For two concurrent model processes, the estimated active state peak is 17.78 GiB. On one shared filesystem the code requires about 55 GiB free and recommends about 92 GiB free. These are **free-space** values after the Python environment, four source checkpoints (about 11.9 GiB), and dataset caches exist. A 100-GiB total quota may therefore be inadequate; trust the measured preflight, not the quota label.

System RAM and disk free space are independent capacities: for example, `31 GiB` in `free -h` is RAM, while `103 GiB available` in `df -h /data` is disk. GPU VRAM and swap do not increase system RAM. Full concurrent building still needs at least 64 GiB RAM (96 GiB preferred). The guarded low-RAM schedule supports two GPU workers on a 32-GiB host only when total RAM is at least 30 GiB, at least 24 GiB is available before launch/build, and `REVISION_FULL_MAX_CONCURRENT_RAM_BUILDERS=1`. A 48-GiB-or-larger host remains preferable. If preflight reports `ready: false`, do not bypass its gate.

## 1. Clone source and create the environment

```bash
cd /data/experiment/LQ
git clone https://github.com/eeeeh123/sg-mmp-reproducibility.git
cd sg-mmp-reproducibility
conda create -p /home/ubuntu/anaconda3/envs/LQ-sgmmp python=3.12 -y
conda activate /home/ubuntu/anaconda3/envs/LQ-sgmmp
python -m pip install --upgrade pip
python -m pip install -r requirements-server.txt
```

Create the project-local launcher configuration. It selects this conda
environment only in shells that source it; it does not modify global shell or
conda configuration. During online staging, it also places the Hugging Face
token and every download cache below `/data/experiment/LQ`:

```bash
cp experiments/revision_full/server_env.template.sh server_env.sh
export REVISION_FULL_ONLINE_STAGING=1
source server_env.sh
which python
python --version
```

Optional separate scratch for reconstructible states:

```bash
export REVISION_FULL_STATE_DIR=/scratch/$USER/sg-mmp-revision-states
mkdir -p "$REVISION_FULL_STATE_DIR"
```

Keep these exports identical in every later shell/tmux session.

## 2. Stage immutable models and all datasets while online

```bash
python experiments/revision_full/download_models.py --models qwen05 qwen15 smollm
hf auth login
python experiments/revision_full/download_models.py --models gemma2
python experiments/revision_full/download_core_datasets.py
```

Gemma requires accepting its Hugging Face license. The model downloader resolves an immutable upstream commit and hashes every weight shard. The dataset downloader resolves all core and panel tasks—including the custom generative ASDiv task—and records row counts, fingerprints, and cache-file hashes. If compute nodes lack internet, run these commands on a networked login node sharing the same `/data` and cache paths.

After staging, freeze network dependence:

```bash
unset REVISION_FULL_ONLINE_STAGING
source server_env.sh
```

## 3. Select the only allowed batch settings before formal output

Start conservatively at generation batch 4 per process and MCQ item batch 2 per process:

```bash
export REVISION_FULL_EVAL_BATCH_SIZE=4
export REVISION_FULL_FORMAT_BATCH_SIZE=2
export REVISION_FULL_MAX_CONCURRENT_RAM_BUILDERS=1
export REVISION_FULL_MIN_AVAILABLE_RAM_GIB=24
export REVISION_FULL_RAM_BUILDER_WAIT_POLL_SECONDS=30
export REVISION_FULL_RAM_BUILDER_WAIT_TIMEOUT_SECONDS=0
source server_env.sh
```

Smoke all architectures using GSM8K train only. The commands may be run in two terminals, one sequence per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/revision_full/run.py smoke-eval --model gemma2 --batch-size 4 --format-batch-size 2
CUDA_VISIBLE_DEVICES=0 python experiments/revision_full/run.py smoke-eval --model qwen15 --batch-size 4 --format-batch-size 2
CUDA_VISIBLE_DEVICES=1 python experiments/revision_full/run.py smoke-eval --model smollm --batch-size 4 --format-batch-size 2
CUDA_VISIBLE_DEVICES=1 python experiments/revision_full/run.py smoke-eval --model qwen05 --batch-size 4 --format-batch-size 2
```

If any smoke test OOMs, change the environment globally and in `server_env.sh` to `2` and `1`, rerun all four smoke tests, and use those values everywhere. Two GPUs do not justify doubling a per-process batch. Once `prepare` creates the v4 lock or any formal sample exists, do not change batch size; the pipeline rejects mixed-batch resume files.

## 4. Lock, verify, and make the GO/NO-GO decision

```bash
source server_env.sh
python -m unittest discover -s experiments/revision_full -p "test*.py" -v
python experiments/revision_full/run.py prepare --force
python experiments/revision_full/format_control.py --prepare-only --force
python experiments/revision_full/server_preflight.py --expected-gpus 2 --concurrent-models 2
```

The preflight checks both visible GPUs, package compatibility, total and currently available system RAM, free space on persistent/state filesystems, model revisions and hashes, every dataset cache hash, offline mode, the v4 protocol lock, full-test size, batch settings, and format manifest. In low-RAM mode it must report `serialized_ram_builders_with_parallel_gpu_evaluation`. Do not launch the matrix unless the final JSON contains `"ready": true`.

Hashing all weight and dataset files reads several gigabytes once and may take minutes on a shared disk. That is intentional: it finds corruption or a wrong cache before multi-day computation.

Before starting the long shards, run the first formal screen-state build for the two largest state estimates, one command per GPU/terminal, with `server_env.sh` sourced in both terminals:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/revision_full/run.py build-screen-bank --model gemma2 --split-id 0 --calib-seed 41
CUDA_VISIBLE_DEVICES=1 python experiments/revision_full/run.py build-screen-bank --model smollm --split-id 0 --calib-seed 41
```

The RAM-builder lock intentionally lets only one of these commands build at a time; the other prints `ram_builder_wait` and continues after the first releases the lock. If Linux has not yet reclaimed enough memory, it prints `ram_builder_memory_wait` and rechecks every 30 seconds instead of terminating the shard. The default `REVISION_FULL_RAM_BUILDER_WAIT_TIMEOUT_SECONDS=0` waits without an automatic timeout; set a positive value only if the site requires a bounded queue wait. This validates real calibration, quantization, CUDA, RAM, the lock, and state writes before hundreds of evaluations. The later shards validate and reuse these exact formal states, so the pilot is not discarded work.

On the 31-GiB host, also build the largest estimated full precision bank **before** launching either long shard:

```bash
source server_env.sh
CUDA_VISIBLE_DEVICES=0 python experiments/revision_full/run.py build-bank --model gemma2 --calib-seed 41
free -h
python experiments/revision_full/server_preflight.py --expected-gpus 2 --concurrent-models 2
```

`build-bank` depends only on the frozen protocol, checkpoint, and WikiText calibration—not on the later layer selection—so this is valid preregistered work and the Gemma shard will reuse it byte-for-byte. It is the mandatory empirical RAM acceptance test that local CPU-only unit tests cannot replace. Its runtime guard refuses to start below 24 GiB `MemAvailable`. The repeated preflight verifies the bank metadata and remaining disk after the real peak artifact exists. If the pilot or repeated preflight fails, do not launch the shards; no test-set accuracy has been generated and no other valid evidence needs to be rerun.

## 5. Generate two non-overlapping model shards

Balance the expensive full-set random-allocation evaluations as well as model
size. Gemma has no random-allocation block, so pair it with Qwen-1.5B; pair
SmolLM with Qwen-0.5B (whose extra causal diagnostic is comparatively smaller):

```bash
source server_env.sh
mkdir -p server_plans logs
python experiments/revision_full/make_server_shard.py --models gemma2 qwen15 > server_plans/gpu0.sh
python experiments/revision_full/make_server_shard.py --models smollm qwen05 > server_plans/gpu1.sh
```

Start persistent sessions. Ensure the cache, offline, state-directory, and batch exports above are visible inside both sessions (placing them in a small `server_env.sh` and sourcing it is convenient):

```bash
tmux new-session -d -s revision_gpu0 "cd /data/experiment/LQ/sg-mmp-reproducibility && source server_env.sh && set -o pipefail && CUDA_VISIBLE_DEVICES=0 bash server_plans/gpu0.sh 2>&1 | tee -a logs/gpu0.log"
tmux new-session -d -s revision_gpu1 "cd /data/experiment/LQ/sg-mmp-reproducibility && source server_env.sh && set -o pipefail && CUDA_VISIBLE_DEVICES=1 bash server_plans/gpu1.sh 2>&1 | tee -a logs/gpu1.log"
```

Each process sees its physical card as `cuda:0`. Model-specific status, runtime summaries, samples, state metadata, and cleanup receipts use separate paths. Dataset/model caches are shared read-only during formal execution. Both GPUs can evaluate simultaneously; only `build-screen-bank` and `build-bank`, which retain calibration activations and growing quantized states in host RAM, are mutually exclusive.

Monitor without modifying outputs:

```bash
nvidia-smi
free -h
tmux ls
tail -f logs/gpu0.log
df -h /data/experiment/LQ
du -sh experiments/revision_full/outputs "${REVISION_FULL_STATE_DIR:-experiments/revision_full/outputs/states}"
```

The plan creates one calibration bank and at most one materialized state per process. Cleanup occurs immediately after all consumers of that state pass exact completeness checks. It hashes small persistent evidence and metadata, writes a receipt, and deletes only reconstructible `.pt` files. Interrupted state writes remove their process-specific `.tmp` files automatically. This adds little GPU time; shared-disk I/O is the main overhead. Do not add `--force` during normal resume. The launch commands use `pipefail` so a failed shard is not reported as successful merely because `tee` exited normally, and `tee -a` preserves earlier diagnostic logs on resume.

If a shard stops, inspect the last traceback, fix the external cause, and rerun the same shard. Completed v4 rows are validated and skipped; partial rows resume only if IDs, provenance, batch, and decoding settings match. Never concatenate JSONL files manually.

## 6. Core analysis, annotation, and external baselines

After both shards finish:

```bash
python experiments/revision_full/analyze.py
python experiments/revision_full/readiness.py --stage core
```

`core` must pass before using any new number. Then annotate each primary model's generated blinded CSV. Every row needs `consensus_output_a_label` and `consensus_output_b_label`; the 40 rows marked `double_code_required=1` also need both rater-1 and rater-2 labels for both outputs. Summarize each sheet:

```bash
python experiments/revision_full/error_analysis.py summarize --annotations <blinded_annotation.csv>
```

Run official TaCQ and HAWQ-V2 in isolated environments following `TACQ_INTEGRATION.md`. TaCQ importance artifacts can be checkpoint-sized and are not included in the 55/92-GiB internal estimate. Run external methods one model at a time on scratch, preserve the canonical 1,319-row samples/config/provenance, register them, and then remove their reconstructible intermediates. HAWQ's official repository is not an LLM-ready evaluator, so validate any adaptation before naming it HAWQ-V2.

Final gate:

```bash
python experiments/revision_full/external_baselines.py validate
python experiments/revision_full/analyze.py
python experiments/revision_full/readiness.py --stage resubmission
```

An external-baseline or annotation failure does not invalidate already hash-verified v4 core outputs, but it blocks the corresponding manuscript claim and the resubmission gate.

## 7. Back up only irreplaceable evidence

```bash
tar -czf revision_full_outputs.tar.gz experiments/revision_full/outputs logs
```

Download that archive to the workstation. Canonical sample JSONL files, panel records, selections, analysis, model/data manifests, state metadata, and cleanup receipts are persistent evidence. Precision banks, materialized states, TaCQ importance arrays, and temporary checkpoints are reconstructible and need not be backed up after their evidence gates pass.

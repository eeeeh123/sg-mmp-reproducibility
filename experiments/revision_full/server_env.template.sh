#!/usr/bin/env bash
# Copy to server_env.sh on the server and source it only after online model/data
# staging is complete. Edit these two paths if the checkout is moved.

export REVISION_FULL_PROJECT_DIR="/data/experiment/LQ/sg-mmp-reproducibility"
export REVISION_FULL_STORAGE_ROOT="/data/experiment/LQ"
export HF_HOME="$REVISION_FULL_STORAGE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export REVISION_FULL_EVAL_BATCH_SIZE=4
export REVISION_FULL_FORMAT_BATCH_SIZE=2
# A 32-GiB host may run two GPU workers, but only one activation-heavy
# screen/precision builder at a time. Evaluation remains concurrent.
export REVISION_FULL_MAX_CONCURRENT_RAM_BUILDERS=1
export REVISION_FULL_MIN_AVAILABLE_RAM_GIB=24
export PYTHONUNBUFFERED=1

# Uncomment only when a sufficiently large, durable-for-the-job scratch path
# exists. Persistent numerical evidence remains in the repository outputs.
# export REVISION_FULL_STATE_DIR="/scratch/$USER/sg-mmp-revision-states"

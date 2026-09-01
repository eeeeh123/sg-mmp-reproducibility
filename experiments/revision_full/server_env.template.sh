#!/usr/bin/env bash
# Copy to server_env.sh on the server, edit storage paths if needed, and source
# the same file in setup, smoke-test, and both formal-run sessions.

export HF_HOME="/data/$USER/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export REVISION_FULL_EVAL_BATCH_SIZE=4
export REVISION_FULL_FORMAT_BATCH_SIZE=2
export PYTHONUNBUFFERED=1

# Uncomment only when a sufficiently large, durable-for-the-job scratch path
# exists. Persistent numerical evidence remains in the repository outputs.
# export REVISION_FULL_STATE_DIR="/scratch/$USER/sg-mmp-revision-states"

#!/usr/bin/env bash
# Copy to server_env.sh on the server. Set REVISION_FULL_ONLINE_STAGING=1 only
# while authenticating/downloading; the default is offline formal execution.

export REVISION_FULL_PROJECT_DIR="/data/experiment/LQ/sg-mmp-reproducibility"
export REVISION_FULL_STORAGE_ROOT="/data/experiment/LQ"
export REVISION_FULL_PYTHON_ENV="/home/ubuntu/anaconda3/envs/LQ-sgmmp"
if [[ ! -x "$REVISION_FULL_PYTHON_ENV/bin/python" ]]; then
  echo "Missing experiment Python: $REVISION_FULL_PYTHON_ENV/bin/python" >&2
  return 1 2>/dev/null || exit 1
fi
case ":$PATH:" in
  *":$REVISION_FULL_PYTHON_ENV/bin:"*) ;;
  *) export PATH="$REVISION_FULL_PYTHON_ENV/bin:$PATH" ;;
esac
export PYTHONNOUSERSITE=1
export XDG_CACHE_HOME="$REVISION_FULL_STORAGE_ROOT/.cache"
export HF_HOME="$REVISION_FULL_STORAGE_ROOT/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN_PATH="$HF_HOME/token"
export HF_ASSETS_CACHE="$HF_HOME/assets"
export HF_XET_CACHE="$HF_HOME/xet"
export HF_MODULES_CACHE="$HF_HOME/modules"
export TORCH_HOME="$REVISION_FULL_STORAGE_ROOT/torch"
unset HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE
mkdir -p "$XDG_CACHE_HOME" "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" \
  "$HF_ASSETS_CACHE" "$HF_XET_CACHE" "$HF_MODULES_CACHE" "$TORCH_HOME"
if [[ "${REVISION_FULL_ONLINE_STAGING:-0}" == "1" ]]; then
  unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE
else
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi
export REVISION_FULL_EVAL_BATCH_SIZE=4
export REVISION_FULL_FORMAT_BATCH_SIZE=2
# A 32-GiB host may run two GPU workers, but only one activation-heavy
# screen/precision builder at a time. Evaluation remains concurrent.
export REVISION_FULL_MAX_CONCURRENT_RAM_BUILDERS=1
export REVISION_FULL_MIN_AVAILABLE_RAM_GIB=24
# Recheck transient post-builder RAM reclamation instead of terminating a shard.
# Zero means no automatic timeout; Ctrl+C still cancels a genuinely blocked job.
export REVISION_FULL_RAM_BUILDER_WAIT_POLL_SECONDS=30
export REVISION_FULL_RAM_BUILDER_WAIT_TIMEOUT_SECONDS=0
export PYTHONUNBUFFERED=1

# Uncomment only when a sufficiently large, durable-for-the-job scratch path
# exists. Persistent numerical evidence remains in the repository outputs.
# export REVISION_FULL_STATE_DIR="/scratch/$USER/sg-mmp-revision-states"

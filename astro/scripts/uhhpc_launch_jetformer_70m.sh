#!/bin/bash
# Launch the 70M jetformer test pretraining run on the UHHPC shared A100
# node (single node, DP=2 — pins two GPUs; this is a SHARED node, check
# nvidia-smi and override GPUS= if 0,1 are busy).
#
# Usage, from the REPO ROOT (paths in the config are cwd-relative):
#   bash astro/scripts/uhhpc_launch_jetformer_70m.sh [extra torchrun args]
# Override the config with CONFIG=<path> (e.g. the physnorm shakeout),
# the GPU pinning with GPUS=<ids>, the venv with ASTROPT3_ENV=<path>.
set -euo pipefail

CONFIG=${CONFIG:-astro/configs/nanotron/astropt3-70m-jetformer.yaml}
GPUS=${GPUS:-0,1}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

# prebuilt-wheel GPU venv (torch 2.8.0+cu128, flash-attn wheel, editable
# nanotron + astro) — recipe in docs/training.md §1
ASTROPT3_ENV=${ASTROPT3_ENV:-$REPO_ROOT/../astroPTv3_gpuenv}
source "$ASTROPT3_ENV/bin/activate"

# Configs are written for DeltaAI (/work/nvme/...); rewrite the prefix to
# this cluster's tree in a temp copy so the checked-in configs stay as-is.
LOCAL_PREFIX=${LOCAL_PREFIX:-/beegfs/general/mjsmith/foundation/astroPT_all}
LOCAL_CONFIG=$(mktemp --suffix=.yaml)
trap 'rm -f "$LOCAL_CONFIG"' EXIT
sed "s|/work/nvme/bfvh/msmith10|$LOCAL_PREFIX|g" "$CONFIG" >"$LOCAL_CONFIG"

export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
if [[ "$NPROC" -lt 2 ]]; then
    echo "WARNING: only $NPROC GPU(s) pinned; config expects dp: 2" >&2
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1 # required by nanotron's comm overlap
# ADR 0015: the corpus streams from the HF hub at train time, so the
# offline flags that suited the local parquet corpus must NOT be set
export WANDB_MODE=${WANDB_MODE:-online}

# no exec: the EXIT trap must fire to clean up the temp config
torchrun \
    --nproc-per-node="$NPROC" \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --max-restarts=0 \
    nanotron/run_train.py --config-file "$LOCAL_CONFIG" "$@"

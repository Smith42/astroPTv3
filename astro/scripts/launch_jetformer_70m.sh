#!/bin/bash
# Launch the 70M jetformer test pretraining run on the reserved DeltaAI GH200
# node (single node, DP=2 — needs both GPUs visible in this Slurm step).
#
# Usage, from the REPO ROOT (paths in the config are cwd-relative):
#   bash astro/scripts/launch_jetformer_70m.sh [extra torchrun args]
#
# Environment: the module + .venv-train overlay (see
# docs/jetformer_plan.md J4 status note); NOT the x86 $ASTROPT3_ENV recipe.
set -euo pipefail

CONFIG=astro/configs/nanotron/astropt3-70m-jetformer.yaml
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

source /etc/profile.d/modules.sh 2>/dev/null || true
module use /sw/user/modules/python
module load python/miniforge3_pytorch/2.12.0
source .venv-train/bin/activate

NPROC=$(nvidia-smi -L | wc -l)
if [[ "$NPROC" -lt 2 ]]; then
    echo "WARNING: only $NPROC GPU(s) visible; config expects dp: 2" >&2
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1   # required by nanotron's comm overlap
export HF_DATASETS_OFFLINE=1           # data is local parquet
export WANDB_MODE=${WANDB_MODE:-online}

# python -m, not the torchrun entry point: torch lives in the module's
# system site-packages, so bare `torchrun` resolves to the module python
# and loses the venv (editable nanotron/astropt3)
exec python -m torch.distributed.run \
    --nproc-per-node="$NPROC" \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --max-restarts=0 \
    nanotron/run_train.py --config-file "$CONFIG" "$@"

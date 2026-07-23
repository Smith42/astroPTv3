#!/bin/bash
# Launch a 70M MMU-streaming run on gpu5 (2x A100 80GB,
# /beegfs/general/mjsmith/gpuenv). The run name is the config basename;
# checkpoints + train.log land in ../astroPTv3_checkpoints/<run>.
#
# Usage, from the REPO ROOT:
#   bash astro/scripts/launch_mmu_gpu5.sh                # pairs+scalars run
#   bash astro/scripts/launch_mmu_gpu5.sh astropt3-70m-jetformer-mmu-nopairs
set -euo pipefail

RUN=${1:-astropt3-70m-jetformer-mmu}

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"
source /beegfs/general/mjsmith/gpuenv/bin/activate

CK=/beegfs/general/mjsmith/foundation/astroPT_all/astroPTv3_checkpoints/$RUN
mkdir -p "$CK"

export CUDA_DEVICE_MAX_CONNECTIONS=1 # required by nanotron's comm overlap
export WANDB_MODE=${WANDB_MODE:-online}

exec python -m torch.distributed.run \
	--nproc-per-node=2 \
	--rdzv-backend=c10d \
	--rdzv-endpoint=localhost:0 \
	--max-restarts=0 \
	nanotron/run_train.py \
	--config-file "astro/configs/nanotron/$RUN.yaml" \
	>>"$CK/train.log" 2>&1

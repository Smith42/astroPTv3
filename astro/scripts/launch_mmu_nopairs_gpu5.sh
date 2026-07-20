#!/bin/bash
# Launch the 70M jetformer MMU-streaming (no-pairs) 20k-step shakeout on gpu5
# (this machine: 2x A100 80GB, /beegfs/general/mjsmith/gpuenv).
# Fresh start for the HF-datasets streaming backend (ADR 0006); the
# pre-datasets 512-step attempt is archived at
# ../astroPTv3_checkpoints/astropt3-70m-jetformer-mmu-nopairs.prereader-512.
#
# Usage, from the REPO ROOT:
#   bash astro/scripts/launch_mmu_nopairs_gpu5.sh
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"
source /beegfs/general/mjsmith/gpuenv/bin/activate

CK=/beegfs/general/mjsmith/foundation/astroPT_all/astroPTv3_checkpoints/astropt3-70m-jetformer-mmu-nopairs
mkdir -p "$CK"

export CUDA_DEVICE_MAX_CONNECTIONS=1 # required by nanotron's comm overlap
export WANDB_MODE=${WANDB_MODE:-online}

exec python -m torch.distributed.run \
	--nproc-per-node=2 \
	--rdzv-backend=c10d \
	--rdzv-endpoint=localhost:0 \
	--max-restarts=0 \
	nanotron/run_train.py \
	--config-file astro/configs/nanotron/astropt3-70m-jetformer-mmu-nopairs.yaml \
	>>"$CK/train.log" 2>&1

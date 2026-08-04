#!/bin/bash
# Launch a 70M MMU-streaming run on DeltaAI (NCSA, 2x GH200 120GB, aarch64).
# Mirror of launch_mmu_gpu5.sh for this machine: it swaps gpu5's gpuenv for the
# module + .venv-train overlay, remaps the config's hardcoded gpu5 /beegfs paths
# to /work/nvme, and pushes loader workers up. gpu5 was pinned at 6/rank by its
# 1 Gbit NIC; DeltaAI's link + 32 cores have far more headroom, so this run is
# the throughput/download test for that headroom.
#
# Usage, from the REPO ROOT:
#   bash astro/scripts/launch_mmu_deltaai.sh                # pairs+scalars run
#   bash astro/scripts/launch_mmu_deltaai.sh astropt3-70m-jetformer-mmu-nopairs
#   WORKERS=8 bash astro/scripts/launch_mmu_deltaai.sh      # override per-rank loaders
set -euo pipefail

RUN=${1:-astropt3-70m-jetformer-mmu}
WORKERS=${WORKERS:-12} # per-rank loading workers; DP=2 -> 24 total on 32 cores

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"

# DeltaAI env: module ships the aarch64 flash-attn build, venv shadows datasets.
# Cray Lmod init (module isn't defined in a non-interactive shell).
source /opt/cray/pe/lmod/lmod/init/bash
module use /sw/user/modules/python
module load python/miniforge3_pytorch/2.12.0
source .venv-train/bin/activate

# DP=2 needs both GPUs; DeltaAI sometimes exposes only 1 per Slurm step.
NGPU=$(nvidia-smi -L | wc -l)
[ "$NGPU" -ge 2 ] || { echo "need 2 GPUs for DP=2, see $NGPU"; exit 1; }

CKROOT=/work/nvme/bfvh/msmith10/astroPTv3_checkpoints
CK=$CKROOT/$RUN
mkdir -p "$CK"

# Remap the config's gpu5 /beegfs paths to /work/nvme and set loader workers,
# writing a machine-local config so the canonical one stays the source of truth.
CFG=$CK/$RUN.deltaai.yaml
sed -e "s#/beegfs/general/mjsmith/foundation/astroPT_all/astroPTv3_checkpoints#$CKROOT#g" \
    -e "s#num_loading_workers: .*#num_loading_workers: $WORKERS#" \
    "astro/configs/nanotron/$RUN.yaml" > "$CFG"

export CUDA_DEVICE_MAX_CONNECTIONS=1 # required by nanotron's comm overlap
export WANDB_MODE=${WANDB_MODE:-online}

exec python -m torch.distributed.run \
	--nproc-per-node=2 \
	--rdzv-backend=c10d \
	--rdzv-endpoint=localhost:0 \
	--max-restarts=0 \
	nanotron/run_train.py \
	--config-file "$CFG" \
	>>"$CK/train.log" 2>&1

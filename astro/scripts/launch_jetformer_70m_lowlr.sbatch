#!/bin/bash
# Single-GPU DeltaAI GH200 sbatch for the 70M jetformer 10x-lower-LR shakeout.
# Wraps the proven module + .venv-train + torchrun recipe from
# launch_jetformer_70m.sh in batch headers. wandb logs automatically (the
# nanotron trainer inits from general.project/run when wandb is importable and
# WANDB_MODE=online) - make sure `wandb login` was run once in this venv.
#
# Submit from the REPO ROOT (config paths are cwd-relative):
#   sbatch astro/scripts/launch_jetformer_70m_lowlr.sbatch
#
# CONFIRM before first submit: --account and --partition below are DeltaAI
# placeholders (the reserved-node workflow used interactive allocs, so they
# aren't recorded). `sacctmgr show assoc user=$USER format=account,partition`.
#
#SBATCH --job-name=astropt3-70m-jf-lowlr
#SBATCH --account=CHANGEME_deltaai_account
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

CONFIG=astro/configs/nanotron/astropt3-70m-jetformer-lowlr.yaml
cd "$SLURM_SUBMIT_DIR"

# DeltaAI GH200 (aarch64) env: module torch + .venv-train overlay at repo root.
source /etc/profile.d/modules.sh
module use /sw/user/modules/python
module load python/miniforge3_pytorch/2.12.0
source .venv-train/bin/activate

export CUDA_DEVICE_MAX_CONNECTIONS=1   # required by nanotron's comm overlap
export HF_DATASETS_OFFLINE=1           # data is local parquet
export WANDB_MODE=${WANDB_MODE:-online}

echo "config=$CONFIG gpus=$(nvidia-smi -L | wc -l) run=astropt3-70m-jetformer-lowlr"

# python -m, not bare torchrun: torch lives in the module's system site-packages,
# so `torchrun` would resolve to the module python and lose the editable venv.
exec python -m torch.distributed.run \
    --nproc-per-node=1 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --max-restarts=0 \
    nanotron/run_train.py --config-file "$CONFIG"

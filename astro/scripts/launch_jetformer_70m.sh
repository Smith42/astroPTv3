#!/bin/bash
# Launch the 70M jetformer test pretraining run on the reserved DeltaAI GH200
# node (single node, DP=2 — needs both GPUs visible in this Slurm step).
#
# Usage, from the REPO ROOT (paths in the config are cwd-relative):
#   bash astro/scripts/launch_jetformer_70m.sh [extra torchrun args]
# Override the config with CONFIG=<path> (e.g. the physnorm shakeout).
# EVAL_GPU=<id> co-launches the eval sidecar (run_probe_sweep.py --watch
# --wandb: val loss, probe, sample panels) on that GPU — on this node DP=2
# already takes both GPUs, so normally leave it unset and run the sweep
# manually or post-hoc. EVAL_OUT= overrides the sidecar output dir.
#
# Environment: the module + .venv-train overlay (see
# docs/jetformer_plan.md J4 status note); NOT the x86 $ASTROPT3_ENV recipe.
set -euo pipefail

CONFIG=${CONFIG:-astro/configs/nanotron/astropt3-70m-jetformer.yaml}
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

# eval sidecar: polls the run's checkpoint dir on a spare GPU, fully
# decoupled from the trainer (filesystem interface: latest.txt)
SIDECAR_PID=
if [[ -n "${EVAL_GPU:-}" ]]; then
    CKPTS=$(awk '$1 == "checkpoints_path:" {print $2; exit}' "$CONFIG")
    # ADR 0006: data_root is "mmu" or "synthetic"; the val split is selected
    # inside the loader (reserved partitions), not by a sibling directory
    VAL_ROOT=$(awk '$1 == "data_root:" {print $2; exit}' "$CONFIG")
    TRAIN_STEPS=$(awk '$1 == "train_steps:" {print $2; exit}' "$CONFIG")
    EVAL_OUT=${EVAL_OUT:-$(dirname "$CKPTS")/eval/$(basename "$CKPTS")}
    mkdir -p "$EVAL_OUT"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python astro/scripts/run_probe_sweep.py \
        --checkpoints-dir "$CKPTS" --out-dir "$EVAL_OUT" --data-root "$VAL_ROOT" \
        --watch --until-step "$TRAIN_STEPS" --wandb \
        > "$EVAL_OUT/sweep.log" 2>&1 &
    SIDECAR_PID=$!
    trap 'kill "$SIDECAR_PID" 2>/dev/null || true' INT TERM
    echo "[launch] eval sidecar pid $SIDECAR_PID on GPU $EVAL_GPU -> $EVAL_OUT" >&2
fi

# python -m, not the torchrun entry point: torch lives in the module's
# system site-packages, so bare `torchrun` resolves to the module python
# and loses the venv (editable nanotron/astropt3)
TORCHRUN=(python -m torch.distributed.run
    --nproc-per-node="$NPROC"
    --rdzv-backend=c10d
    --rdzv-endpoint=localhost:0
    --max-restarts=0
    nanotron/run_train.py --config-file "$CONFIG" "$@")
if [[ -z "$SIDECAR_PID" ]]; then
    exec "${TORCHRUN[@]}"
fi
"${TORCHRUN[@]}"
# training done: --until-step lets the sidecar drain the final checkpoints
# and exit on its own
wait "$SIDECAR_PID"

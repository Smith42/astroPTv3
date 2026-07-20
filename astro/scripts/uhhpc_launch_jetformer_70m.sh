#!/bin/bash
# Launch the 70M jetformer test pretraining run on the UHHPC shared A100
# node (single node, DP=2 — pins two GPUs; this is a SHARED node, check
# nvidia-smi and override GPUS= if 0,1 are busy).
#
# Usage, from the REPO ROOT (paths in the config are cwd-relative):
#   bash astro/scripts/uhhpc_launch_jetformer_70m.sh [extra torchrun args]
# Override the config with CONFIG=<path> (e.g. the physnorm shakeout),
# the GPU pinning with GPUS=<ids>, the venv with ASTROPT3_ENV=<path>.
# EVAL_GPU=<id> co-launches the eval sidecar (run_probe_sweep.py --watch
# --wandb: val loss, probe, sample panels) on a spare GPU — pick one that
# is free AND not in GPUS (e.g. EVAL_GPU=2). EVAL_OUT= overrides the
# sidecar output dir.
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
sed "s|/work/nvme/bfvh/msmith10|$LOCAL_PREFIX|g" "$CONFIG" > "$LOCAL_CONFIG"

export CUDA_VISIBLE_DEVICES="$GPUS"
NPROC=$(awk -F, '{print NF}' <<<"$GPUS")
if [[ "$NPROC" -lt 2 ]]; then
    echo "WARNING: only $NPROC GPU(s) pinned; config expects dp: 2" >&2
fi

export CUDA_DEVICE_MAX_CONNECTIONS=1   # required by nanotron's comm overlap
export HF_DATASETS_OFFLINE=1           # data is local parquet
export WANDB_MODE=${WANDB_MODE:-online}

# eval sidecar: polls the run's checkpoint dir on a spare GPU, fully
# decoupled from the trainer (filesystem interface: latest.txt); paths come
# from the prefix-rewritten config. SAMPLES_EVERY thins the image/spectra
# panels (val-loss + probe still run on every checkpoint); default 1000
# matches this config's checkpoint_interval so imagery logs once per 1k
# steps. SAMPLES_FLOOR suppresses the Pythia schedule's early pow2<=512
# checkpoints; default 1000 -> zero imagery before step 1000, then every
# 1000 after.
SIDECAR_PID=
if [[ -n "${EVAL_GPU:-}" ]]; then
    CKPTS=$(awk '$1 == "checkpoints_path:" {print $2; exit}' "$LOCAL_CONFIG")
    # ADR 0006: data_root is "mmu" or "synthetic"; the val split is selected
    # inside the loader (reserved partitions), not by a sibling directory
    VAL_ROOT=$(awk '$1 == "data_root:" {print $2; exit}' "$LOCAL_CONFIG")
    # the eval sidecar reads the match-index from the env (ADR 0006); the
    # trainer gets the same value explicitly from the config field
    MATCH_INDEX=$(awk '$1 == "match_index:" {print $2; exit}' "$LOCAL_CONFIG")
    [[ -n "$MATCH_INDEX" && "$MATCH_INDEX" != "null" ]] && export ASTROPT3_MATCH_INDEX="$MATCH_INDEX"
    TRAIN_STEPS=$(awk '$1 == "train_steps:" {print $2; exit}' "$LOCAL_CONFIG")
    EVAL_OUT=${EVAL_OUT:-$(dirname "$CKPTS")/eval/$(basename "$CKPTS")}
    SAMPLES_EVERY=${SAMPLES_EVERY:-1000}
    SAMPLES_FLOOR=${SAMPLES_FLOOR:-1000}
    mkdir -p "$EVAL_OUT"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" python astro/scripts/run_probe_sweep.py \
        --checkpoints-dir "$CKPTS" --out-dir "$EVAL_OUT" --data-root "$VAL_ROOT" \
        --watch --until-step "$TRAIN_STEPS" \
        --samples-every "$SAMPLES_EVERY" --samples-floor "$SAMPLES_FLOOR" --wandb \
        > "$EVAL_OUT/sweep.log" 2>&1 &
    SIDECAR_PID=$!
    trap 'kill "$SIDECAR_PID" 2>/dev/null || true; rm -f "$LOCAL_CONFIG"' INT TERM
    echo "[launch] eval sidecar pid $SIDECAR_PID on GPU $EVAL_GPU -> $EVAL_OUT" >&2
fi

# no exec: the EXIT trap must fire to clean up the temp config
torchrun \
    --nproc-per-node="$NPROC" \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:0 \
    --max-restarts=0 \
    nanotron/run_train.py --config-file "$LOCAL_CONFIG" "$@"

if [[ -n "$SIDECAR_PID" ]]; then
    # training done: --until-step lets the sidecar drain the final
    # checkpoints and exit on its own
    wait "$SIDECAR_PID"
fi

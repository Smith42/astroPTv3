#!/usr/bin/env bash
# =============================================================================
# Autoregressive next-patch pretraining on SmolLM2-135M
#
# Stage 1 (default): only patch_projector + regression_head are trained.
#   --freeze_transformer true
#
# Stage 2: unfreeze everything.
#   --freeze_transformer false  --learning_rate 2e-5
#
# Smoke test on the validation split (~86 K examples):
#   bash astro/configs/patches_135m.sh
#
# Full training run: change --train_split to "train".
#
# Multi-GPU:
#   torchrun --nproc_per_node=8 astro/train_patches.py [same args]
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${PATCH_MODEL:-HuggingFaceTB/SmolLM2-135M}"
OUTPUT_DIR="./checkpoints/astro-patches-135m-stage1"

python astro/train_patches.py \
    --model_name_or_path "$MODEL" \
    --patch_size         16 \
    --huber_delta        1.0 \
    --spiral             true \
    --freeze_transformer true \
    \
    --train_split        validation \
    --eval_split         validation \
    --buffer_size        1000 \
    \
    --output_dir         "$OUTPUT_DIR" \
    --num_train_epochs   1 \
    --per_device_train_batch_size  16 \
    --per_device_eval_batch_size   16 \
    --gradient_accumulation_steps  4 \
    --learning_rate      1e-3 \
    --weight_decay       0.01 \
    --warmup_ratio       0.03 \
    --lr_scheduler_type  cosine \
    \
    --bf16               true \
    --tf32               true \
    --gradient_checkpointing true \
    \
    --dataloader_num_workers 4 \
    \
    --logging_steps      10 \
    --eval_strategy      steps \
    --eval_steps         200 \
    --save_strategy      steps \
    --save_steps         500 \
    --save_total_limit   3 \
    \
    --report_to          wandb \
    --run_name           "astro-patches-135m-stage1-$(date +%Y%m%d_%H%M%S)"

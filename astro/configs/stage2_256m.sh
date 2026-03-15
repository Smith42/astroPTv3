#!/usr/bin/env bash
# =============================================================================
# Stage 2 — Full continual pretraining on SmolVLM 256M
#
# All layers are trained (connector + vision tower + language model).
# Lower LRs than Stage 1 to avoid catastrophic forgetting.
#
# Start from a Stage 1 checkpoint: set MODEL to your stage1 output dir.
#
# Single-GPU:
#   bash astro/configs/stage2_256m.sh
#
# Multi-GPU (e.g. 8 GPUs):
#   torchrun --nproc_per_node=8 astro/train.py [same args]
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Point to the Stage 1 output dir (or the base model for a standalone run)
MODEL="${STAGE1_CKPT:-HuggingFaceTB/SmolVLM2-256M-Video-Instruct}"
OUTPUT_DIR="./checkpoints/astro-galaxies-256m-stage2"

python astro/train.py \
    --model_name_or_path "$MODEL" \
    --output_dir         "$OUTPUT_DIR" \
    \
    --astro_train_split        train \
    --astro_eval_split         validation \
    --astro_image_target_size  256 \
    --astro_buffer_size        10000 \
    \
    --vision_tower_lr    5e-6 \
    --connector_lr       1e-4 \
    --language_model_lr  2e-5 \
    \
    --num_train_epochs   1 \
    --per_device_train_batch_size  4 \
    --per_device_eval_batch_size   4 \
    --gradient_accumulation_steps  8 \
    --learning_rate      2e-5 \
    --weight_decay       0.01 \
    --warmup_ratio       0.03 \
    --lr_scheduler_type  cosine \
    \
    --bf16               true \
    --tf32               true \
    --gradient_checkpointing true \
    --disable_flash_attn2    false \
    \
    --packed                     true \
    --apply_diagonal_block_attention true \
    \
    --model_max_length   2048 \
    --dataloader_num_workers 4 \
    \
    --logging_steps      50 \
    --eval_strategy      steps \
    --eval_steps         1000 \
    --save_strategy      steps \
    --save_steps         1000 \
    --save_total_limit   5 \
    \
    --report_to          wandb \
    --run_name           "astro-galaxies-256m-stage2-$(date +%Y%m%d_%H%M%S)"

#!/usr/bin/env bash
# =============================================================================
# Stage 1 — Connector warmup on SmolVLM 256M
#
# Vision tower + language model are FROZEN.
# Only the SigLIP → LM connector (modality_projection / merger) is trained.
#
# Use the validation split (~86 K examples) for a quick smoke-test run.
# Switch to --astro_train_split train for a full training run.
#
# Single-GPU launch:
#   bash astro/configs/stage1_256m.sh
#
# Multi-GPU (e.g. 8 GPUs):
#   torchrun --nproc_per_node=8 astro/train.py [same args]
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
OUTPUT_DIR="./checkpoints/astro-galaxies-256m-stage1"

python astro/train.py \
    --model_name_or_path "$MODEL" \
    --output_dir         "$OUTPUT_DIR" \
    \
    --astro_train_split        validation \
    --astro_eval_split         validation \
    --astro_image_target_size  256 \
    --astro_buffer_size        1000 \
    \
    --vision_tower_lr    0.0 \
    --connector_lr       1e-4 \
    --language_model_lr  0.0 \
    \
    --num_train_epochs   1 \
    --per_device_train_batch_size  4 \
    --per_device_eval_batch_size   4 \
    --gradient_accumulation_steps  4 \
    --learning_rate      1e-4 \
    --weight_decay       0.0 \
    --warmup_ratio       0.03 \
    --lr_scheduler_type  cosine \
    \
    --bf16               true \
    --tf32               true \
    --gradient_checkpointing true \
    --disable_flash_attn2    false \
    \
    --model_max_length   2048 \
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
    --run_name           "astro-galaxies-256m-stage1-$(date +%Y%m%d_%H%M%S)"

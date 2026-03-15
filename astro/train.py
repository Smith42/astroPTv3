"""
astro/train.py
==============
Continual pretraining of SmolVLM on Smith42/galaxies.

Reuses the existing SmolVLM2 model loading / freezing / trainer infrastructure
from vision/smolvlm2/smolvlm/train/, replacing only the data module with our
HuggingFace-streaming GalaxySmolVLMDataset.

Stage 1 — connector warmup (backbone + vision tower frozen):
    connector_lr=1e-4, language_model_lr=0, vision_tower_lr=0

Stage 2 — full continual pretraining (all layers trained):
    connector_lr=1e-4, language_model_lr=2e-5, vision_tower_lr=5e-6

See astro/configs/ for ready-made launch scripts.
"""

from __future__ import annotations

import os
import sys
import logging
import pathlib
from functools import partial

# Make the smolvlm2 package importable from the repo root
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "vision" / "smolvlm2"))

import torch
import transformers
from transformers import AutoProcessor, HfArgumentParser, set_seed

from dataclasses import dataclass, field
from smolvlm.train.args import DataArguments as _BaseDataArguments, ModelArguments, TrainingArguments
from smolvlm.train.train import (
    prepare_model,
    set_trainable_params,
    enable_gradient_checkpointing,
    auto_resume_or_start,
    trainer_save_model_safe,
)
from smolvlm.train.smolvlm_trainer import SmolVLMTrainer

from astro.data import GalaxySmolVLMDataset, galaxy_collate_fn

logger = logging.getLogger(__name__)


@dataclass
class AstroDataArguments(_BaseDataArguments):
    """Extends DataArguments with galaxy-specific fields."""
    astro_train_split: str = field(
        default="train",
        metadata={"help": "HF split for training: 'train' or 'validation'."},
    )
    astro_eval_split: str = field(
        default="validation",
        metadata={"help": "HF split for evaluation. Empty string = no eval."},
    )
    astro_image_target_size: int = field(
        default=256,
        metadata={"help": "Longest-edge size passed to the SigLIP image processor."},
    )
    astro_buffer_size: int = field(
        default=1000,
        metadata={"help": "Streaming shuffle buffer size (0 = deterministic)."},
    )
    astro_max_samples: int = field(
        default=0,
        metadata={"help": "Cap total training examples (0 = no cap)."},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_module(processor, data_args: AstroDataArguments, training_args: TrainingArguments):
    """Build train + eval datasets and a collator from the galaxy pipeline."""
    train_split = data_args.astro_train_split
    eval_split  = data_args.astro_eval_split or None
    img_size    = data_args.astro_image_target_size
    buf_size    = data_args.astro_buffer_size
    max_samples = data_args.astro_max_samples or None

    train_dataset = GalaxySmolVLMDataset(
        processor=processor,
        split=train_split,
        use_crop=True,
        max_samples=max_samples,
        buffer_size=buf_size,
        image_target_size=img_size,
    )

    eval_dataset = None
    if eval_split:
        eval_dataset = GalaxySmolVLMDataset(
            processor=processor,
            split=eval_split,
            use_crop=True,
            max_samples=min(max_samples or 2000, 2000),   # cap eval at 2 K
            buffer_size=0,          # deterministic
            image_target_size=img_size,
        )

    collator = partial(
        galaxy_collate_fn,
        pad_token_id=processor.tokenizer.pad_token_id or 0,
    )

    return {
        "train_dataset":  train_dataset,
        "eval_dataset":   eval_dataset,
        "data_collator":  collator,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = HfArgumentParser((ModelArguments, AstroDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # Derive tune_* flags from LR values (same logic as smolvlm2/train.py)
    training_args.tune_language_model = training_args.language_model_lr > 1e-9
    training_args.tune_mm_connector   = training_args.connector_lr       > 1e-9
    training_args.tune_vision_tower   = training_args.vision_tower_lr    > 1e-9

    # 1. Load model
    logger.info("Loading model: %s", model_args.model_name_or_path)
    model = prepare_model(model_args, training_args)

    # 2. Freeze / unfreeze
    set_trainable_params(model, training_args)

    # 3. Gradient checkpointing
    if training_args.gradient_checkpointing:
        enable_gradient_checkpointing(model, training_args)

    # 4. Processor
    logger.info("Loading processor: %s", model_args.model_name_or_path)
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        trust_remote_code=model_args.trust_remote_code,
    )

    # 5. Data
    logger.info("Building galaxy datasets…")
    data_module = _make_data_module(processor, data_args, training_args)
    logger.info(
        "  train ≈ %s examples | eval ≈ %s examples",
        data_module["train_dataset"].approx_len(),
        data_module["eval_dataset"].approx_len() if data_module["eval_dataset"] else 0,
    )

    # 6. Trainer
    trainer = SmolVLMTrainer(
        model=model,
        args=training_args,
        **data_module,
    )

    # 7. Train
    resume = auto_resume_or_start(training_args)
    if resume:
        logger.info("Resuming from checkpoint in %s", training_args.output_dir)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # 8. Save
    model.config.use_cache = True
    trainer.save_state()
    trainer_save_model_safe(trainer)
    logger.info("Done. Model saved to %s", training_args.output_dir)


if __name__ == "__main__":
    train()

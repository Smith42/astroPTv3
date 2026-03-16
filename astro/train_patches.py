"""
astro/train_patches.py
======================
Autoregressive next-patch pretraining for SmolLM AstroPT (Option A).

Bypasses SigLIP entirely.  Raw 16×16 pixel patches are projected into the
SmolLM embedding space, the causal transformer predicts the next patch, and
Huber loss is computed on the continuous patch values.

This is the self-supervised training objective from the original AstroPT,
extended to use the SmolLM backbone instead of GPT-2.

Stage 1 (connector warmup):
    Only patch_projector and regression_head are trained.
    Set --freeze_transformer true (default).

Stage 2 (full continual pretraining):
    All parameters trained at lower LRs.
    Set --freeze_transformer false.

Launch:
    bash astro/configs/patches_135m.sh
"""

from __future__ import annotations

import os
import sys
import logging
import pathlib
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "vision" / "smolvlm2"))

import torch
import transformers
from transformers import HfArgumentParser, TrainingArguments, set_seed

from astro.data.patch_dataset import GalaxyPatchDataset
from astro.model.patch_model import build_patch_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class PatchModelArguments:
    model_name_or_path: str = field(
        default="HuggingFaceTB/SmolLM2-135M",
        metadata={"help": "SmolLM backbone to load (text-only causal LM)."},
    )
    patch_size: int = field(
        default=16,
        metadata={"help": "Pixel side length of each square patch."},
    )
    huber_delta: float = field(
        default=1.0,
        metadata={"help": "Delta for Huber (smooth L1) loss."},
    )
    freeze_transformer: bool = field(
        default=True,
        metadata={"help": "Stage 1: freeze transformer, train projector+head only."},
    )
    spiral: bool = field(
        default=True,
        metadata={"help": "Apply spiral patch ordering (matches original AstroPT)."},
    )


@dataclass
class PatchDataArguments:
    train_split: str = field(
        default="train",
        metadata={"help": "'train', 'validation', or 'test'."},
    )
    eval_split: str = field(
        default="validation",
        metadata={"help": "Split for evaluation. Empty = no eval."},
    )
    buffer_size: int = field(
        default=1000,
        metadata={"help": "Streaming shuffle buffer size (0 = deterministic)."},
    )
    max_samples: int = field(
        default=0,
        metadata={"help": "Cap training examples (0 = no cap)."},
    )


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------

def patch_collate_fn(batch):
    """
    All examples have identical shapes (1024, 768) so no padding needed.
    """
    return {
        "patches": torch.stack([b["patches"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Custom Trainer to log patch loss clearly
# ---------------------------------------------------------------------------

class PatchTrainer(transformers.Trainer):
    """Thin subclass — Trainer handles everything, we just name the loss."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = HfArgumentParser((PatchModelArguments, PatchDataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    set_seed(training_args.seed)

    # 1. Build model
    logger.info("Building AstroPatchModel from %s", model_args.model_name_or_path)
    compute_dtype = (
        torch.bfloat16 if training_args.bf16 else
        torch.float16  if training_args.fp16 else
        torch.float32
    )
    model, _tokenizer = build_patch_model(
        model_name=model_args.model_name_or_path,
        patch_dim=model_args.patch_size ** 2 * 3,
        huber_delta=model_args.huber_delta,
        torch_dtype=compute_dtype,
        freeze_transformer=model_args.freeze_transformer,
    )

    if training_args.gradient_checkpointing:
        model.transformer.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # 2. Datasets
    max_samples = data_args.max_samples or None

    train_ds = GalaxyPatchDataset(
        split=data_args.train_split,
        patch_size=model_args.patch_size,
        spiral=model_args.spiral,
        max_samples=max_samples,
        buffer_size=data_args.buffer_size,
    )

    eval_ds = None
    if data_args.eval_split:
        eval_ds = GalaxyPatchDataset(
            split=data_args.eval_split,
            patch_size=model_args.patch_size,
            spiral=model_args.spiral,
            max_samples=min(max_samples or 2000, 2000),
            buffer_size=0,
        )

    logger.info(
        "  train ≈ %s examples | eval ≈ %s examples",
        train_ds.approx_len(),
        eval_ds.approx_len() if eval_ds else 0,
    )

    # 3. Train
    trainer = PatchTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=patch_collate_fn,
    )

    # Resume from checkpoint if one exists
    ckpts = list(pathlib.Path(training_args.output_dir).glob("checkpoint-*"))
    if ckpts:
        logger.info("Resuming from checkpoint in %s", training_args.output_dir)
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # 4. Save
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)
    logger.info("Done. Model saved to %s", training_args.output_dir)


if __name__ == "__main__":
    train()

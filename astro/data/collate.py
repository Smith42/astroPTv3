"""
Collator for GalaxySmolVLMDataset batches.

Pads input_ids / attention_mask / labels to the longest sequence in the batch.
Stacks pixel_values when present (fixed shape per example when
do_image_splitting=False).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100


def galaxy_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int = 0,
) -> Dict[str, torch.Tensor]:
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [b["attention_mask"] for b in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )

    out = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

    if "pixel_values" in batch[0]:
        try:
            out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
        except RuntimeError:
            # Shape mismatch shouldn't happen with do_image_splitting=False,
            # but fall back to a list so the model can handle it.
            out["pixel_values"] = [b["pixel_values"] for b in batch]

    return out

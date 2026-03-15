"""
Collator for GalaxySmolVLMDataset batches.

Handles both unpacked batches (standard padding) and packed batches
(sequences already concatenated by PackingIterableDataset, with subseq_ids
replacing attention_mask for the flash_attention_2 varlen path).
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
    """
    Works for both packed and unpacked batches.

    Unpacked: pads input_ids / attention_mask / labels to longest sequence.
    Packed  : sequences are already max-length; subseq_ids is present and
              attention_mask contains integer sub-sequence IDs (not 0/1).
    """
    input_ids = pad_sequence(
        [b["input_ids"] for b in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = pad_sequence(
        [b["labels"] for b in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX,
    )

    # attention_mask: use subseq_ids if present (packed path), else 0/1
    if "subseq_ids" in batch[0]:
        attention_mask = pad_sequence(
            [b["subseq_ids"] for b in batch],
            batch_first=True,
            padding_value=0,  # 0 = padding position
        )
    else:
        attention_mask = pad_sequence(
            [b["attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0,
        )

    out = {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "labels":         labels,
    }

    if "pixel_values" in batch[0]:
        pv = [b["pixel_values"] for b in batch]
        try:
            # Unpacked: each pv is (C, H, W) → stack to (B, C, H, W)
            # Packed: each pv is (N_images, C, H, W) → cat to (total_images, C, H, W)
            if pv[0].dim() == 3:
                out["pixel_values"] = torch.stack(pv)
            else:
                out["pixel_values"] = torch.cat(pv, dim=0)
        except RuntimeError:
            out["pixel_values"] = pv  # fallback: list

    return out

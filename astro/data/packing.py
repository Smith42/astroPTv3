"""
PackingIterableDataset
======================
Greedy sequence-packing wrapper for any IterableDataset that yields
{input_ids, attention_mask, labels, pixel_values?}.

Packs multiple short examples into a single sequence up to ``max_length``
tokens, emitting ``subseq_ids`` so the diagonal block attention
(apply_varlen_patch from smolvlm2) correctly prevents cross-example
attention via flash_attention_2 varlen.

subseq_ids convention (same as PackedConcatDataset):
    1-indexed integers, one per token.  Padding positions get 0.
    e.g. three examples packed together:
    [1,1,1,1, 2,2,2,2,2, 3,3,3]  (no padding needed here)

pixel_values handling:
    Each example contributes one image tensor of shape (C, H, W).
    Packed pixel_values are stacked → (N_images, C, H, W).
    The model iterates over <image> tokens in input_ids in order to
    assign the correct image embedding to each token, so order matters.
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import torch
from torch.utils.data import IterableDataset


class PackingIterableDataset(IterableDataset):
    """
    Wraps an iterable dataset and greedily packs examples into
    sequences of up to ``max_length`` tokens.

    Parameters
    ----------
    dataset : IterableDataset
        Source dataset yielding {input_ids, attention_mask, labels, pixel_values?}.
    max_length : int
        Maximum token count per packed sequence.
    """

    def __init__(self, dataset: IterableDataset, max_length: int):
        super().__init__()
        self.dataset = dataset
        self.max_length = max_length

    def approx_len(self) -> int:
        base = getattr(self.dataset, "approx_len", lambda: 0)()
        # Each packed sequence holds ~max_length / avg_seq_len examples.
        # We can't know avg_seq_len without scanning, so return a lower bound.
        return base  # conservative; real packed count will be smaller

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buffer: List[Dict] = []
        token_count = 0

        for item in self.dataset:
            seq_len = item["input_ids"].size(0)

            # Single example already exceeds max_length — yield alone, truncated
            if seq_len > self.max_length:
                if buffer:
                    yield _merge(buffer)
                    buffer, token_count = [], 0
                yield _merge([_truncate(item, self.max_length)])
                continue

            # Would overflow — flush current buffer first
            if token_count + seq_len > self.max_length and buffer:
                yield _merge(buffer)
                buffer, token_count = [], 0

            buffer.append(item)
            token_count += seq_len

        # Flush any remaining examples
        if buffer:
            yield _merge(buffer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(item: Dict, max_length: int) -> Dict:
    """Hard-truncate a single example to max_length tokens."""
    out = {
        "input_ids":      item["input_ids"][:max_length],
        "attention_mask": item["attention_mask"][:max_length],
        "labels":         item["labels"][:max_length],
    }
    if "pixel_values" in item:
        out["pixel_values"] = item["pixel_values"]
    return out


def _merge(items: List[Dict]) -> Dict:
    """
    Concatenate a list of examples into one packed sequence with subseq_ids.
    """
    all_input_ids      = []
    all_attention_mask = []
    all_labels         = []
    all_subseq_ids     = []
    all_pixel_values   = []

    for seq_id, item in enumerate(items, start=1):
        n = item["input_ids"].size(0)
        all_input_ids.append(item["input_ids"])
        all_attention_mask.append(item["attention_mask"])
        all_labels.append(item["labels"])
        all_subseq_ids.append(
            torch.full((n,), fill_value=seq_id, dtype=torch.long)
        )
        if "pixel_values" in item:
            all_pixel_values.append(item["pixel_values"])

    out: Dict[str, torch.Tensor] = {
        "input_ids":      torch.cat(all_input_ids),
        "attention_mask": torch.cat(all_subseq_ids),  # varlen patch reads this
        "labels":         torch.cat(all_labels),
        "subseq_ids":     torch.cat(all_subseq_ids),
    }
    # NOTE: attention_mask is set to subseq_ids here so that
    # apply_varlen_patch (which reads attention_mask as integer sub-seq IDs)
    # gets the right input.  The standard 0/1 mask is redundant when using
    # flash_attention_2 varlen — all packed tokens are "real" tokens.

    if all_pixel_values:
        # Stack along a new leading dim: (N_images, C, H, W)
        try:
            out["pixel_values"] = torch.stack(all_pixel_values)
        except RuntimeError:
            # Shapes don't match (shouldn't happen with fixed-size crops)
            out["pixel_values"] = all_pixel_values[0].unsqueeze(0)

    return out

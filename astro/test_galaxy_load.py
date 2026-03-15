#!/usr/bin/env python3
"""
astro/test_galaxy_load.py
=========================
Smoke test for the Smith42/galaxies data pipeline.

Loads N examples from the validation split, runs them through the SmolVLM
processor, and prints shapes + a decoded sample to confirm everything works.

Usage
-----
    # From the repo root (astroPT3/):
    python astro/test_galaxy_load.py

    # Load more examples or use a different split:
    python astro/test_galaxy_load.py --split train --n 16 --model HuggingFaceTB/SmolVLM2-256M-Video-Instruct
"""

import argparse
import sys
import os

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from astro.data.galaxy_dataset import GalaxyIterableDataset
from astro.data.smolvlm_adapter import GalaxySmolVLMDataset, IGNORE_INDEX, build_metadata_text


# ---------------------------------------------------------------------------
# Collate: pad a list of variable-length tensors into a batch
# ---------------------------------------------------------------------------

def collate_fn(batch):
    """Simple left-pad collator that handles variable sequence lengths."""
    input_ids = torch.nn.utils.rnn.pad_sequence(
        [b["input_ids"] for b in batch], batch_first=True, padding_value=0
    )
    attention_mask = torch.nn.utils.rnn.pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        [b["labels"] for b in batch], batch_first=True, padding_value=IGNORE_INDEX
    )
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if "pixel_values" in batch[0]:
        # pixel_values may have different shapes if image splitting is on;
        # here do_image_splitting=False so shapes match.
        try:
            out["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
        except RuntimeError as e:
            print(f"[collate] pixel_values stack failed: {e} — skipping")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Galaxy data pipeline smoke test")
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
        help="HuggingFace model ID for the SmolVLM processor",
    )
    parser.add_argument(
        "--split", default="validation", choices=["train", "validation", "test"]
    )
    parser.add_argument("--n", type=int, default=8, help="Number of examples to load")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Test raw GalaxyIterableDataset (no processor) instead",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Raw dataset test
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 1: GalaxyIterableDataset (raw images)")
    print(f"{'='*60}")
    raw_ds = GalaxyIterableDataset(
        split=args.split,
        use_crop=True,
        max_samples=args.n,
        buffer_size=0,  # deterministic for test
    )
    print(f"  Approximate split size: {raw_ds.approx_len():,}")

    raw_items = []
    for item in raw_ds:
        raw_items.append(item)
        if len(raw_items) == 1:
            meta_keys = [k for k in item if k not in ("image", "dr8_id", "galaxy_size")]
            print(f"  Image size      : {item['image'].size}  mode={item['image'].mode}")
            print(f"  dr8_id          : {item['dr8_id']}")
            print(f"  galaxy_size     : {item['galaxy_size']}")
            print(f"  Metadata fields : {len(meta_keys)}  ({', '.join(meta_keys[:6])}{'…' if len(meta_keys) > 6 else ''})")
            for k in meta_keys[:4]:
                print(f"    {k:45s}: {item[k]:.4f}")

    print(f"  Loaded {len(raw_items)} raw items ✓")

    if args.raw:
        # Show what the metadata text target would look like
        print(f"\n  Example metadata target text:")
        print("  " + "\n  ".join(build_metadata_text(raw_items[0]).splitlines()))
        print("\nDone (--raw mode, skipping processor test).")
        return

    # ------------------------------------------------------------------
    # 2. Processor test
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 2: Loading processor  ({args.model})")
    print(f"{'='*60}")
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"  Tokenizer vocab size : {processor.tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # 3. SmolVLM adapter test
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("Step 3: GalaxySmolVLMDataset (tokenised)")
    print(f"{'='*60}")
    smol_ds = GalaxySmolVLMDataset(
        processor=processor,
        split=args.split,
        use_crop=True,
        max_samples=args.n,
        buffer_size=0,
        image_target_size=256,
    )

    smol_items = []
    for item in smol_ds:
        smol_items.append(item)
        if len(smol_items) == 1:
            print(f"  Keys              : {list(item.keys())}")
            print(f"  input_ids shape   : {item['input_ids'].shape}")
            print(f"  attention_mask    : {item['attention_mask'].shape}")
            print(f"  labels shape      : {item['labels'].shape}")
            if "pixel_values" in item:
                print(f"  pixel_values      : {item['pixel_values'].shape}")
            # Decode the visible (non-masked) label tokens
            non_masked = item["labels"][item["labels"] != IGNORE_INDEX]
            decoded = processor.tokenizer.decode(non_masked, skip_special_tokens=True)
            print(f"  Label text        : {decoded!r}")

    print(f"  Processed {len(smol_items)} items ✓")

    # ------------------------------------------------------------------
    # 4. DataLoader test
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Step 4: DataLoader  (batch_size={args.batch_size})")
    print(f"{'='*60}")
    loader = DataLoader(
        smol_ds,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # keep simple for smoke test
    )

    for batch_idx, batch in enumerate(loader):
        print(f"  Batch {batch_idx}:")
        for k, v in batch.items():
            print(f"    {k:20s}: {v.shape}  dtype={v.dtype}")
        # Count non-masked label tokens
        active = (batch["labels"] != IGNORE_INDEX).sum().item()
        total = batch["labels"].numel()
        print(f"    active label tokens: {active}/{total}  ({100*active/total:.1f}%)")
        break

    print(f"\nAll checks passed ✓")
    print(
        "\nNext step: run Stage 1 connector warmup training.\n"
        "  See astro/PLAN.md § Phase 2 for details."
    )


if __name__ == "__main__":
    main()

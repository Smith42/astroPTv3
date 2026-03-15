"""
GalaxySmolVLMDataset
====================
Wraps GalaxyIterableDataset and converts each galaxy into the conversation
format expected by the SmolVLM2 training pipeline, then tokenises via the
HuggingFace processor.

Training objective
------------------
Input  : <image> token (galaxy cutout, default 256×256)
Target : structured text summarising the galaxy's measured physical properties,
         e.g.

    Galaxy 613985_1104.
    Redshift: z=0.1814 (spectroscopic), z_photo=0.1810.
    Stellar mass: log M/M_sun = 10.23.
    SFR: 1.234 M_sun/yr (total, median).
    Photometry (DESI grz): g=18.20, r=17.80, z=17.40.
    Sérsic profile: n=2.31, b/a=0.72.
    Morphology (GZ vote fractions): smooth=0.83, featured=0.15,
        spiral arms=0.04, merging=0.01.
    Petrosian half-light radius: 3.21 arcsec.

Loss is computed on the assistant response only; user/system tokens are masked.
Fields with NaN / missing values are omitted from the target text.

Usage
-----
    from transformers import AutoProcessor
    from astro.data.smolvlm_adapter import GalaxySmolVLMDataset

    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
    ds = GalaxySmolVLMDataset(processor=processor, split="validation", max_samples=64)
    item = next(iter(ds))
    # item.keys() → {input_ids, attention_mask, labels, pixel_values}
"""

from __future__ import annotations

from typing import Dict, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

from .galaxy_dataset import GalaxyIterableDataset

IGNORE_INDEX = -100

_SYSTEM = (
    "You are an astronomical assistant analysing galaxy images from the "
    "DESI Legacy Survey DR8."
)


# ------------------------------------------------------------------
# Metadata → text formatter
# ------------------------------------------------------------------

def _fmt(val: Optional[float], fmt: str = ".3f") -> Optional[str]:
    """Return formatted string or None if val is missing."""
    if val is None:
        return None
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return None


def build_metadata_text(item: dict) -> str:
    """
    Convert a galaxy item dict into the assistant's target text.
    Missing fields are omitted rather than filled with placeholders.
    """
    lines = [f"Galaxy {item['dr8_id']}."]

    # --- Redshift ---
    spec_z   = _fmt(item.get("redshift"),  ".4f")
    photo_z  = _fmt(item.get("photo_z"),   ".4f")
    if spec_z and photo_z:
        lines.append(f"Redshift: z={spec_z} (spectroscopic), z_photo={photo_z}.")
    elif spec_z:
        lines.append(f"Redshift: z={spec_z} (spectroscopic).")
    elif photo_z:
        lines.append(f"Redshift: z_photo={photo_z} (photometric).")

    # --- Stellar mass ---
    mass = _fmt(item.get("elpetro_mass_log"), ".2f")
    if mass:
        lines.append(f"Stellar mass: log M/M_sun = {mass}.")

    # --- SFR ---
    sfr = _fmt(item.get("total_sfr_median"), ".3f")
    if sfr:
        lines.append(f"SFR: {sfr} M_sun/yr (total, median).")

    # --- Photometry ---
    g = _fmt(item.get("mag_g_desi"), ".2f")
    r = _fmt(item.get("mag_r_desi"), ".2f")
    z = _fmt(item.get("mag_z_desi"), ".2f")
    phot_parts = [f"g={g}" if g else None,
                  f"r={r}" if r else None,
                  f"z={z}" if z else None]
    phot_parts = [p for p in phot_parts if p]
    if phot_parts:
        lines.append(f"Photometry (DESI grz): {', '.join(phot_parts)}.")

    # --- Sérsic profile ---
    sn  = _fmt(item.get("sersic_n"),  ".2f")
    sba = _fmt(item.get("sersic_ba"), ".2f")
    if sn and sba:
        lines.append(f"Sérsic profile: n={sn}, b/a={sba}.")
    elif sn:
        lines.append(f"Sérsic index: n={sn}.")

    # --- Galaxy Zoo morphology ---
    smooth   = _fmt(item.get("smooth-or-featured_smooth_fraction"),           ".2f")
    featured = _fmt(item.get("smooth-or-featured_featured-or-disk_fraction"), ".2f")
    spiral   = _fmt(item.get("has-spiral-arms_yes_fraction"),                 ".2f")
    merger   = _fmt(item.get("merging_merger_fraction"),                      ".2f")
    morph_parts = []
    if smooth:   morph_parts.append(f"smooth={smooth}")
    if featured: morph_parts.append(f"featured={featured}")
    if spiral:   morph_parts.append(f"spiral arms={spiral}")
    if merger:   morph_parts.append(f"merging={merger}")
    if morph_parts:
        lines.append(f"Morphology (GZ vote fractions): {', '.join(morph_parts)}.")

    # --- Angular size ---
    r50 = _fmt(item.get("est_petro_th50"), ".2f")
    if r50:
        lines.append(f"Petrosian half-light radius: {r50} arcsec.")

    return "\n".join(lines)


# ------------------------------------------------------------------
# Label masking
# ------------------------------------------------------------------

def _mask_non_assistant_tokens(
    input_ids: torch.Tensor,
    tokenizer,
) -> torch.Tensor:
    """
    Mask all tokens before the first assistant response header so loss
    is only computed on what the model needs to predict.

    The SmolVLM chat template marks the assistant turn with
    ``<|im_start|>assistant``.
    """
    labels = input_ids.clone()

    header_ids = tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
    header_len = len(header_ids)
    header_t = torch.tensor(header_ids, dtype=input_ids.dtype)

    n = input_ids.size(0)
    start_idx = None
    for i in range(n - header_len + 1):
        if torch.equal(input_ids[i : i + header_len], header_t):
            start_idx = i
            break

    if start_idx is not None:
        labels[: start_idx + header_len] = IGNORE_INDEX
    else:
        labels[:] = IGNORE_INDEX  # safe fallback

    return labels


# ------------------------------------------------------------------
# Main dataset
# ------------------------------------------------------------------

class GalaxySmolVLMDataset(IterableDataset):
    """
    Iterable dataset yielding tokenised SmolVLM training examples for galaxies.

    Parameters
    ----------
    processor : transformers.ProcessorMixin
        SmolVLM processor (tokeniser + SigLIP image processor).
    split : str
        "train", "validation", or "test".
    use_crop : bool
        True (default) = 256×256 crop; False = full 512×512 image.
    max_samples : int | None
        Cap on examples; None = full split.
    buffer_size : int
        Streaming shuffle buffer.
    image_target_size : int
        ``longest_edge`` passed to the SigLIP image processor.
    """

    def __init__(
        self,
        processor,
        split: str = "validation",
        use_crop: bool = True,
        max_samples: Optional[int] = None,
        buffer_size: int = 1000,
        image_target_size: int = 256,
        max_length: Optional[int] = None,
    ):
        super().__init__()
        self.processor = processor
        self.image_target_size = image_target_size
        self.max_length = max_length or getattr(
            processor.tokenizer, "model_max_length", None
        )
        self._galaxy_ds = GalaxyIterableDataset(
            split=split,
            use_crop=use_crop,
            max_samples=max_samples,
            buffer_size=buffer_size,
        )

    def approx_len(self) -> int:
        return self._galaxy_ds.approx_len()

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        self.processor.image_processor.size = {"longest_edge": self.image_target_size}
        self.processor.image_processor.do_resize = True
        self.processor.image_processor.do_image_splitting = False

        for item in self._galaxy_ds:
            try:
                yield self._process_item(item)
            except Exception as exc:
                print(
                    f"[GalaxySmolVLMDataset] skipping {item.get('dr8_id')}: {exc}"
                )
                continue

    def _process_item(self, item: dict) -> Dict[str, torch.Tensor]:
        image = item["image"]
        assistant_text = build_metadata_text(item)

        conversation = [
            {"role": "system",    "content": [{"type": "text",  "text": _SYSTEM}]},
            {"role": "user",      "content": [{"type": "image"}]},
            {"role": "assistant", "content": [{"type": "text",  "text": assistant_text}]},
        ]

        text_input = self.processor.apply_chat_template(
            conversation, add_generation_prompt=False
        )
        encoded = self.processor(
            text=text_input,
            images=[image],
            return_tensors="pt",
            padding=False,
        )

        input_ids      = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]

        if self.max_length and input_ids.size(0) > self.max_length:
            raise ValueError(
                f"Sequence length {input_ids.size(0)} exceeds max_length {self.max_length} "
                f"for galaxy {item.get('dr8_id')} — skipping."
            )

        labels         = _mask_non_assistant_tokens(input_ids, self.processor.tokenizer)

        out: Dict[str, torch.Tensor] = {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }
        if "pixel_values" in encoded:
            out["pixel_values"] = encoded["pixel_values"][0]

        return out

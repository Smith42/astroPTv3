"""
GalaxyPatchDataset
==================
Streams Smith42/galaxies@v2.0 and returns sequences of raw image patches
for autoregressive next-patch prediction (Option A training objective).

Patch pipeline
--------------
1. Load 512×512 RGB image (image column — full resolution, matches original AstroPT).
2. Divide into 16×16 pixel patches → 32×32 = 1024 patches, each (768,) = 16×16×3.
3. Normalise pixel values to [0, 1] by dividing by 255 (matches original AstroPT).
4. Apply spiral ordering (matches original AstroPT — improves spatial locality).
5. Return:
     patches  shape (1024, 768)  — full patch sequence

The autoregressive split (patches_in / patches_target) and modality boundary
tokens (<galaxy_start> etc.) are handled inside AstroPatchModel.forward so
the dataset stays modality-agnostic.

Usage
-----
    from astro.data.patch_dataset import GalaxyPatchDataset

    ds = GalaxyPatchDataset(split="validation")
    item = next(iter(ds))
    item["patches"].shape      # (1024, 768)
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image

_datasets = None


def _get_datasets():
    global _datasets
    if _datasets is None:
        import datasets as _ds
        _datasets = _ds
    return _datasets


HF_DATASET_ID = "Smith42/galaxies"
HF_REVISION   = "v2.0"

PATCH_SIZE  = 16   # pixels per patch side
IMAGE_SIZE  = 512  # full-resolution image is 512×512 (matches original AstroPT)
N_PATCHES   = (IMAGE_SIZE // PATCH_SIZE) ** 2   # 1024
PATCH_DIM   = PATCH_SIZE * PATCH_SIZE * 3        # 768  (16×16×3)


# ------------------------------------------------------------------
# Spiral ordering — ported directly from Smith42/astroPT
# ------------------------------------------------------------------

def _spiral_indices(n: int) -> np.ndarray:
    """Return a flat index array that maps raster positions → spiral order."""
    a = np.arange(n * n)
    b = a.reshape((n, n))
    m = None
    for i in range(n, 0, -2):
        m = np.r_[m, b[0, :], b[1:, -1], b[-1, :-1][::-1], b[1:-1, 0][::-1]]
        b = b[1:-1, 1:-1]
    a[list(m[1:])] = list(a)
    a = abs(a - n * n + 1)
    return a.reshape((n, n)).flatten()


# Cache the indices — same for all 512×512 images with patch_size=16
_SPIRAL_IDX = _spiral_indices(IMAGE_SIZE // PATCH_SIZE)   # length 1024


def spiralise(patches: torch.Tensor) -> torch.Tensor:
    """Reorder (N_patches, dim) from raster to spiral order."""
    return patches[_SPIRAL_IDX]


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class GalaxyPatchDataset(IterableDataset):
    """
    Streams Smith42/galaxies and yields patch sequences for next-patch
    autoregressive pretraining.

    Parameters
    ----------
    split : str
        "train", "validation", or "test".
    patch_size : int
        Side length of each square patch in pixels (default 16).
    spiral : bool
        Apply spiral ordering before returning patches (default True,
        matching the original AstroPT).
    max_samples : int | None
        Cap on examples; None = full split.
    hf_token : str | None
        HuggingFace auth token (dataset is public).
    buffer_size : int
        Streaming shuffle buffer (0 = deterministic).
    """

    def __init__(
        self,
        split: str = "validation",
        patch_size: int = PATCH_SIZE,
        spiral: bool = True,
        max_samples: Optional[int] = None,
        hf_token: Optional[str] = None,
        buffer_size: int = 1000,
    ):
        super().__init__()
        self.split       = split
        self.patch_size  = patch_size
        self.spiral      = spiral
        self.max_samples = max_samples
        self.hf_token    = hf_token
        self.buffer_size = buffer_size

        p = patch_size
        self._patch_dim   = p * p * 3
        self._n_side      = IMAGE_SIZE // p        # patches per side
        self._spiral_idx  = _spiral_indices(self._n_side)

    def approx_len(self) -> int:
        approx = {
            "train":      8_474_566,
            "validation":    86_499,
            "test":          86_471,
        }
        n = approx.get(self.split, 0)
        if self.max_samples is not None:
            n = min(n, self.max_samples)
        return n

    def __iter__(self) -> Iterator[dict]:
        ds = _get_datasets()

        kwargs: dict = {
            "revision": HF_REVISION,
            "streaming": True,
            "trust_remote_code": False,
        }
        if self.hf_token:
            kwargs["token"] = self.hf_token

        hf_ds = ds.load_dataset(HF_DATASET_ID, split=self.split, **kwargs)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            hf_ds = hf_ds.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
                contiguous=True,
            )

        if self.buffer_size > 0:
            hf_ds = hf_ds.shuffle(buffer_size=self.buffer_size, seed=42)

        count = 0
        for row in hf_ds:
            if self.max_samples is not None and count >= self.max_samples:
                break

            try:
                item = self._process_row(row)
            except Exception as exc:
                print(f"[GalaxyPatchDataset] skipping {row.get('dr8_id')}: {exc}")
                continue

            yield item
            count += 1

    def _process_row(self, row: dict) -> dict:
        image = row["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB").resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR
        )

        # (H, W, C) → (C, H, W), float32, normalised to [0, 1] (matches original AstroPT)
        arr = np.array(image, dtype=np.float32)          # (512, 512, 3)
        arr = arr / 255.0                                 # [0, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, 512, 512)

        # Extract patches: (N_patches, patch_dim)
        p = self.patch_size
        patches = (
            tensor
            .unfold(1, p, p)   # (3, n_side, 256, p)   — unfold H
            .unfold(2, p, p)   # (3, n_side, n_side, p, p)
            .contiguous()
            .view(3, self._n_side * self._n_side, p * p)   # (3, N, p²)
            .permute(1, 0, 2)                              # (N, 3, p²)
            .reshape(-1, self._patch_dim)                  # (N, 768)
        )

        if self.spiral:
            patches = patches[self._spiral_idx]

        if torch.isnan(patches).any():
            raise ValueError("NaN in patches")

        return {
            "dr8_id": row["dr8_id"],
            "patches": patches,    # (1024, 768) — full sequence; split handled in model
        }

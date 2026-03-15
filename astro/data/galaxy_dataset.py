"""
GalaxyIterableDataset
=====================
Streams Smith42/galaxies revision="v2.0" from HuggingFace.  The v2.0
dataset already includes all metadata columns inline — no separate join
needed.

Usage
-----
    from astro.data.galaxy_dataset import GalaxyIterableDataset

    ds = GalaxyIterableDataset(split="validation")
    for item in ds:
        img = item["image"]           # PIL.Image 256×256 RGB
        z   = item.get("redshift")    # float | None
        m   = item.get("elpetro_mass_log")
        break
"""

from __future__ import annotations

from typing import Iterator, List, Optional

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
HF_REVISION   = "v2.0"   # includes all metadata columns inline

# ------------------------------------------------------------------
# Curated metadata fields forwarded from the dataset row.
# NaN / sentinel values (< -90) are dropped in __iter__.
# ------------------------------------------------------------------
METADATA_FIELDS: List[str] = [
    # Redshift
    "redshift",
    "photo_z",
    # Stellar mass
    "elpetro_mass_log",
    # Star formation rate
    "total_sfr_median",
    # DESI grz photometry
    "mag_g_desi",
    "mag_r_desi",
    "mag_z_desi",
    # Sérsic light-profile
    "sersic_n",
    "sersic_ba",
    # Galaxy Zoo vote fractions
    "smooth-or-featured_smooth_fraction",
    "smooth-or-featured_featured-or-disk_fraction",
    "has-spiral-arms_yes_fraction",
    "merging_merger_fraction",
    # Angular size
    "est_petro_th50",
]


class GalaxyIterableDataset(IterableDataset):
    """
    Streams Smith42/galaxies (v2.0) with images and metadata.

    Parameters
    ----------
    split : str
        "train", "validation", or "test".
    use_crop : bool
        True (default) → ``image_crop`` 256×256.
        False → full ``image`` 512×512.
    max_samples : int | None
        Cap on yielded examples; None = full split.
    hf_token : str | None
        HuggingFace auth token (dataset is public).
    buffer_size : int
        Streaming shuffle buffer.  0 = deterministic (eval).
    """

    def __init__(
        self,
        split: str = "validation",
        use_crop: bool = True,
        max_samples: Optional[int] = None,
        hf_token: Optional[str] = None,
        buffer_size: int = 1000,
    ):
        super().__init__()
        self.split = split
        self.use_crop = use_crop
        self.max_samples = max_samples
        self.hf_token = hf_token
        self.buffer_size = buffer_size
        self._image_col = "image_crop" if use_crop else "image"

    def __iter__(self) -> Iterator[dict]:
        import math as _math

        ds = _get_datasets()

        kwargs: dict = {
            "revision": HF_REVISION,
            "streaming": True,
            "trust_remote_code": False,
        }
        if self.hf_token:
            kwargs["token"] = self.hf_token

        hf_ds = ds.load_dataset(HF_DATASET_ID, split=self.split, **kwargs)

        # Shard across DataLoader workers for disjoint coverage
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

            image = row[self._image_col]
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image).convert("RGB")
            else:
                image = image.convert("RGB")

            item: dict = {
                "image":       image,
                "dr8_id":      row["dr8_id"],
                "galaxy_size": int(row["galaxy_size"]),
            }

            # Attach metadata fields, skipping missing / sentinel values
            for field in METADATA_FIELDS:
                val = row.get(field)
                if val is not None:
                    try:
                        v = float(val)
                        if _math.isfinite(v) and v > -90:
                            item[field] = v
                    except (TypeError, ValueError):
                        pass

            yield item
            count += 1

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

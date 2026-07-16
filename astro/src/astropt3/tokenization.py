"""Special-token vocabulary and patchification.

The special-token map is FROZEN: released checkpoints depend on these ids.
64 ids are reserved so that adding modalities never resizes the embedding.
Layout: 0 <|pad|>, 1 <|bos|>, then 3 consecutive ids per modality in
alphabetical registry order: <|begin_m|>, <|m|> (placeholder), <|end_m|>.
ids 8-63 are reserved for future modalities (time series, tabular, ...).

Patchification is pinned to the verified MMU schemas:
- images:  float32 (3, 152, 152) cubes, center-cropped to 96x96 by the
  sequencer -> patch 8 -> 144 tokens of 192 floats
- spectra: float32 (7781,) flux -> pad to 7936 -> 31 patches of 256 floats
"""

import einops
import numpy as np
import torch
import torch.nn.functional as F

VOCAB_SIZE = 64
PAD_ID = 0
BOS_ID = 1

# Frozen id assignment. Extend by appending modalities; never reorder.
_MODALITY_ID_BLOCKS = {
    "images": 2,  # begin=2, placeholder=3, end=4
    "spectra": 5,  # begin=5, placeholder=6, end=7
}


def modality_token_ids(name: str) -> tuple[int, int, int]:
    """Return (begin_id, placeholder_id, end_id) for a modality."""
    base = _MODALITY_ID_BLOCKS[name]
    return base, base + 1, base + 2


def special_token_map() -> dict[str, int]:
    """Human-readable token -> id map (for docs and released tokenizer config)."""
    tokens = {"<|pad|>": PAD_ID, "<|bos|>": BOS_ID}
    for name, base in _MODALITY_ID_BLOCKS.items():
        tokens[f"<|begin_{name}|>"] = base
        tokens[f"<|{name}|>"] = base + 1
        tokens[f"<|end_{name}|>"] = base + 2
    return tokens


def patchify_image(flux: torch.Tensor, patch_size: int) -> torch.Tensor:
    """(c, h, w) image cube -> (n_patches, patch_size*patch_size*c) tokens."""
    c, h, w = flux.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"image size {(h, w)} not divisible by patch size {patch_size}")
    return einops.rearrange(
        flux, "c (h p1) (w p2) -> (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
    )


def unpatchify_image(patches: torch.Tensor, patch_size: int, channels: int, side: int) -> torch.Tensor:
    """Inverse of :func:`patchify_image`."""
    n_side = side // patch_size
    return einops.rearrange(
        patches,
        "(h w) (p1 p2 c) -> c (h p1) (w p2)",
        h=n_side,
        w=n_side,
        p1=patch_size,
        p2=patch_size,
        c=channels,
    )


def patchify_spectrum(
    flux: torch.Tensor, lam: torch.Tensor, patch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D spectrum -> (n_patches, patch_size) tokens + per-patch mean wavelength.

    Flux is zero-padded up to a multiple of patch_size (as in astroPT); the
    per-patch position is the mean of the *unpadded* wavelengths in the patch
    (padded bins would drag the last patch's position toward zero).
    """
    (w,) = flux.shape
    pad_w = (patch_size - w % patch_size) % patch_size
    padded_flux = F.pad(flux, (0, pad_w))
    padded_lam = F.pad(lam, (0, pad_w))
    patches = einops.rearrange(padded_flux, "(n p) -> n p", p=patch_size)
    lam_patches = einops.rearrange(padded_lam, "(n p) -> n p", p=patch_size)
    counts = einops.rearrange(
        F.pad(torch.ones_like(lam), (0, pad_w)), "(n p) -> n p", p=patch_size
    ).sum(dim=1)
    lam_mean = lam_patches.sum(dim=1) / counts.clamp(min=1)
    return patches, lam_mean


def unpatchify_spectrum(patches: torch.Tensor, length: int) -> torch.Tensor:
    """Inverse of :func:`patchify_spectrum` (drops the zero padding)."""
    return einops.rearrange(patches, "n p -> (n p)")[:length]


def normalize_wavelength(lam: torch.Tensor) -> torch.Tensor:
    """Normalize wavelength in Angstroms to ~[0, 1] (astroPT convention)."""
    return (lam - 3000.0) / 7000.0


def spiral_index(n: int) -> np.ndarray:
    """Spiral index array of side length n (astroPT local_datasets._spiral)."""
    a = np.arange(n * n)
    b = a.reshape((n, n))
    m = None
    for i in range(n, 0, -2):
        m = np.r_[m, b[0, :], b[1:, -1], b[-1, :-1][::-1], b[1:-1, 0][::-1]]
        b = b[1:-1, 1:-1]
    a[list(m[1:])] = list(a)
    a = abs(a - n * n + 1)
    return a.reshape((n, n))


def spiralise(patches: torch.Tensor) -> torch.Tensor:
    """Reorder raster-order ViT patches into spiral order (astroPT Fig. 8)."""
    n = int(np.sqrt(len(patches)))
    assert n * n == len(patches), "spiralise needs a square-rootable patch count"
    indices = spiral_index(n).reshape(-1)
    out = torch.empty_like(patches)
    out[indices] = patches
    return out


def antispiralise(patches: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`spiralise`."""
    n = int(np.sqrt(len(patches)))
    assert n * n == len(patches), "antispiralise needs a square-rootable patch count"
    indices = spiral_index(n).reshape(-1)
    return patches[indices]

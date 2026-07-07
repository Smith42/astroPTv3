"""Per-modality normalization.

Pipeline for images: asinh stretch on the raw flux cube (compresses the huge
dynamic range of linear survey flux), then patchify, then per-patch
standardization (the astroPT convention: the model predicts standardized
patches). Spectra skip the stretch and are standardized per patch.
"""

import torch

EPS = 1e-8


def asinh_stretch(flux: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """asinh(flux / scale): linear near zero, logarithmic in the wings.

    ``scale`` should sit near the sky-noise level so faint structure stays
    linear; the pilot default is calibrated by scripts/compute_norm_stats.py.
    """
    return torch.asinh(flux / scale)


def per_patch_standardize(patches: torch.Tensor) -> torch.Tensor:
    """Standardize each patch vector to zero mean, unit std (astroPT train.py)."""
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, keepdim=True)
    return (patches - mean) / (std + EPS)

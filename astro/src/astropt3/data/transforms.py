"""Per-modality normalization.

Pipeline for images: physical band-registry normalization on the raw flux
cube (rescale to LegacySurvey nanomaggies -> bright-pixel clamp -> arcsinh
range compression; see :mod:`.band_registry`), then patchify, then per-patch
standardization (the astroPT convention: the model predicts standardized
patches; jetformer configs skip this step to keep the token map invertible).
Spectra skip the stretch and are standardized per patch.
"""

import torch

EPS = 1e-8


def per_patch_standardize(patches: torch.Tensor) -> torch.Tensor:
    """Standardize each patch vector to zero mean, unit std (astroPT train.py)."""
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, keepdim=True)
    return (patches - mean) / (std + EPS)

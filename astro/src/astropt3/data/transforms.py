"""Per-modality normalization.

Pipeline for images: asinh stretch on the raw flux cube (compresses the huge
dynamic range of linear survey flux), then patchify, then per-patch
standardization (the astroPT convention: the model predicts standardized
patches). Spectra skip the stretch and are standardized per patch.

The stretch follows the Platonic Universe recipe (Smith et al. 2026): per-band
flux percentiles set the offset/scale and a fixed steepness ``alpha = 20``
(Lupton et al. 2004) places the asinh knee at the 99th percentile. See
:func:`asinh_stretch` for the two PU steps we deliberately omit.
"""

import torch

EPS = 1e-8

# asinh stretch steepness; Lupton et al. (2004), the Platonic Universe default.
ASINH_ALPHA = 20.0


def _broadcast_band(value, flux: torch.Tensor) -> torch.Tensor:
    """Shape a scalar or per-band ``(C,)`` param to broadcast over ``flux``.

    ``flux`` is channel-first ``(C, H, W)``; a per-band vector becomes
    ``(C, 1, 1)`` and a scalar is left as-is.
    """
    t = torch.as_tensor(value, dtype=flux.dtype, device=flux.device)
    if t.ndim == 1:
        t = t.reshape(-1, *([1] * (flux.ndim - 1)))
    return t


def asinh_params_from_percentiles(p1, p99, alpha: float = ASINH_ALPHA):
    """PU ``(offset, scale)`` from per-band 1st/99th flux percentiles.

    ``offset = p1`` and ``scale = (p99 - p1) / alpha``, so that in
    :func:`asinh_stretch` flux at ``p1`` maps to 0 and the asinh knee sits at
    ``p99``. ``p1``/``p99`` are per-band (shape ``(C,)``), computed by
    ``scripts/compute_norm_stats.py`` (Phase 2) over batches of <=10k images.
    """
    p1 = torch.as_tensor(p1, dtype=torch.float32)
    p99 = torch.as_tensor(p99, dtype=torch.float32)
    return p1, (p99 - p1) / alpha


def asinh_stretch(
    flux: torch.Tensor,
    scale=1.0,
    offset=0.0,
) -> torch.Tensor:
    """Per-band asinh stretch: ``asinh((flux - offset) / scale)``.

    ``flux`` is a channel-first image cube ``(C, H, W)``. Under the Platonic
    Universe recipe ``offset``/``scale`` come from per-band flux percentiles
    via :func:`asinh_params_from_percentiles` (``offset = p1``,
    ``scale = (p99 - p1) / alpha``, ``alpha = 20``): faint structure stays
    linear near ``p1`` while bright cores are logarithmically compressed.
    Per-band values (shape ``(C,)``) broadcast over the channel axis; scalar
    defaults reproduce plain ``asinh(flux)``.

    PU additionally divides by ``asinh(alpha)`` and clips to ``[0, 1]``; we omit
    both. The global ``1/asinh(alpha)`` factor cancels under the downstream
    per-patch standardization, and clipping would flatten the galaxy cores the
    decoder heads must predict (AGENTS.md).
    """
    offset = _broadcast_band(offset, flux)
    scale = _broadcast_band(scale, flux)
    return torch.asinh((flux - offset) / scale)


def per_patch_standardize(patches: torch.Tensor) -> torch.Tensor:
    """Standardize each patch vector to zero mean, unit std (astroPT train.py)."""
    mean = patches.mean(dim=-1, keepdim=True)
    std = patches.std(dim=-1, keepdim=True)
    return (patches - mean) / (std + EPS)

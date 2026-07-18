"""Physical spectra normalization: DESI f_λ → AB nanomaggies → arcsinh.

ADR 0007, the spectra counterpart of :mod:`band_registry`: rescale the raw
survey flux to the AB/maggie system, then ``arcsinh(f_ν/knee)`` with the knee
at the modality's own sky-noise scale. Tokens are flux in knee units — no
per-corpus calibration, and the map is a fixed invertible change of variables
(required by the jetformer exact-likelihood loss).

DESI coadds ship ``flux`` in 10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹ (f_λ) on a wavelength
grid that is a *format constant* — 3600–9824 Å in 7781 bins of exactly 0.8 Å
(verified on ``pilot_v2`` shards) — so the λ²-dependent conversion to f_ν
inverts even for unconditionally sampled spectra, with no per-object side
information. The AB reference is 1 nMgy = 10⁻⁹·3631 Jy; ``f_ν = f_λ·λ²/c``.

Non-DESI spectra (future surveys) need their own unit/grid entries the way
``band_registry`` needs band entries; unknown grids raise rather than
silently pass through.
"""

from __future__ import annotations

import torch

# f_ν[nMgy] = f_λ[10⁻¹⁷ erg s⁻¹ cm⁻² Å⁻¹] · λ_Å² · FNU_NMGY_PER_FLAM
_C_ANGSTROM_PER_S = 2.99792458e18
_NMGY_ERG_S_CM2_HZ = 3.631e-29
FNU_NMGY_PER_FLAM = 1e-17 / (_C_ANGSTROM_PER_S * _NMGY_ERG_S_CM2_HZ)  # ≈9.19e-8

# arcsinh knee (nMgy): the measured DESI per-0.8 Å fiber sky-noise scale
# (σ p50 ≈ 1.9, p99 ≈ 15.7 nMgy on pilot_v2) — same design rule as the image
# knee (0.01 nMgy = broadband sky noise). Tokens land O(1): noise p50 ≈ 0.19,
# signal p50 ≈ 0.30, max ≈ 5.2.
_DIV_FACTOR = 10.0

# The DESI coadd wavelength grid, a constant of the data format.
DESI_LAMBDA_MIN = 3600.0
DESI_LAMBDA_STEP = 0.8
DESI_LAMBDA_BINS = 7781
DESI_LAMBDA_GRID = DESI_LAMBDA_MIN + DESI_LAMBDA_STEP * torch.arange(
    DESI_LAMBDA_BINS, dtype=torch.float64
)


def _check_desi_grid(lam: torch.Tensor) -> None:
    if lam.shape[-1] != DESI_LAMBDA_BINS or not torch.allclose(
        lam.double(), DESI_LAMBDA_GRID, atol=0.01
    ):
        raise NotImplementedError(
            "physical spectra normalization only supports the DESI coadd grid "
            f"({DESI_LAMBDA_BINS} bins, {DESI_LAMBDA_MIN}-"
            f"{DESI_LAMBDA_MIN + DESI_LAMBDA_STEP * (DESI_LAMBDA_BINS - 1)} A); "
            "add the new survey's unit/grid to data/spectral.py"
        )


def spectral_normalize(
    flux: torch.Tensor, lam: torch.Tensor, divisor: float = _DIV_FACTOR
) -> torch.Tensor:
    """DESI f_λ → ``arcsinh(f_ν/divisor)``: flux in knee units of 10 nMgy.

    ``lam`` is the record's wavelength grid in Å (broadcast against ``flux``);
    it must be the DESI format grid. ``divisor`` is the arcsinh knee in
    nanomaggies — pass the checkpoint's ``config.spectra_norm_divisor`` so
    data and inverse stay in the regime the model was trained on. Mask-zeroed
    pixels stay 0 (``arcsinh(0) = 0``).
    """
    _check_desi_grid(lam)
    fnu = flux * lam.to(flux.dtype) ** 2 * FNU_NMGY_PER_FLAM
    return torch.arcsinh(fnu / divisor)


def spectral_inverse(
    tokens: torch.Tensor, lam: torch.Tensor, divisor: float = _DIV_FACTOR
) -> torch.Tensor:
    """Invert :func:`spectral_normalize` back to DESI f_λ units.

    Exact (no clamp on this modality); ``divisor`` must match the forward
    pass (the checkpoint's ``config.spectra_norm_divisor``).
    """
    _check_desi_grid(lam)
    fnu = torch.sinh(tokens) * divisor
    return fnu / (lam.to(tokens.dtype) ** 2 * FNU_NMGY_PER_FLAM)

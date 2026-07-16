"""Physical band-registry image normalization (data/band_registry.py)."""

import math

import pytest
import torch

from astropt3.data.band_registry import (
    _DIV_FACTOR,
    BAND_REGISTRY,
    RAW_BANDS,
    clamp_flux,
    physical_inverse,
    physical_normalize,
    rescale_factors,
)

DES_BANDS = ["des-g", "des-r", "des-z"]


def test_registry_constants():
    # DES is the reference scale -> identity rescale.
    assert rescale_factors("des-g") == (1.0, 1.0)
    for band in BAND_REGISTRY:
        zpscale, pxscale = rescale_factors(band)
        assert math.isfinite(zpscale) and zpscale > 0, band
        assert math.isfinite(pxscale) and pxscale > 0, band
        cf = clamp_flux(band)
        assert math.isfinite(cf) and cf > 0, band


def test_normalize_des_is_arcsinh_over_divisor():
    torch.manual_seed(0)
    flux = torch.randn(3, 8, 8) * 0.05  # nMgy-scale, below the ~398 nMgy clamp
    out = physical_normalize(flux, DES_BANDS)
    assert torch.allclose(out, torch.arcsinh(flux / _DIV_FACTOR) * _DIV_FACTOR)
    assert torch.isfinite(out).all()
    # output is bounded by the compressed bright ceiling
    ceiling = math.asinh(clamp_flux("des-g") / _DIV_FACTOR) * _DIV_FACTOR
    assert out.abs().max() <= ceiling


def test_raw_bands_pass_through():
    flux = torch.randn(3, 4, 4) * 100
    out = physical_normalize(flux, ["rgb-r", "rgb-g", "rgb-b"])
    assert out is flux
    assert physical_inverse(flux, ["rgb-r", "rgb-g", "rgb-b"]) is flux


def test_unknown_band_raises():
    flux = torch.randn(1, 4, 4)
    with pytest.raises(NotImplementedError, match="euclid-g"):
        physical_normalize(flux, ["euclid-g"])
    with pytest.raises(NotImplementedError, match="euclid-g"):
        physical_inverse(flux, ["euclid-g"])


def test_empty_band_list_passes_through():
    # all(...) is vacuously True on [] -> nothing to key the physics on
    flux = torch.randn(3, 4, 4)
    assert physical_normalize(flux, []) is flux


def test_roundtrip_below_clamp_is_exact_and_clamp_is_lossy():
    torch.manual_seed(0)
    flux = torch.randn(3, 8, 8).double() * 10  # well below the DES ~398 nMgy clamp
    back = physical_inverse(physical_normalize(flux, DES_BANDS), DES_BANDS)
    assert torch.allclose(back, flux, atol=1e-9)

    bright = torch.full((3, 2, 2), 2 * clamp_flux("des-g"), dtype=torch.float64)
    clipped = physical_inverse(physical_normalize(bright, DES_BANDS), DES_BANDS)
    assert torch.allclose(clipped, torch.full_like(bright, clamp_flux("des-g")))

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
    assert torch.allclose(out, torch.arcsinh(flux / _DIV_FACTOR))
    assert torch.isfinite(out).all()
    # output is bounded by the compressed bright ceiling
    ceiling = math.asinh(clamp_flux("des-g") / _DIV_FACTOR)
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


def test_non_default_divisor_changes_output_and_roundtrips():
    torch.manual_seed(0)
    flux = torch.randn(3, 8, 8).double() * 0.05
    default = physical_normalize(flux, DES_BANDS)
    moved = physical_normalize(flux, DES_BANDS, divisor=0.5)
    assert not torch.allclose(default, moved)  # the knob actually acts
    back = physical_inverse(moved, DES_BANDS, divisor=0.5)
    assert torch.allclose(back, flux, atol=1e-9)


def test_sequencer_uses_config_divisor(tiny_config):
    """config.image_norm_divisor must reach the sequencer's normalization."""
    from astropt3 import AstroPT3Config
    from astropt3.data.packing import IMAGE_CROP, ObjectSequencer
    from astropt3.data.synthetic import make_record
    from astropt3.tokenization import patchify_image

    record = make_record(3)
    moved_config = AstroPT3Config(
        **{**tiny_config.to_dict(), "tokeniser": "jetformer", "image_norm_divisor": 0.5}
    )
    seq = ObjectSequencer(moved_config).build(record)
    flux = torch.as_tensor(record["image"]["flux"])
    h, w = flux.shape[-2:]
    top, left = (h - IMAGE_CROP) // 2, (w - IMAGE_CROP) // 2
    flux = flux[..., top : top + IMAGE_CROP, left : left + IMAGE_CROP]
    expected = patchify_image(
        physical_normalize(flux, record["image"]["band"], divisor=0.5), 8
    )
    assert torch.allclose(seq.values["images"], expected)
    default_seq = ObjectSequencer(
        AstroPT3Config(**{**tiny_config.to_dict(), "tokeniser": "jetformer"})
    ).build(record)
    assert not torch.allclose(seq.values["images"], default_seq.values["images"])

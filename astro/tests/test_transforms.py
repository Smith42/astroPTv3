import math

import torch

from astropt3.data.packing import ObjectSequencer
from astropt3.data.synthetic import make_record
from astropt3.data.transforms import (
    ASINH_ALPHA,
    asinh_params_from_percentiles,
    asinh_stretch,
)


def test_pu_stretch_anchors_percentiles():
    # Platonic Universe recipe: flux at p1 -> 0, and the asinh knee at p99
    # gives asinh(alpha) before the (omitted) 1/asinh(alpha) normalization.
    p1 = torch.tensor([1.0, 2.0, 3.0])
    p99 = torch.tensor([11.0, 22.0, 33.0])
    offset, scale = asinh_params_from_percentiles(p1, p99)
    assert torch.allclose(scale, (p99 - p1) / ASINH_ALPHA)

    flux_p1 = p1.reshape(3, 1, 1) * torch.ones(3, 2, 2)
    flux_p99 = p99.reshape(3, 1, 1) * torch.ones(3, 2, 2)
    assert torch.allclose(asinh_stretch(flux_p1, scale, offset), torch.zeros(3, 2, 2), atol=1e-6)
    assert torch.allclose(
        asinh_stretch(flux_p99, scale, offset),
        torch.full((3, 2, 2), math.asinh(ASINH_ALPHA)),
        atol=1e-5,
    )


def test_pu_stretch_is_per_band():
    # Different per-band offset/scale must act independently on each channel.
    flux = torch.stack([torch.full((4, 4), 5.0), torch.full((4, 4), 5.0), torch.full((4, 4), 5.0)])
    offset, scale = asinh_params_from_percentiles(
        torch.tensor([0.0, 5.0, 0.0]), torch.tensor([100.0, 25.0, 20.0])
    )
    out = asinh_stretch(flux, scale, offset)
    # band 1 sits exactly at its p1 -> 0; bands 0 and 2 do not.
    assert out[1].abs().max() < 1e-6
    assert out[0].abs().min() > 0 and out[2].abs().min() > 0


def test_stretch_scalar_default_is_plain_asinh():
    flux = torch.randn(3, 8, 8) * 10
    assert torch.allclose(asinh_stretch(flux), torch.asinh(flux))


def test_sequencer_without_stats_matches_plain_asinh(tiny_config):
    # No calibration -> plain asinh(flux), the smoke-path fallback.
    seq = ObjectSequencer(tiny_config)
    assert seq.asinh_offset == 0.0 and seq.asinh_scale == 1.0


def test_sequencer_with_percentiles_uses_pu_params(tiny_config):
    seq = ObjectSequencer(
        tiny_config, image_p1=[0.0, 0.0, 0.0], image_p99=[10.0, 20.0, 30.0]
    )
    _, scale = asinh_params_from_percentiles([0.0, 0.0, 0.0], [10.0, 20.0, 30.0])
    assert torch.allclose(seq.asinh_scale, scale)
    # a record still sequences into standardized patches
    obj = seq.build(make_record(3))
    assert torch.isfinite(obj.values["images"]).all()

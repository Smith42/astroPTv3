"""Fixed physical normalization for the ADR 0008 scalar modalities.

Mirrors ``band_registry.py`` and honours the ADR 0007 discipline: every
transform is fixed, invertible, physically motivated, and carries no fitted
constants — a checkpoint is never coupled to the corpus statistics it was
trained on. Unknown scalar names raise.

- ``Z``          -> ``log(1 + z)``: the standard photometric-redshift working
                    variable; errors become naturally fractional, so a
                    predicted sigma reads as the literature's ``dz/(1+z)``.
- ``ebv``        -> ``ebv / 0.1``: fixed knee putting typical Galactic
                    extinction at O(1) (band-registry philosophy).
- ``photometry`` -> ``arcsinh(f / 0.01 nMgy)`` per band: literally the band
                    registry's image transform, so aperture photometry and
                    image pixels live in the same units.
"""

from __future__ import annotations

import torch

from .band_registry import _DIV_FACTOR

# ebv knee: typical Galactic E(B-V) is a few hundredths of a magnitude
_EBV_DIV = 0.1


def _photometry_fwd(x: torch.Tensor) -> torch.Tensor:
    return torch.arcsinh(x / _DIV_FACTOR)


def _photometry_inv(x: torch.Tensor) -> torch.Tensor:
    return torch.sinh(x) * _DIV_FACTOR


# name -> (forward, inverse); both operate elementwise on tensors
SCALAR_TRANSFORMS = {
    "Z": (torch.log1p, torch.expm1),
    "ebv": (lambda x: x / _EBV_DIV, lambda x: x * _EBV_DIV),
    "photometry": (_photometry_fwd, _photometry_inv),
}


def _transforms(name: str):
    if name not in SCALAR_TRANSFORMS:
        raise NotImplementedError(
            f"no scalar normalization for {name!r}; add it to "
            "scalar_registry.SCALAR_TRANSFORMS"
        )
    return SCALAR_TRANSFORMS[name]


def scalar_normalize(name: str, value: torch.Tensor) -> torch.Tensor:
    return _transforms(name)[0](value)


def scalar_inverse(name: str, value: torch.Tensor) -> torch.Tensor:
    return _transforms(name)[1](value)


if __name__ == "__main__":
    for name in SCALAR_TRANSFORMS:
        x = torch.tensor([0.0, 0.03, 0.7, 1.5, 42.0])
        rt = scalar_inverse(name, scalar_normalize(name, x))
        assert torch.allclose(rt, x, atol=1e-5), name
    try:
        scalar_normalize("sSFR", torch.tensor(1.0))
    except NotImplementedError:
        pass
    else:
        raise AssertionError("unknown scalar must raise")
    print(f"ok: {len(SCALAR_TRANSFORMS)} scalar transforms round-trip")

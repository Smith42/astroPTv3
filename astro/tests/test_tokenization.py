import torch

from astropt3.tokenization import (
    BOS_ID,
    PAD_ID,
    VOCAB_SIZE,
    antispiralise,
    modality_token_ids,
    patchify_image,
    patchify_spectrum,
    special_token_map,
    spiralise,
    unpatchify_image,
    unpatchify_spectrum,
)


def test_special_token_ids_frozen():
    assert PAD_ID == 0 and BOS_ID == 1
    assert modality_token_ids("images") == (2, 3, 4)
    assert modality_token_ids("spectra") == (5, 6, 7)
    ids = set(special_token_map().values())
    assert len(ids) == 8 and max(ids) < VOCAB_SIZE


def test_image_patchify_roundtrip():
    flux = torch.randn(3, 152, 152)
    patches = patchify_image(flux, patch_size=8)
    assert patches.shape == (361, 192)
    back = unpatchify_image(patches, patch_size=8, channels=3, side=152)
    assert torch.equal(back, flux)


def test_spectrum_patchify_roundtrip():
    flux = torch.randn(7781)
    lam = torch.linspace(3600.0, 9824.0, 7781)
    patches, lam_mean = patchify_spectrum(flux, lam, patch_size=256)
    assert patches.shape == (31, 256)
    assert lam_mean.shape == (31,)
    back = unpatchify_spectrum(patches, length=7781)
    assert torch.equal(back, flux)


def test_spectrum_last_patch_position_ignores_padding():
    flux = torch.randn(7781)
    lam = torch.linspace(3600.0, 9824.0, 7781)
    _, lam_mean = patchify_spectrum(flux, lam, patch_size=256)
    # last patch holds 7781 - 30*256 = 101 real bins + 155 padded zeros
    expected = lam[30 * 256 :].mean()
    assert torch.allclose(lam_mean[-1], expected)
    # a padded-mean would be dragged far below the true value
    assert lam_mean[-1] > lam_mean[-2]


def test_spiralise_roundtrip():
    patches = torch.randn(361, 192)
    assert torch.equal(antispiralise(spiralise(patches)), patches)
    assert not torch.equal(spiralise(patches), patches)

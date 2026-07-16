"""Sampling/generation from a jetformer checkpoint (astropt3.generation)."""

from pathlib import Path

import pytest
import torch

from astropt3.config_io import load_model_config
from astropt3.data.packing import ObjectSequencer
from astropt3.data.synthetic import make_record
from astropt3.generation import generate, reconstruct, sample_gmm

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "model" / "test-tiny-jetformer.yaml"


@pytest.fixture(scope="module")
def jet_config():
    config, _ = load_model_config(CONFIG)
    return config


@pytest.fixture(scope="module")
def smoke_model(jet_config, tmp_path_factory):
    """A briefly-trained tiny jetformer model, save/load-roundtripped."""
    from transformers import AutoModel

    from astropt3.train_smoke import configure_optimizer, make_batches
    from astropt3.modeling_astropt3 import AstroPT3Model

    torch.manual_seed(0)
    model = AstroPT3Model(jet_config)
    model.train()
    opt = configure_optimizer(model, lr=1e-3)
    for batch in make_batches(jet_config, n_objects=10, objects_per_batch=2, seq_len=896):
        out = model(**batch)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        opt.step()
    save_dir = tmp_path_factory.mktemp("ckpt") / "jet"
    model.save_pretrained(save_dir)
    return AutoModel.from_pretrained(save_dir).eval()


@pytest.fixture(scope="module")
def template(jet_config):
    # image_only_fraction=0 so the template has both spans
    return ObjectSequencer(jet_config).build(make_record(3, image_only_fraction=0.0))


def test_sample_gmm_argmax_is_mixture_mean():
    torch.manual_seed(0)
    logits_pi = torch.randn(5, 3)
    mu = torch.randn(5, 3, 4)
    log_sigma = torch.randn(5, 3, 4)
    got = sample_gmm(logits_pi, mu, log_sigma, argmax=True)
    pi = torch.softmax(logits_pi, dim=-1)
    assert torch.allclose(got, (pi.unsqueeze(-1) * mu).sum(-2))


def test_unconditional_shapes(smoke_model, template):
    g = torch.Generator().manual_seed(0)
    out = generate(smoke_model, template, {"images", "spectra"}, n=2, generator=g)
    assert set(out) == {"images", "spectra"}
    assert out["images"].shape == (2, 144, 192)
    assert out["spectra"].shape == (2, 31, 256)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_seeding_reproducible(smoke_model, template):
    # spectra-only (31 tokens): the full-span draw above is the expensive one
    def draw(seed):
        g = torch.Generator().manual_seed(seed)
        return generate(smoke_model, template, {"spectra"}, n=2, generator=g)["spectra"]

    a = draw(0)
    assert torch.allclose(draw(0), a)  # same seed reproduces
    assert not torch.allclose(draw(1), a)  # different seed differs


def test_argmax_is_deterministic_without_generator(smoke_model, template):
    a = generate(smoke_model, template, {"spectra"}, n=1, argmax=True)
    b = generate(smoke_model, template, {"spectra"}, n=1, argmax=True)
    assert torch.allclose(a["spectra"], b["spectra"])


def test_image_to_spectra_teacher_forces_images(smoke_model, template):
    g = torch.Generator().manual_seed(0)
    out = generate(smoke_model, template, {"spectra"}, n=1, generator=g)
    assert set(out) == {"spectra"}  # teacher-forced spans are not returned
    assert out["spectra"].shape == (1, 31, 256)


def test_generate_rejects_affine_and_missing_span(smoke_model, template, jet_config):
    from astropt3 import AstroPT3Config, AstroPT3Model

    affine = AstroPT3Model(AstroPT3Config(**{**jet_config.to_dict(), "tokeniser": "affine"}))
    with pytest.raises(ValueError, match="not jetformer"):
        generate(affine, template, {"images"})

    image_only = ObjectSequencer(jet_config).build(make_record(0, image_only_fraction=1.1))
    with pytest.raises(ValueError, match="spectra"):
        generate(smoke_model, image_only, {"spectra"})


def test_reconstruct_shapes(smoke_model, template):
    preds = reconstruct(smoke_model, template)
    assert preds["images"].shape == (144, 192)
    assert preds["spectra"].shape == (31, 256)
    assert all(torch.isfinite(v).all() for v in preds.values())

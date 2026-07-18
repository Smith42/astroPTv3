"""Randomized 50/50 bimodal span order (ADR 0005 amendment)."""

import pytest
import torch

from astropt3 import AstroPT3Config
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.data.synthetic import make_record
from astropt3.tokenization import modality_token_ids


def _first_begin(seq):
    """Name of the modality whose span opens first in the sequence."""
    begins = {modality_token_ids(m)[0]: m for m in ("images", "spectra")}
    for tok in seq.input_ids.tolist():
        if tok in begins:
            return begins[tok]
    raise AssertionError("no modality span found")


def _config(tiny_config, **over):
    return AstroPT3Config(**{**tiny_config.to_dict(), **over})


def _bimodal(i):
    return make_record(i, image_only_fraction=0.0)


def test_off_by_default_keeps_alphabetical_order(tiny_config):
    seq = ObjectSequencer(tiny_config)
    assert not seq.shuffle_modality_order
    for i in range(5):
        assert _first_begin(seq.build(_bimodal(i))) == "images"


def test_shuffle_is_deterministic_5050_and_flips_with_epoch(tiny_config):
    seq = ObjectSequencer(_config(tiny_config, shuffle_modality_order=True))
    firsts = [_first_begin(seq.build(_bimodal(i))) for i in range(200)]
    frac = firsts.count("spectra") / len(firsts)
    assert 0.35 < frac < 0.65  # crc32 parity is ~uniform over object ids
    # deterministic per (record, epoch); flips when the epoch flips parity
    for i in range(5):
        assert _first_begin(seq.build(_bimodal(i))) == firsts[i]
        assert _first_begin(seq.build(_bimodal(i), epoch=1)) != firsts[i]


def test_single_modality_records_are_unaffected(tiny_config):
    seq = ObjectSequencer(_config(tiny_config, shuffle_modality_order=True))
    image_only = make_record(4, image_only_fraction=1.0)
    assert _first_begin(seq.build(image_only)) == "images"
    spec_only = make_record(4, image_only_fraction=0.0, spectrum_only_fraction=1.0)
    assert _first_begin(seq.build(spec_only)) == "spectra"


def test_explicit_order_override_and_validation(tiny_config):
    seq = ObjectSequencer(tiny_config)
    record = _bimodal(3)
    rev = seq.build(record, modality_order=["spectra", "images"])
    assert _first_begin(rev) == "spectra"
    # per-modality payloads are identical either way — only the skeleton moves
    fwd = seq.build(record)
    for m in ("images", "spectra"):
        assert torch.equal(rev.values[m], fwd.values[m])
    with pytest.raises(ValueError, match="exactly"):
        seq.build(record, modality_order=["spectra"])
    with pytest.raises(ValueError, match="exactly"):
        seq.build(
            make_record(4, image_only_fraction=1.0),
            modality_order=["spectra", "images"],
        )


def test_model_forward_on_spectra_first_batch(tiny_config):
    """The mask-based loss alignment must not care about span order."""
    from astropt3.modeling_astropt3 import AstroPT3Model

    config = _config(tiny_config, shuffle_modality_order=True)
    torch.manual_seed(0)
    model = AstroPT3Model(config).eval()
    seq = ObjectSequencer(config)
    objs = [
        seq.build(_bimodal(3), modality_order=["spectra", "images"]),
        seq.build(_bimodal(5), modality_order=["images", "spectra"]),
    ]
    batch = PackedCollator(config, seq_len=896)(objs)
    out = model(**batch)
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"images", "spectra"}


def test_spectra_to_images_mode(tiny_config):
    """spectra-to-images samples an image span conditioned on the spectrum."""
    from astropt3.eval.samples import build_template, sample_template
    from astropt3.modeling_astropt3 import AstroPT3Model

    config = _config(
        tiny_config, tokeniser="jetformer", shuffle_modality_order=True
    )
    torch.manual_seed(0)
    model = AstroPT3Model(config).eval()
    seq = ObjectSequencer(config)
    record = _bimodal(3)
    template = build_template(seq, record, "spectra-to-images")
    generator = torch.Generator().manual_seed(0)
    sampled = sample_template(model, template, "spectra-to-images", n=2, generator=generator)
    assert set(sampled) == {"images"}
    assert sampled["images"].shape == (2, *template.values["images"].shape)
    assert torch.isfinite(sampled["images"]).all()
    # an images-first template must be rejected — no conditioning flows
    images_first = seq.build(record, modality_order=["images", "spectra"])
    with pytest.raises(ValueError, match="spectra-first"):
        sample_template(model, images_first, "spectra-to-images", n=1)


def test_default_modes_gates_on_the_config_flag(tiny_config):
    from astropt3.eval.samples import default_modes

    fixed = _config(tiny_config, tokeniser="jetformer")
    assert "spectra-to-images" not in default_modes(fixed)
    shuffled = _config(
        tiny_config, tokeniser="jetformer", shuffle_modality_order=True
    )
    assert "spectra-to-images" in default_modes(shuffled)

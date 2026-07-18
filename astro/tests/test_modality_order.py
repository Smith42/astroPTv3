"""Uniform random span order, seeded per (object_id, epoch) (ADR 0008)."""

import pytest
import torch

from astropt3 import AstroPT3Config
from astropt3.data.packing import ObjectSequencer, PackedCollator
from astropt3.data.synthetic import make_record
from astropt3.tokenization import _MODALITY_ID_BLOCKS, modality_token_ids


def _first_begin(seq):
    """Name of the modality whose span opens first in the sequence."""
    begins = {modality_token_ids(m)[0]: m for m in _MODALITY_ID_BLOCKS}
    for tok in seq.input_ids.tolist():
        if tok in begins:
            return begins[tok]
    raise AssertionError("no modality span found")


def _bimodal(i):
    return make_record(i, image_only_fraction=0.0)


def test_shuffle_is_uniform_and_deterministic(sequencer):
    # N=2 (scalar-free): the shuffle reduces to ADR 0005's 50/50 flip
    firsts = [
        _first_begin(sequencer.build(_bimodal(i), include_scalars=False))
        for i in range(200)
    ]
    frac = firsts.count("spectra") / len(firsts)
    assert 0.35 < frac < 0.65  # seeded shuffle is ~uniform over object ids
    # N=5 (all spans): every modality leads somewhere, roughly uniformly
    full_firsts = [_first_begin(sequencer.build(_bimodal(i))) for i in range(300)]
    counts = {m: full_firsts.count(m) for m in set(full_firsts)}
    assert set(counts) == {"images", "spectra", "Z", "ebv", "photometry"}
    assert all(0.1 < c / 300 < 0.35 for c in counts.values()), counts
    # deterministic per (record, epoch); a new epoch reseeds the shuffle
    for i in range(5):
        assert _first_begin(sequencer.build(_bimodal(i))) == full_firsts[i]
    epoch1 = [
        _first_begin(sequencer.build(_bimodal(i), epoch=1)) for i in range(300)
    ]
    assert epoch1 != full_firsts  # the aggregate order sequence moves
    assert epoch1 == [
        _first_begin(sequencer.build(_bimodal(i), epoch=1)) for i in range(300)
    ]


def test_single_span_records_are_unaffected(sequencer):
    image_only = make_record(4, image_only_fraction=1.0)
    assert _first_begin(sequencer.build(image_only, include_scalars=False)) == "images"
    spec_only = make_record(4, image_only_fraction=0.0, spectrum_only_fraction=1.0)
    assert _first_begin(sequencer.build(spec_only, include_scalars=False)) == "spectra"


def test_explicit_order_override_and_validation(sequencer):
    record = _bimodal(3)
    rev = sequencer.build(
        record, modality_order=["spectra", "images"], include_scalars=False
    )
    assert _first_begin(rev) == "spectra"
    fwd = sequencer.build(
        record, modality_order=["images", "spectra"], include_scalars=False
    )
    assert _first_begin(fwd) == "images"
    # per-modality payloads are identical either way — only the skeleton moves
    for m in ("images", "spectra"):
        assert torch.equal(rev.values[m], fwd.values[m])
    with pytest.raises(ValueError, match="exactly"):
        sequencer.build(record, modality_order=["spectra"], include_scalars=False)
    with pytest.raises(ValueError, match="exactly"):
        sequencer.build(
            make_record(4, image_only_fraction=1.0),
            modality_order=["spectra", "images"],
            include_scalars=False,
        )


def test_model_forward_on_mixed_order_batch(tiny_config):
    """The mask-based loss alignment must not care about span order."""
    from astropt3.modeling_astropt3 import AstroPT3Model

    torch.manual_seed(0)
    model = AstroPT3Model(tiny_config).eval()
    seq = ObjectSequencer(tiny_config)
    objs = [
        seq.build(_bimodal(3), modality_order=["spectra", "images"], include_scalars=False),
        seq.build(_bimodal(5), modality_order=["images", "spectra"], include_scalars=False),
    ]
    batch = PackedCollator(tiny_config, seq_len=896)(objs)
    out = model(**batch)
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"images", "spectra"}


def test_spectra_to_images_mode(tiny_config):
    """spectra-to-images samples an image span conditioned on the spectrum."""
    from astropt3.eval.samples import build_template, sample_template
    from astropt3.modeling_astropt3 import AstroPT3Model

    config = AstroPT3Config(**{**tiny_config.to_dict(), "tokeniser": "jetformer"})
    torch.manual_seed(0)
    model = AstroPT3Model(config).eval()
    seq = ObjectSequencer(config)
    record = _bimodal(3)
    template = build_template(seq, record, "spectra-to-images")
    assert set(template.masks) == {"images", "spectra"}  # scalar-free (ADR 0008)
    generator = torch.Generator().manual_seed(0)
    sampled = sample_template(model, template, "spectra-to-images", n=2, generator=generator)
    assert set(sampled) == {"images"}
    assert sampled["images"].shape == (2, *template.values["images"].shape)
    assert torch.isfinite(sampled["images"]).all()
    # an images-first template must be rejected — no conditioning flows
    images_first = seq.build(
        record, modality_order=["images", "spectra"], include_scalars=False
    )
    with pytest.raises(ValueError, match="spectra-first"):
        sample_template(model, images_first, "spectra-to-images", n=1)


def test_default_modes_includes_spectra_to_images(tiny_config):
    from astropt3.eval.samples import default_modes

    jet = AstroPT3Config(**{**tiny_config.to_dict(), "tokeniser": "jetformer"})
    assert default_modes(jet) == [
        "unconditional",
        "image-to-spectra",
        "spectra-to-images",
    ]
    assert default_modes(tiny_config) == ["reconstruct"]

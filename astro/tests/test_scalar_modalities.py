"""ADR 0008 scalar modalities: registry, spans, loss, generation, metrics."""

import pytest
import torch

from astropt3 import AstroPT3Config
from astropt3.data.scalar_registry import scalar_inverse, scalar_normalize
from astropt3.data.synthetic import make_record
from astropt3.modalities import gmm_nll


def test_scalar_registry_roundtrip_and_unknown_raises():
    x = torch.tensor([0.0, 0.03, 0.7, 1.5, 42.0])
    for name in ("Z", "ebv", "photometry"):
        rt = scalar_inverse(name, scalar_normalize(name, x))
        assert torch.allclose(rt, x, atol=1e-5), name
    with pytest.raises(NotImplementedError):
        scalar_normalize("sSFR", x)


def test_scalar_spans_and_gating(sequencer, full_record):
    obj = sequencer.build(full_record)
    assert {"Z", "ebv", "photometry"} <= set(obj.masks)
    assert obj.values["Z"].shape == (1, 1)
    assert obj.values["photometry"].shape == (1, 3)
    # normalized truth round-trips to the record value
    z = float(scalar_inverse("Z", obj.values["Z"][0, 0]))
    assert abs(z - full_record["Z"]) < 1e-5
    # ZWARN != 0 suppresses the Z span (ADR 0005's reliability cut, reused)
    bad = dict(full_record, ZWARN=True)
    assert "Z" not in sequencer.build(bad).masks
    # missing scalar fields are absent spans, not errors
    partial = {k: v for k, v in full_record.items() if k != "flux_r"}
    assert "photometry" not in sequencer.build(partial).masks
    # include_scalars=False strips every scalar span (probe sequences)
    assert set(sequencer.build(full_record, include_scalars=False).masks) == {
        "images",
        "spectra",
    }


def test_scalar_loss_matches_manual_gmm_nll(tiny_model, sequencer, collator, full_record):
    """Scalar losses are gmm_nll on the raw normalized value, no logdet."""
    batch = collator([sequencer.build(full_record, modality_order=[
        "images", "spectra", "Z", "ebv", "photometry"
    ])])
    with torch.no_grad():
        out = tiny_model(**batch)
        from astropt3.modeling_astropt3 import left_shift_mask

        for name in ("Z", "ebv", "photometry"):
            mask = batch["modality_masks"][name]
            hidden = out.last_hidden_state[left_shift_mask(mask)]
            logits_pi, mu, log_sigma = tiny_model.decoders[name](hidden)
            manual = gmm_nll(
                batch["modality_values"][name], logits_pi, mu, log_sigma
            ).mean()
            assert torch.allclose(out.modality_losses[name], manual, atol=1e-6), name


def test_scalar_heads_under_both_tokenisers(tiny_config):
    from astropt3 import AstroPT3Model
    from astropt3.modalities import GMMHead

    jet_config = AstroPT3Config(**{**tiny_config.to_dict(), "tokeniser": "jetformer"})
    for config in (tiny_config, jet_config):
        model = AstroPT3Model(config)
        for name in ("Z", "ebv", "photometry"):
            assert isinstance(model.decoders[name], GMMHead)
            assert model.decoders[name].k == config.scalar_gmm_k
            if config.tokeniser == "jetformer":
                assert name not in model.flows  # scalars never flow


def test_unconditional_generation_covers_scalars(tiny_config):
    from astropt3 import AstroPT3Model
    from astropt3.data.packing import ObjectSequencer
    from astropt3.generation import generate

    config = AstroPT3Config(**{**tiny_config.to_dict(), "tokeniser": "jetformer"})
    torch.manual_seed(0)
    model = AstroPT3Model(config).eval()
    template = ObjectSequencer(config).build(make_record(3, image_only_fraction=0.0))
    sampled = generate(
        model, template, set(template.masks), n=1,
        generator=torch.Generator().manual_seed(0),
    )
    assert {"Z", "ebv", "photometry"} <= set(sampled)
    assert sampled["Z"].shape == (1, 1, 1)
    assert sampled["photometry"].shape == (1, 1, 3)
    assert all(torch.isfinite(v).all() for v in sampled.values())


def test_scalar_head_metrics(tiny_model):
    from astropt3.eval.scalar_head import collect_scalar_objects, scalar_head_metrics

    objects, targets = collect_scalar_objects(
        tiny_model.config, "synthetic", "Z", n_objects=16
    )
    assert len(objects) == 16
    # the target span is pinned last so every observation conditions it
    for obj in objects:
        assert int(obj.masks["Z"].nonzero()[0]) > max(
            int(obj.masks[m].nonzero()[-1]) for m in obj.masks if m != "Z"
        )
    result = scalar_head_metrics(tiny_model, objects, targets, target="Z")
    assert result["n_objects"] == 16
    for key in ("nmad", "outlier_frac", "coverage_1sig", "bias", "r2"):
        assert torch.isfinite(torch.tensor(result[key])), key
    assert result["r2"] <= 1.0

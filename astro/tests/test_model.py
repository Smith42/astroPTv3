import torch

from astropt3.data.synthetic import record_stream


def _batch(sequencer, collator, records):
    return collator([sequencer.build(r) for r in records])


def test_forward_backward(tiny_config, sequencer, collator):
    from astropt3 import AstroPT3Model

    torch.manual_seed(0)
    model = AstroPT3Model(tiny_config)
    model.train()
    records = list(record_stream(4))
    batch = _batch(sequencer, collator, records)
    out = model(**batch)
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"images", "spectra", "Z", "ebv", "photometry"}
    assert all(torch.isfinite(v) for v in out.modality_losses.values())
    out.loss.backward()
    missing = [
        n for n, p in model.named_parameters() if p.requires_grad and p.grad is None
    ]
    assert not missing, f"params without grad: {missing}"


def test_pad_invariance(tiny_model, tiny_config, sequencer, full_record):
    """Trailing pad tokens must not change the loss at all."""
    from astropt3.data.packing import PackedCollator

    obj = sequencer.build(full_record)
    tight = PackedCollator(tiny_config, seq_len=397)([obj])
    padded = PackedCollator(tiny_config, seq_len=520)([obj])
    with torch.no_grad():
        loss_tight = tiny_model(**tight).loss
        loss_padded = tiny_model(**padded).loss
    assert torch.allclose(loss_tight, loss_padded, atol=1e-6)


def test_doc_mask_blocks_cross_object_attention(tiny_model, tiny_config, sequencer):
    """Perturbing object A's pixels must not move object B's hidden states."""
    from astropt3.data.packing import PackedCollator

    records = list(record_stream(2))
    objs = [sequencer.build(r) for r in records]
    collator = PackedCollator(tiny_config, seq_len=len(objs[0]) + len(objs[1]))
    batch = collator(objs)
    assert batch["input_ids"].shape[0] == 1  # both objects share one row

    perturbed = {
        "input_ids": batch["input_ids"],
        "position_ids": batch["position_ids"],
        "modality_masks": batch["modality_masks"],
        "modality_positions": batch["modality_positions"],
        "modality_values": {
            k: v.clone() for k, v in batch["modality_values"].items()
        },
    }
    n_a = objs[0].values["images"].shape[0]
    perturbed["modality_values"]["images"][:n_a] += 10.0

    with torch.no_grad():
        h0 = tiny_model(**batch, compute_loss=False).last_hidden_state
        h1 = tiny_model(**perturbed, compute_loss=False).last_hidden_state

    start_b = len(objs[0])
    assert not torch.allclose(h0[0, :start_b], h1[0, :start_b])  # A moved
    assert torch.equal(h0[0, start_b:], h1[0, start_b:])  # B bit-identical


def test_image_only_batch(tiny_model, sequencer, collator, image_only_record):
    batch = collator([sequencer.build(image_only_record)])
    with torch.no_grad():
        out = tiny_model(**batch)
    assert "spectra" not in out.modality_losses
    assert torch.isfinite(out.loss)


def test_loss_matches_manual_computation(tiny_model, sequencer, collator, full_record):
    """outputs.loss must equal the Huber losses recomputed from predictions.

    Scalar-free build: the scalar spans' GMM NLL terms are covered by
    test_scalar_modalities' manual-loss check.
    """
    import torch.nn.functional as F

    batch = collator([sequencer.build(full_record, include_scalars=False)])
    with torch.no_grad():
        out = tiny_model(**batch)
    manual = []
    for m in ("images", "spectra"):
        manual.append(
            F.huber_loss(
                out.predictions[m],
                batch["modality_values"][m],
                delta=tiny_model.config.huber_delta,
            )
        )
    expected = torch.stack(manual).mean()
    assert torch.allclose(out.loss, expected, atol=1e-6)

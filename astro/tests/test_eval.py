"""Eval hooks (Phase 4): fixed-batch val loss and the ridge redshift probe.

CPU-sized: a random tiny model, few batches/objects. The GPU suite
(test_phase4_gpu.py) exercises the same code over real converted nanotron
checkpoints and asserts learning across steps.
"""

import math

import numpy as np
import torch

from astropt3.data.nanotron_loader import PackedMicroBatches
from astropt3.eval import linear_probe, val_loss


def test_val_loss_deterministic_and_finite(tiny_model):
    kwargs = dict(n_batches=2, micro_batch_size=2, seq_len=896)
    first = val_loss.evaluate(tiny_model, "synthetic", **kwargs)
    second = val_loss.evaluate(tiny_model, "synthetic", **kwargs)
    assert first == second
    assert math.isfinite(first["loss"]) and first["n_batches"] == 2
    assert set(first["modality_losses"]) >= {"images"}


def test_val_batches_are_held_out(tiny_config):
    # val stream indices start at the offset — disjoint from training indices
    batches = list(
        val_loss.val_batches(
            tiny_config, "synthetic", n_batches=1, micro_batch_size=2, seq_len=896
        )
    )
    assert len(batches) == 1
    train_first = next(iter(PackedMicroBatches(tiny_config, 2, 896)))
    assert not torch.equal(batches[0]["input_ids"], train_first["input_ids"]) or not torch.equal(
        batches[0]["modality_values"]["images"], train_first["images_values"]
    )


def test_probe_objects_carry_target(tiny_config):
    objects, targets = linear_probe.collect_probe_objects(tiny_config, "synthetic", "Z", 8)
    assert len(objects) == 8 and targets.shape == (8,)
    assert np.isfinite(targets).all()
    # Z only exists on spectra-bearing records
    assert all("spectra" in o.masks for o in objects)


def test_embeddings_align_with_objects(tiny_model, tiny_config):
    objects, targets = linear_probe.collect_probe_objects(tiny_config, "synthetic", "Z", 6)
    X = linear_probe.embed_objects(tiny_model, tiny_config, objects, seq_len=896, objects_per_batch=4)
    assert X.shape == (6, tiny_config.hidden_size)
    assert np.isfinite(X).all()
    # packing must not change per-object embeddings: singleton batches agree
    X_single = linear_probe.embed_objects(tiny_model, tiny_config, objects, seq_len=896, objects_per_batch=1)
    assert np.allclose(X, X_single, atol=1e-4)


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 16))
    w = rng.normal(size=16)
    y = X @ w + 0.05 * rng.normal(size=400)
    result = linear_probe.ridge_r2(X, y, seed=0)
    assert result["r2"] > 0.95


def test_ridge_no_signal_r2_near_zero():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 16))
    y = rng.normal(size=400)
    result = linear_probe.ridge_r2(X, y, seed=0)
    assert result["r2"] < 0.2

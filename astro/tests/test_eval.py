"""Eval hooks (Phase 4): fixed-batch val loss and the ridge redshift probe.

CPU-sized: a random tiny model, few batches/objects. The GPU suite
(test_phase4_gpu.py) exercises the same code over real converted nanotron
checkpoints and asserts learning across steps.
"""

import math

import numpy as np
import pytest
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


def test_probe_skips_records_without_pool_modality(tiny_config, tmp_path):
    # spectrum-only rows (ADR 0005) carry Z but have no image tokens to pool
    from astropt3.data import mmu
    from astropt3.data.synthetic import make_record

    records = []
    for i in range(6):
        records.append(make_record(i, image_only_fraction=0.0, spectrum_only_fraction=1.0))
        records.append(make_record(i + 100, image_only_fraction=0.0))
    mmu.write_shard(records, tmp_path / "shard-00000.parquet")

    objects, targets = linear_probe.collect_probe_objects(
        tiny_config, str(tmp_path), "Z", 6, pool_modality="images"
    )
    assert all("images" in o.masks for o in objects)
    # with a spectra pool the spectrum-only rows qualify: all 12 carry Z
    objects, _ = linear_probe.collect_probe_objects(
        tiny_config, str(tmp_path), "Z", 12, pool_modality="spectra"
    )
    assert len(objects) == 12


def test_probe_uses_all_objects_when_stream_exhausted(tiny_config, tmp_path):
    # a val split smaller than n_objects degrades to all qualifying records
    from astropt3.data import mmu
    from astropt3.data.synthetic import make_record

    records = [make_record(i, image_only_fraction=0.0) for i in range(6)]
    mmu.write_shard(records, tmp_path / "shard-00000.parquet")

    with pytest.warns(UserWarning, match=r"6/2048"):
        objects, targets = linear_probe.collect_probe_objects(
            tiny_config, str(tmp_path), "Z", 2048
        )
    assert len(objects) == 6 and targets.shape == (6,)

    # but zero qualifying records is still an error
    with pytest.raises(ValueError, match="no records carry target"):
        linear_probe.collect_probe_objects(tiny_config, str(tmp_path), "NOT_A_TARGET", 8)


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


def _sweep_module():
    """run_probe_sweep.py is a script, not a package module: load it by path."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "run_probe_sweep.py"
    spec = importlib.util.spec_from_file_location("run_probe_sweep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_steps_to_eval_defaults_to_every_completed_step():
    sweep = _sweep_module()
    completed = [1, 2, 4, 512, 1000, 2000]
    assert sweep.steps_to_eval(completed, done=set()) == completed
    # already-evaluated steps are never revisited (the JSONL is authoritative)
    assert sweep.steps_to_eval(completed, done={1, 2, 1000}) == [4, 512, 2000]


def test_eval_every_drops_the_pythia_powers_of_two():
    sweep = _sweep_module()
    # the schedule WRITES 1,2,4,...,512 + every 1000; --eval-every 1000 keeps
    # only the multiples, unlike should_checkpoint(step, 1000)
    completed = [1, 2, 4, 8, 128, 512, 1000, 2000, 3000]
    assert sweep.steps_to_eval(completed, done=set(), eval_every=1000) == [1000, 2000, 3000]


def test_samples_cadence_gates_the_whole_step():
    sweep = _sweep_module()
    # --samples-every/--samples-floor suppress val loss + probe too, not just
    # panels: keep should_checkpoint multiples at or above the floor
    completed = [1, 2, 4, 8, 128, 512, 1000, 1500, 2000, 3000]
    assert sweep.steps_to_eval(
        completed, done=set(), samples_every=1000, samples_floor=1000
    ) == [1000, 2000, 3000]
    # floor alone keeps the interval multiples above it
    assert sweep.steps_to_eval(completed, done=set(), samples_floor=500) == [
        512,
        1000,
        1500,
        2000,
        3000,
    ]


def test_until_step_bounds_the_todo_list():
    sweep = _sweep_module()
    completed = [1, 2, 1000, 2000, 20000, 21000]
    assert sweep.steps_to_eval(completed, done=set(), until_step=20000)[-1] == 20000
    assert sweep.steps_to_eval(completed, done=set(), until_step=20000, eval_every=1000) == [
        1000,
        2000,
        20000,
    ]

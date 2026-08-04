"""CPU tests for the nanotron micro-batch adapter (no nanotron import).

The adapter's output contract is what the nanotron fork's
``AstroPT3ForTraining.forward(**micro_batch)`` consumes; here the same flat
dicts are regrouped and fed to the HF model, which shares the packing/loss
semantics.
"""

from itertools import islice
from pathlib import Path

import pytest
import torch

from astropt3.data import nanotron_loader, streaming
from astropt3.data.nanotron_loader import (
    PackedMicroBatches,
    regroup_micro_batch as regroup,
)
from astropt3.tokenization import BOS_ID, modality_token_ids
from fake_mmu import fake_open_stream

MBS = 2
SEQ_LEN = 896


@pytest.fixture(scope="module")
def micro_batches(tiny_config):
    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN)
    return list(islice(iter(stream), 3))


def test_micro_batch_contract(tiny_config, micro_batches):
    registry = tiny_config.modality_registry()
    for flat in micro_batches:
        assert flat["input_ids"].shape == (MBS, SEQ_LEN)
        assert flat["position_ids"].shape == (MBS, SEQ_LEN)
        assert flat["input_ids"].dtype == torch.long
        for name in registry.names():
            mod = registry.get_config(name)
            mask = flat[f"{name}_mask"]
            values = flat[f"{name}_values"]
            positions = flat[f"{name}_positions"]
            assert mask.shape == (MBS, SEQ_LEN) and mask.dtype == torch.bool
            assert values.shape == (int(mask.sum()), mod.input_size)
            assert values.dtype == torch.float32
            assert len(positions) == len(values)
            # placeholder ids sit exactly at the mask positions
            _, placeholder_id, _ = modality_token_ids(name)
            assert (flat["input_ids"][mask] == placeholder_id).all()
            # <|bos|> leads every object, so no modality token at position 0
            assert not mask[:, 0].any()
        # each row starts a fresh object: position_ids restart at 0
        assert (flat["position_ids"][:, 0] == 0).all()
        assert (flat["input_ids"][:, 0] == BOS_ID).all()


def test_batches_feed_hf_model(tiny_config, tiny_model, micro_batches):
    names = tiny_config.modality_registry().names()
    for flat in micro_batches:
        out = tiny_model(**regroup(flat, names))
        assert torch.isfinite(out.loss)


def test_absent_modality_ships_typed_empty_tensors(tiny_config, tiny_model):
    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, synthetic_image_only_fraction=1.0
    )
    flat = next(iter(stream))
    assert not flat["spectra_mask"].any()
    assert flat["spectra_values"].shape == (0, 256)
    assert flat["spectra_values"].dtype == torch.float32
    assert flat["spectra_positions"].shape == (0, 1)  # continuous positions
    assert flat["spectra_positions"].dtype == torch.float32
    # Z rides with the spectrum (ADR 0008): image-only records ship it empty
    assert flat["Z_values"].shape == (0, 1)
    assert flat["Z_positions"].dtype == torch.long
    out = tiny_model(**regroup(flat, tiny_config.modality_registry().names()))
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"images", "ebv", "photometry"}


def test_synthetic_stream_disjoint_across_ranks_and_workers(tiny_config):
    # rank/worker sharding strides over record indices
    ds_a = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, rank=0, world_size=2)
    ds_b = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, rank=1, world_size=2)
    a = [r["object_id"] for r in islice(ds_a._synthetic_records(0, None), 20)]
    b = [r["object_id"] for r in islice(ds_b._synthetic_records(0, None), 20)]
    assert not set(a) & set(b)
    assert len(set(a)) == 20


def test_deterministic_across_instances(tiny_config):
    first = next(iter(PackedMicroBatches(tiny_config, MBS, SEQ_LEN)))
    second = next(iter(PackedMicroBatches(tiny_config, MBS, SEQ_LEN)))
    for key in first:
        assert torch.equal(first[key], second[key]), key


def test_mmu_stream_loops_epochs(tiny_config, monkeypatch):
    # the fake sources hold 24 records each: pulling many batches must cross
    # an epoch boundary without exhausting the endless stream
    monkeypatch.setattr("astropt3.data.streaming.open_stream", fake_open_stream)
    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, data_root="mmu", match_index="present"
    )
    batches = list(islice(iter(stream), 8))
    assert len(batches) == 8
    for flat in batches:
        assert flat["input_ids"].shape == (MBS, SEQ_LEN)


def test_more_workers_than_partitions_raises_a_named_error(tiny_config, monkeypatch):
    # datasets only WARNS and stops the surplus workers, so an over-subscribed
    # run trains on a fraction of its loaders. The fake corpus has 3 train
    # cells, so 4 workers must fail loudly with the remedy in the message.
    monkeypatch.setattr("astropt3.data.streaming.open_stream", fake_open_stream)
    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, data_root="mmu", match_index="present"
    )
    loader = torch.utils.data.DataLoader(stream, batch_size=None, num_workers=4)
    with pytest.raises(ValueError, match="reduce num_loading_workers"):
        next(iter(loader))


def test_run_configs_fit_the_crossmatch_partition_ceiling():
    """dp x num_loading_workers must fit the ~165-cell crossmatch corpus."""
    import yaml

    # 173 cells in the published index minus streaming.VAL_PARTITIONS. Offline
    # constant on purpose; recompute with load_match_index() after a rebuild.
    ceiling = 173 - streaming.VAL_PARTITIONS
    configs = sorted((Path(__file__).parents[1] / "configs" / "nanotron").glob("*.yaml"))
    assert configs, "no nanotron run configs found"
    for path in configs:
        config = yaml.safe_load(path.read_text())
        stage = config["data_stages"][0]["data"]
        if stage["dataset"].get("data_root") != "mmu":
            continue
        demand = config["parallelism"]["dp"] * stage["num_loading_workers"]
        assert demand <= ceiling, (
            f"{path.name}: dp({config['parallelism']['dp']}) x "
            f"num_loading_workers({stage['num_loading_workers']}) = {demand} "
            f"exceeds the {ceiling} train partitions"
        )


class _FlakyStream:
    """Wraps a real fake stream to raise a 'client has been closed' RuntimeError
    once mid-iteration (the DNS-blip signature), delegating state_dict so the
    loader can resume the rebuilt stream from the pre-error snapshot."""

    def __init__(self, inner, fail_at):
        self._inner = inner
        self._fail_at = fail_at

    def __iter__(self):
        for i, rec in enumerate(self._inner):
            if self._fail_at is not None and i == self._fail_at:
                self._fail_at = None
                raise RuntimeError("client has been closed")
            yield rec

    def state_dict(self):
        return self._inner.state_dict()

    def load_state_dict(self, s):
        self._inner.load_state_dict(s)


def test_transient_error_rebuilds_and_reclaims(tiny_config, monkeypatch):
    # A DNS blip surfaces as this RuntimeError; the loader must ride it out by
    # rebuilding the stream AND reclaiming the abandoned one (gc.collect), or its
    # datasets/pyarrow prefetch buffers leak per rebuild to the cgroup OOM.
    builds = {"n": 0}

    def flaky(**kw):
        builds["n"] += 1
        return _FlakyStream(
            fake_open_stream(**kw), fail_at=5 if builds["n"] == 1 else None
        )

    collects = {"n": 0}
    real_collect = nanotron_loader.gc.collect

    def spy_collect(*a, **k):
        collects["n"] += 1
        return real_collect(*a, **k)

    monkeypatch.setattr("astropt3.data.streaming.open_stream", flaky)
    monkeypatch.setattr(nanotron_loader.time, "sleep", lambda *_: None)
    monkeypatch.setattr(nanotron_loader.gc, "collect", spy_collect)

    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, data_root="mmu", match_index="present"
    )
    batches = list(islice(iter(stream), 4))

    assert builds["n"] >= 2, "the stream was never rebuilt — error path not taken"
    assert collects["n"] >= 1, "rebuild did not reclaim the abandoned stream"
    assert len(batches) == 4  # recovered and kept producing valid batches
    for flat in batches:
        assert flat["input_ids"].shape == (MBS, SEQ_LEN)

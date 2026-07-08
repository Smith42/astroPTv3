"""CPU tests for the nanotron micro-batch adapter (no nanotron import).

The adapter's output contract is what the nanotron fork's
``AstroPT3ForTraining.forward(**micro_batch)`` consumes; here the same flat
dicts are regrouped and fed to the HF model, which shares the packing/loss
semantics.
"""

from itertools import islice

import pytest
import torch

from astropt3.data import mmu
from astropt3.data.nanotron_loader import (
    PackedMicroBatches,
    flatten_packed_batch,
    regroup_micro_batch as regroup,
)
from astropt3.data.synthetic import record_stream
from astropt3.tokenization import BOS_ID, modality_token_ids

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
    out = tiny_model(**regroup(flat, tiny_config.modality_registry().names()))
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"images"}


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


def test_mmu_stream_loops_epochs(tiny_config, tmp_path):
    # 8 objects/shard x 2 shards: pulling many batches must cross an epoch
    # boundary without exhausting the stream
    records = list(record_stream(16))
    mmu.write_shard(records[:8], tmp_path / "shard-00000.parquet")
    mmu.write_shard(records[8:], tmp_path / "shard-00001.parquet")
    stream = PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, data_root=str(tmp_path), shuffle_buffer_size=4
    )
    batches = list(islice(iter(stream), 8))  # 8 batches x >=2 objects/row > 16 records
    assert len(batches) == 8
    for flat in batches:
        assert flat["input_ids"].shape == (MBS, SEQ_LEN)

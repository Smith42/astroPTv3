"""Checkpoint-resume of the nanotron micro-batch stream (Phase 4).

The contract under test: ``state_dict()`` taken after consuming k
micro-batches, loaded into a FRESH ``PackedMicroBatches``, reproduces exactly
the micro-batches an uninterrupted stream would have produced next — no
sample replay, no gap — for both the synthetic and the MMU-parquet record
sources. The object-id log is the audit trail: resumed log lines must
continue the uninterrupted log with no duplicates.
"""

from itertools import islice

import pytest
import torch

from astropt3.data import mmu
from astropt3.data.nanotron_loader import PackedMicroBatches
from astropt3.data.synthetic import record_stream

MBS = 2
SEQ_LEN = 896
N_BEFORE = 3  # micro-batches consumed before the checkpoint
N_AFTER = 4  # micro-batches compared after resume


@pytest.fixture(scope="module")
def small_shard_dir(tmp_path_factory):
    # few records so N_AFTER batches cross an epoch boundary after resume
    out = tmp_path_factory.mktemp("resume_shards")
    records = list(record_stream(48))
    for k in range(0, 48, 12):
        mmu.write_shard(records[k : k + 12], out / f"shard-{k:05d}.parquet")
    return out


def flat_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def make_stream(tiny_config, data_root, **kwargs):
    return PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, data_root=str(data_root), **kwargs
    )


@pytest.mark.parametrize("source", ["synthetic", "mmu"])
def test_resume_continues_stream_exactly(source, tiny_config, small_shard_dir, tmp_path):
    root = "synthetic" if source == "synthetic" else small_shard_dir
    log_a = tmp_path / f"{source}_a.log"

    ds_a = make_stream(tiny_config, root, object_id_log=log_a)
    it_a = iter(ds_a)
    consumed = list(islice(it_a, N_BEFORE))
    assert len(consumed) == N_BEFORE
    state = ds_a.state_dict()
    reference = list(islice(it_a, N_AFTER))  # uninterrupted continuation

    log_b = tmp_path / f"{source}_b.log"
    ds_b = make_stream(tiny_config, root, object_id_log=log_b)
    ds_b.load_state_dict(state)
    resumed = list(islice(iter(ds_b), N_AFTER))

    for i, (ref, res) in enumerate(zip(reference, resumed)):
        assert flat_equal(ref, res), f"micro-batch {i} diverged after resume"

    # audit trail: the resumed log is exactly the uninterrupted log's tail
    lines_a = log_a.with_name(log_a.name + ".dp0").read_text().splitlines()
    lines_b = log_b.with_name(log_b.name + ".dp0").read_text().splitlines()
    assert lines_a[-len(lines_b) :] == lines_b
    trained_before = lines_a[: len(lines_a) - len(lines_b)]
    assert not set(trained_before) & set(lines_b), "resume replayed trained objects"


def test_state_is_row_start_not_consumption_point(tiny_config):
    # state after k batches must rewind to the current partial row's first
    # record: a fresh stream fast-forwarded with it reproduces batch k+1
    ds = make_stream(tiny_config, "synthetic")
    it = iter(ds)
    next(it)
    state = ds.state_dict()
    # the packer has drawn more records than the checkpoint position exposes
    # (the partial row + the overflow record are re-drawn on resume)
    assert state["records"] > 0

    ds2 = make_stream(tiny_config, "synthetic")
    ds2.load_state_dict(state)
    assert flat_equal(next(iter(ds2)), next(it))


def test_state_dict_none_when_not_stateful(tiny_config):
    ds = make_stream(tiny_config, "synthetic", stateful=False)
    next(iter(ds))
    assert ds.state_dict() is None


def test_load_rejects_wrong_data_root(tiny_config, small_shard_dir):
    ds = make_stream(tiny_config, "synthetic")
    with pytest.raises(ValueError, match="data_root"):
        ds.load_state_dict({"records": 0, "epoch": 0, "hf_state": None, "data_root": str(small_shard_dir)})


def test_mmu_resume_across_epoch_boundary(tiny_config, small_shard_dir):
    # ~4-5 objects per micro-batch and 48 records per epoch: checkpoint at 9
    # batches and compare 4 more, so the continuation wraps into epoch 1
    ds = make_stream(tiny_config, small_shard_dir)
    it = iter(ds)
    consumed = list(islice(it, 9))
    assert len(consumed) == 9, "expected the small corpus to still yield batches"
    state = ds.state_dict()
    reference = list(islice(it, 4))

    ds2 = make_stream(tiny_config, small_shard_dir)
    ds2.load_state_dict(state)
    resumed = list(islice(iter(ds2), 4))
    for ref, res in zip(reference, resumed):
        assert flat_equal(ref, res)
    assert ds2._epoch >= 1, "continuation never crossed an epoch boundary"

"""Checkpoint-resume of the nanotron micro-batch stream (Phase 4).

The contract under test: ``state_dict()`` taken after consuming k
micro-batches, loaded into a FRESH ``PackedMicroBatches``, reproduces exactly
the micro-batches an uninterrupted stream would have produced next — no
sample replay, no gap — for both the synthetic and the live-MMU record
sources. The object-id log is the audit trail: resumed log lines must
continue the uninterrupted log with no duplicates.

ADR 0006 budgeted for replaying the in-flight partition on resume; because
each source buffers its current partition whole and checkpoints the row
offset into it, the original no-replay guarantee survives streaming — hence
these tests are unchanged in strictness.

The same contract must hold end-to-end through
``build_astropt3_dataloader`` + ``loader_state_dict`` at any
``num_workers`` — with workers the state lives in the worker processes and
rides torchdata's StatefulDataLoader.
"""

from itertools import islice
from types import SimpleNamespace

import pytest
import torch

from astropt3.data.nanotron_loader import (
    STATE_FILE_TEMPLATE,
    STATE_SUBDIR,
    LOADER_STATE_FORMAT,
    PackedMicroBatches,
    build_astropt3_dataloader,
    loader_state_dict,
)
from fake_mmu import fake_open_stream

MBS = 2
# two whole objects per row (objects are 180/147 tokens post-crop), so the
# object-per-batch counts in the epoch-boundary comments below stay true
SEQ_LEN = 384
N_BEFORE = 3  # micro-batches consumed before the checkpoint
N_AFTER = 4  # micro-batches compared after resume


@pytest.fixture(autouse=True)
def stub_mmu(monkeypatch):
    """Every ``data_root="mmu"`` stream draws from the offline fake."""
    monkeypatch.setattr("astropt3.data.streaming.open_stream", fake_open_stream)


def flat_equal(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def make_stream(tiny_config, data_root, **kwargs):
    return PackedMicroBatches(
        tiny_config, MBS, SEQ_LEN, data_root=str(data_root), **kwargs
    )


@pytest.mark.parametrize("source", ["synthetic", "mmu"])
def test_resume_continues_stream_exactly(source, tiny_config, tmp_path):
    log_a = tmp_path / f"{source}_a.log"

    ds_a = make_stream(tiny_config, source, object_id_log=log_a)
    it_a = iter(ds_a)
    consumed = list(islice(it_a, N_BEFORE))
    assert len(consumed) == N_BEFORE
    state = ds_a.state_dict()
    reference = list(islice(it_a, N_AFTER))  # uninterrupted continuation

    log_b = tmp_path / f"{source}_b.log"
    ds_b = make_stream(tiny_config, source, object_id_log=log_b)
    ds_b.load_state_dict(state)
    resumed = list(islice(iter(ds_b), N_AFTER))

    for i, (ref, res) in enumerate(zip(reference, resumed)):
        assert flat_equal(ref, res), f"micro-batch {i} diverged after resume"

    # audit trail: the resumed log is exactly the uninterrupted log's tail —
    # this is the exact-resume guarantee. (The old per-object "no replay"
    # disjointness check is gone: interleave_datasets(all_exhausted)
    # oversamples the smaller sources by design, so an object legitimately
    # recurs within an epoch — the tail-continuation below is the real
    # invariant, not object uniqueness.)
    lines_a = log_a.with_name(log_a.name + ".dp0").read_text().splitlines()
    lines_b = log_b.with_name(log_b.name + ".dp0").read_text().splitlines()
    assert lines_a[-len(lines_b) :] == lines_b


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


def test_load_rejects_wrong_data_root(tiny_config):
    ds = make_stream(tiny_config, "synthetic")
    with pytest.raises(ValueError, match="data_root"):
        ds.load_state_dict(
            {"records": 0, "epoch": 0, "stream_state": None, "data_root": "mmu"}
        )


def test_rejects_a_stale_local_corpus_path(tiny_config):
    # ADR 0006 deleted the local reshard; a config still pointing at it must
    # fail loudly rather than silently stream something else
    with pytest.raises(ValueError, match="streamed live"):
        make_stream(tiny_config, "../astroPTv3_data/pilot_v2/train")


def build_loader(tiny_config, root, num_workers, resume_state_dir=None):
    dataset_args = SimpleNamespace(
        data_root=str(root),
        synthetic_image_only_fraction=0.3,
        object_id_log=None,
    )
    return build_astropt3_dataloader(
        dataset_args=dataset_args,
        model_config=tiny_config,
        micro_batch_size=MBS,
        sequence_length=SEQ_LEN,
        dp_rank=0,
        dp_size=1,
        num_workers=num_workers,
        resume_state_dir=resume_state_dir,
    )


def save_state(state, tmp_path):
    state_dir = tmp_path / STATE_SUBDIR
    state_dir.mkdir(exist_ok=True)
    torch.save(state, state_dir / STATE_FILE_TEMPLATE.format(rank=0))
    return state_dir


@pytest.mark.parametrize("source", ["synthetic", "mmu"])
@pytest.mark.parametrize("num_workers", [0, 2])
def test_stateful_loader_resume(source, num_workers, tiny_config, tmp_path):
    loader_a = build_loader(tiny_config, source, num_workers)
    it_a = iter(loader_a)
    consumed = list(islice(it_a, N_BEFORE))
    assert len(consumed) == N_BEFORE
    state = loader_state_dict(loader_a)
    assert state["format"] == LOADER_STATE_FORMAT
    assert state["num_workers"] == num_workers
    reference = list(islice(it_a, N_AFTER))  # uninterrupted continuation

    state_dir = save_state(state, tmp_path)
    loader_b = build_loader(tiny_config, source, num_workers, resume_state_dir=state_dir)
    resumed = list(islice(iter(loader_b), N_AFTER))
    for i, (ref, res) in enumerate(zip(reference, resumed)):
        assert flat_equal(ref, res), f"micro-batch {i} diverged after resume"


def test_loader_state_rejects_num_workers_mismatch(tiny_config, tmp_path):
    loader = build_loader(tiny_config, "synthetic", 2)
    next(iter(loader))
    state_dir = save_state(loader_state_dict(loader), tmp_path)
    with pytest.raises(ValueError, match="num_loading_workers"):
        build_loader(tiny_config, "synthetic", 0, resume_state_dir=state_dir)


def test_legacy_dataset_state_requires_zero_workers(tiny_config, tmp_path):
    ds = make_stream(tiny_config, "synthetic")
    it = iter(ds)
    next(it)
    state_dir = save_state(ds.state_dict(), tmp_path)
    with pytest.raises(ValueError, match="num_loading_workers == 0"):
        build_loader(tiny_config, "synthetic", 2, resume_state_dir=state_dir)
    # at workers == 0 a legacy (dataset-format) file still resumes exactly
    loader = build_loader(tiny_config, "synthetic", 0, resume_state_dir=state_dir)
    assert flat_equal(next(iter(loader)), next(it))


def test_mmu_resume_across_epoch_boundary(tiny_config):
    # the fake corpus is one finite epoch of 48 records; the loader reopens the
    # stream at epoch+1, so consuming many micro-batches rolls past a boundary.
    # Resume must reproduce the exact micro-batches across that reopen.
    ds = make_stream(tiny_config, "mmu")
    it = iter(ds)
    consumed = list(islice(it, 20))  # > one epoch of packed rows
    assert len(consumed) == 20, "endless (epoch-looping) stream must keep yielding"
    assert ds._epoch >= 1, "should have crossed at least one epoch boundary"
    state = ds.state_dict()
    reference = list(islice(it, 4))

    ds2 = make_stream(tiny_config, "mmu")
    ds2.load_state_dict(state)
    resumed = list(islice(iter(ds2), 4))
    for ref, res in zip(reference, resumed):
        assert flat_equal(ref, res)


def test_mmu_stream_realizes_the_source_weights(tiny_config, tmp_path):
    """Images dominate spectra in the packed stream (no match index, so the
    two-source 0.8/0.2 mix; ids < 1000 are images, >= 1000 are spectra)."""
    log = tmp_path / "mix.log"
    ds = make_stream(tiny_config, "mmu", object_id_log=log)
    list(islice(iter(ds), 40))
    ids = log.with_name(log.name + ".dp0").read_text().splitlines()
    is_image = [int(i.rsplit("_", 1)[1]) < 1000 for i in ids]
    assert sum(is_image) > len(ids) - sum(is_image), "images should dominate spectra"

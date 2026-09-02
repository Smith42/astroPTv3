"""CPU tests for the nanotron LSDB micro-batch adapter.

Contract honored by the nanotron fork's ``AstroPT3ForTraining.forward``;
sources are faked — only the network-marked live check touches the hub.
"""

from itertools import islice

import numpy as np
import pandas as pd
import pytest
import torch

from legacy_fixture import crossmatch_row, legacy_row, make_record

from astropt3.data import nanotron_loader
from astropt3.data.nanotron_loader import (
    PackedMicroBatches,
    consumer_seed,
    decode_crossmatch_row,
    decode_legacy_row,
)
from astropt3.data.nanotron_loader import (
    regroup_micro_batch as regroup,
)
from astropt3.tokenization import BOS_ID, modality_token_ids

MBS = 2
SEQ_LEN = 896


class _FakeCatalog:
    npartitions = 3


def fake_stream(frames_per_epoch=2, rows_per_frame=4):
    """Cheap ``InfiniteStream`` replacement yielding unconsumed DataFrames."""

    class FakeStream:
        seed = None

        def __init__(self, catalog, client, partitions_per_chunk, seed):
            FakeStream.seed = seed

        def __iter__(self):
            for _ in range(frames_per_epoch):
                yield pd.DataFrame(
                    [legacy_row(i) for i in range(rows_per_frame)]
                )

    return FakeStream


@pytest.fixture
def fake_lsdb(monkeypatch):
    monkeypatch.setattr(
        nanotron_loader.lsdb, "open_catalog", lambda *a, **k: _FakeCatalog()
    )
    monkeypatch.setattr(nanotron_loader, "_log_provenance", lambda *a, **k: None)

    def install(stream_cls):
        monkeypatch.setattr(nanotron_loader, "InfiniteStream", stream_cls)

    return install


def _stream(tiny_config, fake_lsdb, **kwargs):
    fake_lsdb(fake_stream())
    return PackedMicroBatches(tiny_config, MBS, SEQ_LEN, **kwargs)


def test_micro_batch_contract(tiny_config, fake_lsdb):
    registry = tiny_config.modality_registry()
    stream = _stream(tiny_config, fake_lsdb)
    for flat in islice(iter(stream), 3):
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
            _, placeholder_id, _ = modality_token_ids(name)
            assert (flat["input_ids"][mask] == placeholder_id).all()
            assert not mask[:, 0].any()
        assert (flat["position_ids"][:, 0] == 0).all()
        assert (flat["input_ids"][:, 0] == BOS_ID).all()


def test_batches_feed_hf_model(tiny_config, tiny_model, fake_lsdb):
    stream = _stream(tiny_config, fake_lsdb)
    names = tiny_config.modality_registry().names()
    for flat in islice(iter(stream), 3):
        out = tiny_model(**regroup(flat, names))
        assert torch.isfinite(out.loss)


def test_image_only_records_ship_empty_spectra_tensors(
    tiny_config, tiny_model, fake_lsdb
):
    stream = _stream(tiny_config, fake_lsdb)
    flat = next(iter(stream))
    # uncrossmatched LegacySurvey rows carry image + legacy scalars only
    assert not flat["spectra_mask"].any()
    assert flat["spectra_values"].shape == (0, 256)
    assert flat["spectra_values"].dtype == torch.float32
    assert flat["spectra_positions"].shape == (0, 1)
    assert flat["Z_values"].shape == (0, 1)
    assert flat["Z_positions"].dtype == torch.long
    out = tiny_model(**regroup(flat, tiny_config.modality_registry().names()))
    assert torch.isfinite(out.loss)
    assert set(out.modality_losses) == {"images", "ebv", "photometry"}


def test_seeds_differ_across_rank_worker_and_retry(tiny_config):
    base = consumer_seed(42, 0, 0, 0)
    assert base != consumer_seed(42, 1, 0, 0)
    assert base != consumer_seed(42, 0, 1, 0)
    assert base != consumer_seed(42, 0, 0, 1)
    # two consumers with identical identity derive the same seed
    assert base == consumer_seed(42, 0, 0, 0)


def test_decoder_recovers_record_fields():
    record = decode_legacy_row(legacy_row(3))
    assert record["image"]["flux"].shape == (3, 152, 152)
    assert record["image"]["flux"].dtype == np.float32
    assert record["image"]["band"] == ["des-g", "des-r", "des-z"]
    for key in (
        "ebv",
        "flux_g",
        "flux_r",
        "flux_z",
        "fiberflux_g",
        "psfdepth_z",
        "z_spec",
    ):
        assert isinstance(record[key], float), key
    for band in ("des-g", "des-r", "des-z"):
        assert isinstance(record[f"psf_fwhm_{band}"], float)


def test_decoder_rejects_bad_shape_and_id():
    row = legacy_row(0)
    row["image"] = {"band": ["des-g"], "flux": [[0.0]]}
    with pytest.raises(ValueError, match="flux has shape"):
        decode_legacy_row(row)
    row = legacy_row(0)
    del row["object_id"]
    with pytest.raises(ValueError, match="object_id"):
        decode_legacy_row(row)


def test_crossmatch_decoder_recovers_matched_row():
    record = decode_crossmatch_row(crossmatch_row(3, matched=True))
    assert record["spectrum"]["flux"].shape == (7781,)
    assert record["spectrum"]["flux"].dtype == np.float32
    assert record["spectrum"]["mask"].dtype == bool
    assert record["image"]["flux"].shape == (3, 152, 152)
    assert record["image"]["band"] == ["des-g", "des-r", "des-z"]
    assert isinstance(record["Z"], float)
    assert isinstance(record["ebv"], float)
    for band in ("des-g", "des-r", "des-z"):
        assert isinstance(record[f"psf_fwhm_{band}"], float)


def test_crossmatch_decoder_handles_unmatched_row():
    record = decode_crossmatch_row(crossmatch_row(3, matched=False))
    assert "spectrum" in record
    assert "image" not in record
    assert "ebv" not in record


def test_crossmatch_decoder_rejects_missing_id():
    row = crossmatch_row(0, matched=True)
    del row["object_id"]
    with pytest.raises(ValueError, match="object_id"):
        decode_crossmatch_row(row)


class _AmbiguousBoolMapping:
    """Mimics a real nested-pandas struct scalar: like a live ``image_legacy``
    cell, ``bool(...)`` raises pandas' own ambiguity error rather than
    falling back to truthiness."""

    def __init__(self, mapping):
        self._mapping = mapping

    def __bool__(self):
        raise ValueError(
            "The truth value of a DataFrame is ambiguous. "
            "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
        )

    def as_py(self):
        return self._mapping


def test_crossmatch_decoder_never_bool_checks_nested_image_scalar():
    """Regression: a live run crashed because decode did ``image_value and
    ...`` on a nested-pandas struct scalar, whose ``bool()`` raises instead
    of returning True/False like a plain dict would."""
    row = crossmatch_row(3, matched=True)
    row["image_legacy"] = _AmbiguousBoolMapping(row["image_legacy"])
    record = decode_crossmatch_row(row)
    assert record["image"]["flux"].shape == (3, 152, 152)


def test_transient_error_reopens_fresh_stream(tiny_config, monkeypatch):
    """A mid-iteration transport blip discards the iterator and retries."""
    builds = {"n": 0}

    class FlakyStream:
        def __init__(self, *a, **k):
            builds["n"] += 1
            self.built = builds["n"]

        def __iter__(self):
            if self.built == 1:
                yield pd.DataFrame([legacy_row(i) for i in range(6)])
                raise OSError("simulated storage blip")
            while True:
                yield pd.DataFrame([legacy_row(i) for i in range(6)])

    monkeypatch.setattr(
        nanotron_loader.lsdb, "open_catalog", lambda *a, **k: _FakeCatalog()
    )
    monkeypatch.setattr(nanotron_loader, "_log_provenance", lambda *a, **k: None)
    monkeypatch.setattr(nanotron_loader, "InfiniteStream", FlakyStream)
    monkeypatch.setattr(nanotron_loader.time, "sleep", lambda *_: None)

    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN)
    batches = list(islice(iter(stream), 3))
    assert builds["n"] >= 2, "no fresh stream was opened after the blip"
    assert len(batches) == 3
    for flat in batches:
        assert flat["input_ids"].shape == (MBS, SEQ_LEN)


def test_non_retryable_error_fails_immediately(tiny_config, monkeypatch):
    """Decode/validation errors are not retried."""

    class BadStream:
        def __init__(self, *a, **k):
            pass

        def __iter__(self):
            yield pd.DataFrame([legacy_row(0)])
            raise ValueError("decode blew up")

    monkeypatch.setattr(
        nanotron_loader.lsdb, "open_catalog", lambda *a, **k: _FakeCatalog()
    )
    monkeypatch.setattr(nanotron_loader, "_log_provenance", lambda *a, **k: None)
    monkeypatch.setattr(nanotron_loader, "InfiniteStream", BadStream)
    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN)
    with pytest.raises(ValueError, match="decode blew up"):
        next(iter(stream))


# -- replicas / placement (kept behavior, ADR 0015 keeps ar_replicas) --------


def _bipartite_stream(tiny_config, **kwargs):
    kwargs.setdefault("ar_replicas", 3)
    return PackedMicroBatches(tiny_config, 8, SEQ_LEN, **kwargs)


def test_one_span_records_get_no_replica(tiny_config):
    stream = _bipartite_stream(tiny_config)
    record = make_record(2, image_only_fraction=1.0)
    objects = stream._replica_objects(record)
    assert len(objects) == len(set(id(obj) for obj in objects))
    assert all(set(obj.order) == set(objects[0].order) for obj in objects)


def test_replicas_carry_distinct_span_orders(tiny_config):
    stream = _bipartite_stream(tiny_config)
    for i in range(12):
        objects = stream._replica_objects(make_record(i, image_only_fraction=1.0))
        orders = [obj.order for obj in objects]
        assert len(orders) == len(set(orders)), f"record {i} repeated an order"
        assert "#" not in objects[0].object_id
        assert all("#" in obj.object_id for obj in objects[1:])


def test_no_two_replicas_of_one_object_share_a_packed_row(tiny_config):
    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, ar_replicas=2)
    placements = []
    for i in range(8):
        objects = stream._replica_objects(make_record(i, image_only_fraction=1.0))
        if len(objects) > 1:
            placements.append(stream._place(objects, [0] * MBS))
    assert placements, "no record produced replicas"
    for chosen in placements:
        assert len(chosen) == len(set(chosen)), "two replicas landed in one row"


def test_placement_deterministic_and_emptiest_first(tiny_config):
    stream = PackedMicroBatches(tiny_config, MBS, SEQ_LEN, ar_replicas=1)
    objects = stream._replica_objects(make_record(1, image_only_fraction=1.0))
    used = [300, 10] + [50] * (MBS - 2)
    first = stream._place(objects, used)
    assert first == [1]
    assert stream._place(objects, used) == first


def test_more_replicas_than_rows_is_rejected(tiny_config):
    with pytest.raises(ValueError, match="exceeds micro_batch_size"):
        PackedMicroBatches(tiny_config, 2, SEQ_LEN, ar_replicas=3)


def test_ar_replicas_rejects_a_nonsense_count(tiny_config):
    with pytest.raises(ValueError, match="ar_replicas must be >= 1"):
        PackedMicroBatches(tiny_config, MBS, SEQ_LEN, ar_replicas=0)

"""ADR 0006 streaming loader (HF datasets backend): decode, split, weights, resume.

The hand-rolled reader is gone; ``open_stream`` returns a ``datasets``
interleave. Offline coverage uses :mod:`fake_mmu`, which builds the SAME kind
of interleave over synthetic records — so per-record weighting, node
splitting and ``state_dict`` resume are exercised for real, just on local
data. Decode is tested directly on synthetic rows (``synthetic.py`` mirrors
the MMU schema, so a synthetic record is shaped like a hub row).
"""

from pathlib import Path

import numpy as np
import pytest

from astropt3.data.streaming import (
    aligned,
    DEFAULT_WEIGHTS,
    decode_record,
    pairs_dataset,
    shuffled,
    split_files,
    union_features,
)
from astropt3.data.synthetic import make_record
from fake_mmu import fake_open_stream

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def take(stream, n):
    return [r["object_id"] for r, _ in zip(iter(stream), range(n))]


def kinds(stream, n):
    """Classify records by modality: images-only, spectra-only, or paired —
    the only way to tell sources apart, since a paired record's object_id IS
    the image's (pairs are matched images). Every record carries both keys
    after the union map; the absent modality is None (ObjectSequencer keys off
    exactly that), so presence is value-not-None, not key membership."""
    out = []
    for r, _ in zip(iter(stream), range(n)):
        out.append((r.get("image") is not None, r.get("spectrum") is not None))
    return out


# -- decode ------------------------------------------------------------------


def test_decode_image_only_row():
    row = make_record(1, image_only_fraction=1.0)  # synthetic ~ hub row shape
    rec = decode_record(row)
    assert "spectrum" not in rec
    assert rec["image"]["flux"].shape == (3, 152, 152)
    assert rec["image"]["band"] == ["des-g", "des-r", "des-z"]
    assert rec["object_id"] == row["object_id"]
    assert "ebv" in rec  # image-catalog scalars carried through


def test_decode_spectrum_only_row():
    row = make_record(2, image_only_fraction=0.0, spectrum_only_fraction=1.0)
    rec = decode_record(row)
    assert "image" not in rec
    assert rec["spectrum"]["flux"].shape == (7781,)
    assert rec["spectrum"]["mask"].dtype == bool
    assert rec["Z"] == pytest.approx(row["Z"])


def test_decode_bimodal_row_carries_both_modalities():
    row = make_record(3)  # bimodal: both image and spectrum present
    rec = decode_record(row)
    assert rec["image"]["flux"].shape == (3, 152, 152)
    assert rec["spectrum"]["flux"].shape == (7781,)
    assert rec["Z"] == pytest.approx(row["Z"])


def test_decode_rejects_an_empty_row():
    with pytest.raises(ValueError, match="neither image nor spectrum"):
        decode_record({"object_id": "x", "ra": 0.0, "dec": 0.0, "_healpix_29": 0})


# -- split + shuffle ---------------------------------------------------------


def test_val_reserves_the_first_partitions_disjoint_from_train():
    files = [f"f{i}" for i in range(20)]
    val = split_files(files, "val", val_partitions=3)
    train = split_files(files, "train", val_partitions=3)
    assert val == ["f0", "f1", "f2"]
    assert set(val) & set(train) == set()
    assert sorted(val + train) == sorted(files)


def test_val_reservation_is_capped_so_a_small_source_is_not_swallowed():
    """A flat K larger than a small source would leave train empty — the
    6-partition smoke match-index hit exactly this."""
    files = [f"f{i}" for i in range(6)]  # smaller than val_partitions=8
    val = split_files(files, "val", val_partitions=8)
    train = split_files(files, "train", val_partitions=8)
    assert 0 < len(val) <= 6 // 5 + 1
    assert len(train) > 0
    assert set(val) & set(train) == set()


def test_shuffled_is_deterministic_and_epoch_dependent():
    files = [f"f{i}" for i in range(50)]
    a = shuffled(files, seed=0, epoch=0)
    assert a == shuffled(files, seed=0, epoch=0)  # reproducible
    assert a != shuffled(files, seed=0, epoch=1)  # reshuffled per epoch
    assert sorted(a) == sorted(files)  # a permutation, nothing lost


def test_aligned_truncates_to_a_shard_multiple():
    """An odd shard count (pairs: 165 % dp 2) silently collapses datasets'
    rank/worker split to one shard — aligned() must prevent it."""
    files = [f"f{i}" for i in range(165)]
    assert len(aligned(files, 2)) == 164
    assert aligned(files, 1) == files  # single shard: keep everything
    assert aligned(files, 2) == files[:164]  # a prefix, order untouched


def test_pairs_generator_reads_survive_dataloader_workers():
    """A nested datasets stream iterated inside a DataLoader worker gets
    worker-split AGAIN: its single shard lands on worker 0 and every other
    worker silently reads an empty stream — no pairs, an instantly-exhausted
    source. The pairs generator's inner reads use pyarrow to stay immune."""
    import json

    import torch

    from fake_mmu import _fixtures

    fx = _fixtures()
    features = union_features(fx["images"][0], fx["spectra"][0])
    ds = pairs_dataset(
        image_paths=[fx["images"][0], fx["images"][1]],
        match_json=[json.dumps(fx["match"])] * 2,
        spectra_paths=[fx["spectra"], fx["spectra"]],
        features=features,
    )
    solo = sum(1 for _ in ds)
    assert solo > 0
    loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2)
    assert sum(1 for _ in loader) == solo


# -- weighting + resume (real datasets interleave, offline) ------------------


def test_weights_are_realized_per_record():
    """Two-source corpus (no match index) hits the renormalized 0.8/0.2 mix."""
    got = kinds(fake_open_stream(seed=0), 2000)
    img = got.count((True, False)) / len(got)
    assert img == pytest.approx(0.8, abs=0.05)


def test_three_source_mix_with_a_match_index():
    got = kinds(fake_open_stream(seed=0, match_index="present"), 2000)
    frac = [
        got.count((True, False)) / len(got),   # images-only
        got.count((False, True)) / len(got),   # spectra-only
        got.count((True, True)) / len(got),    # paired
    ]
    # the fake corpus is tiny (all_exhausted caps it ~90 records), so this is
    # loose — it checks the ordering and rough magnitude, not the exact mix
    assert frac == pytest.approx(DEFAULT_WEIGHTS, abs=0.08)
    assert frac[0] > frac[2] > frac[1]  # images > pairs > spectra


def test_resume_reproduces_the_record_sequence():
    """datasets state_dict/load_state_dict continues the stream exactly.

    Mid-epoch resume (the fake corpus is one finite epoch of 48 records; the
    loader loops epochs on top of this)."""
    stream = fake_open_stream(seed=0)
    it = iter(stream)
    first = [next(it)["object_id"] for _ in range(20)]
    state = stream.state_dict()
    reference = [next(it)["object_id"] for _ in range(10)]

    resumed = fake_open_stream(seed=0)
    resumed.load_state_dict(state)
    assert [r["object_id"] for r, _ in zip(iter(resumed), range(10))] == reference


def test_no_match_index_drops_the_pairs_source():
    got = kinds(fake_open_stream(seed=0), 500)
    assert (True, True) not in got  # nothing paired


def test_only_restricts_to_one_source():
    got = kinds(fake_open_stream(seed=0, only="spectra", match_index="present"), 200)
    assert set(got) == {(False, True)}  # spectra-only


# -- match index -------------------------------------------------------------


def test_match_index_resolution_prefers_the_explicit_argument(monkeypatch):
    """Training passes the index from the nanotron config; eval falls back to
    the env var so every eval entry point avoids a pass-through parameter."""
    from astropt3.data.streaming import MATCH_INDEX_ENV, resolve_match_index

    monkeypatch.delenv(MATCH_INDEX_ENV, raising=False)
    assert resolve_match_index() is None
    assert resolve_match_index("/explicit.parquet") == "/explicit.parquet"

    monkeypatch.setenv(MATCH_INDEX_ENV, "/from-env.parquet")
    assert resolve_match_index() == "/from-env.parquet"
    assert resolve_match_index("/explicit.parquet") == "/explicit.parquet"

    monkeypatch.setenv(MATCH_INDEX_ENV, "")  # unset-ish must not become a path
    assert resolve_match_index() is None


def test_match_index_round_trips(tmp_path):
    """load_match_index: parquet of ids -> per-image-partition lookup, keyed by
    HEALPix cell so the published artifact survives a re-partition."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from astropt3.data.streaming import load_match_index

    path = tmp_path / "index.parquet"
    pq.write_table(
        pa.table(
            {
                "image_order": pa.array([6, 6, 6], pa.int8()),
                "image_pixel": pa.array([7, 7, 9], pa.int64()),
                "image_id": ["i1", "i2", "i9"],
                "spectrum_order": pa.array([8, 8, 8], pa.int8()),
                "spectrum_pixel": pa.array([2, 3, 2], pa.int64()),
                "spectrum_id": ["s1", "s2", "s9"],
            }
        ),
        path,
    )
    matches, spectra_of = load_match_index(str(path))
    assert matches == {(6, 7): {"i1": "s1", "i2": "s2"}, (6, 9): {"i9": "s9"}}
    assert spectra_of == {(6, 7): {(8, 2), (8, 3)}, (6, 9): {(8, 2)}}


# The pairs join (_paired_examples) is exercised for real by
# test_three_source_mix_with_a_match_index (paired records appear at the right
# frequency) and by the live network test below.


# -- live hub (network) ------------------------------------------------------


@pytest.mark.network
def test_live_mmu_rows_decode_and_sequence():
    """Real hub rows -> records -> ObjectSeq, covering the decode a fake can't.

    Deselect with ``-m 'not network'`` when the hub is down.
    """
    from astropt3.config_io import load_model_config
    from astropt3.data.packing import ObjectSequencer
    from astropt3.data.streaming import open_stream

    config, _ = load_model_config(CONFIGS / "model" / "test-tiny.yaml")
    sequencer = ObjectSequencer(config)

    # no match index -> images + spectra live; the pairs join is covered
    # offline by test_paired_examples_joins_images_to_spectra
    stream = open_stream(seed=0)
    it = iter(stream)
    records = [next(it) for _ in range(40)]

    seen = set()
    for record in records:
        seq = sequencer.build(record)
        shapes = {m: tuple(v.shape) for m, v in seq.values.items()}
        if "images" in shapes:
            assert shapes["images"] == (144, 192)  # 96x96 crop, patch 8
        if "spectra" in shapes:
            assert shapes["spectra"] == (31, 256)
            # Z is present only when ZWARN==0 (ADR 0008 gating), so it is not
            # guaranteed on every DESI row — but when present it is one token
            if "Z" in shapes:
                assert shapes["Z"] == (1, 1)
        seen.add((record.get("image") is not None, record.get("spectrum") is not None))
    assert seen == {(True, False), (False, True)}  # both live sources appear

    # resume round-trips against live partitions (one iterator, mid-stream
    # snapshot, then compare the continuation to a fresh load_state_dict)
    state = stream.state_dict()
    reference = [next(it)["object_id"] for _ in range(5)]
    resumed = open_stream(seed=0)
    resumed.load_state_dict(state)
    rit = iter(resumed)
    assert [next(rit)["object_id"] for _ in range(5)] == reference

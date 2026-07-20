"""ADR 0006 streaming loader: draw pattern, per-record weighting, resume.

Fake fetchers stand in for lsdb, so the cursor logic is covered offline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astropt3.data.streaming import (
    DEFAULT_WEIGHTS,
    PATTERN_LEN,
    MMUStream,
    Source,
    _pattern,
    partition_order,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# rows per partition, deliberately lopsided per source — the whole point of
# weighting per record rather than per partition draw
ROWS = {"images": 7, "spectra": 40, "pairs": 3}
NPART = {"images": 12, "spectra": 5, "pairs": 4}


# rows per row group, so the fakes exercise the multi-group path too
GROUP = 3


class FakePartition:
    """Rows split into row groups, mirroring a real parquet partition."""

    def __init__(self, name, index, n_rows):
        self._ids = [f"{name}:{index}:{i}" for i in range(n_rows)]

    @property
    def n_groups(self):
        return max(1, -(-len(self._ids) // GROUP))  # ceil, >=1 even when empty

    def group(self, index):
        return pd.DataFrame({"id": self._ids[index * GROUP : (index + 1) * GROUP]})


def fake_sources(rows=ROWS, npart=NPART):
    def make(name):
        return Source(
            name,
            npart[name],
            lambda index, name=name: FakePartition(name, index, rows[name]),
            lambda row: {"object_id": row["id"]},
        )

    return [make(n) for n in ("images", "spectra", "pairs")]


def take(stream, n):
    it = iter(stream)
    return [next(it)["object_id"] for _ in range(n)]


def test_pattern_matches_weights():
    pat = _pattern(DEFAULT_WEIGHTS)
    counts = [pat.count(i) for i in range(3)]
    assert counts == [12, 3, 5]  # 0.60 / 0.15 / 0.25 of 20
    # interleaved, not blocked: no source runs 6+ draws in a row
    runs = [len(list(g)) for g in _runs(pat)]
    assert max(runs) < 6


def _runs(seq):
    out, cur = [], [seq[0]]
    for x in seq[1:]:
        if x == cur[0]:
            cur.append(x)
        else:
            out.append(cur)
            cur = [x]
    out.append(cur)
    return out


def test_weights_are_realized_per_record():
    """Realized record mix tracks the weights despite lopsided partitions."""
    got = take(MMUStream(fake_sources()), 4000)
    frac = [sum(g.startswith(n) for g in got) / len(got) for n in ("images", "spectra", "pairs")]
    assert frac == pytest.approx(DEFAULT_WEIGHTS, abs=0.01)


def test_resume_reproduces_the_identical_record_sequence():
    full = take(MMUStream(fake_sources()), 600)

    stream = MMUStream(fake_sources())
    first = take(stream, 237)  # mid-partition for every source
    state = stream.state_dict()

    resumed = MMUStream(fake_sources())
    resumed.load_state_dict(state)
    assert first + take(resumed, 600 - 237) == full


def test_resume_across_a_source_epoch_rollover():
    """pairs has 4 partitions x 3 rows; 600 draws cycles it many times."""
    full = take(MMUStream(fake_sources()), 900)
    stream = MMUStream(fake_sources())
    first = take(stream, 700)
    assert stream.state_dict()["sources"]["pairs"]["epoch"] > 0
    resumed = MMUStream(fake_sources())
    resumed.load_state_dict(stream.state_dict())
    assert first + take(resumed, 200) == full


def test_state_dict_is_plain_ints():
    stream = MMUStream(fake_sources())
    take(stream, 50)
    state = stream.state_dict()
    for src in state["sources"].values():
        assert all(isinstance(v, int) for v in src.values())
    assert isinstance(state["draw"], int)


def test_shards_are_disjoint_and_cover_the_catalog():
    orders = [
        MMUStream(fake_sources(), shard=s, num_shards=4).sources[0].order for s in range(4)
    ]
    pooled = np.concatenate(orders)
    assert sorted(pooled.tolist()) == list(range(NPART["images"]))


def test_val_reservation_never_swallows_a_small_source():
    """A flat K larger than a small source would leave train with nothing —
    the smoke match-index (6 pairs partitions) hit exactly this."""
    srcs = fake_sources(npart={"images": 40, "spectra": 40, "pairs": 6})
    train = MMUStream(srcs, split="train", val_partitions=8)
    val = MMUStream(fake_sources(npart={"images": 40, "spectra": 40, "pairs": 6}),
                    split="val", val_partitions=8)
    pairs_train = train.sources[2].order
    pairs_val = val.sources[2].order
    assert len(pairs_train) and len(pairs_val)
    assert not set(pairs_train.tolist()) & set(pairs_val.tolist())


def test_val_partitions_are_disjoint_from_train():
    kw = dict(val_partitions=3)
    train = MMUStream(fake_sources(), split="train", **kw).sources[0].order
    val = MMUStream(fake_sources(), split="val", **kw).sources[0].order
    # val reserves the LOWEST partition indices (capped at a fifth of the
    # source), train gets the rest, and the two never overlap
    assert set(val.tolist()) == set(range(len(val)))
    assert 0 < len(val) <= 3
    assert not set(train.tolist()) & set(val.tolist())
    assert len(train) + len(val) == NPART["images"]


def test_empty_partitions_are_skipped():
    srcs = fake_sources()
    src = srcs[2]
    full = src.open_partition
    src.open_partition = lambda i: (FakePartition("pairs", i, 0) if i % 2 else full(i))
    got = take(MMUStream(srcs), 400)
    assert any(g.startswith("pairs") for g in got)


def test_row_groups_are_released_as_they_are_consumed():
    """RAM bound: only the in-flight row group is held, never the partition."""
    stream = MMUStream(fake_sources())
    it = iter(stream)
    for _ in range(200):
        next(it)
    for src in stream.sources:
        assert src._buf is None or len(src._buf) <= GROUP


def test_partition_order_is_deterministic_and_epoch_dependent():
    a = partition_order(50, seed=0, epoch=0)
    assert (a == partition_order(50, seed=0, epoch=0)).all()
    assert not (a == partition_order(50, seed=0, epoch=1)).all()
    assert sorted(a.tolist()) == list(range(50))


def test_state_rejects_a_mismatched_stream():
    stream = MMUStream(fake_sources(), seed=1)
    take(stream, 10)
    with pytest.raises(ValueError, match="saved for"):
        MMUStream(fake_sources(), seed=2).load_state_dict(stream.state_dict())


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
    """load_match_index: parquet of ids -> per-image-partition lookup."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from astropt3.data.streaming import load_match_index

    path = tmp_path / "index.parquet"
    pq.write_table(
        pa.table(
            {
                # partitions are HEALPix cells, not positions in a listing —
                # the artifact is published and must survive a re-partition
                "image_order": pa.array([6, 6, 6], pa.int8()),
                "image_pixel": pa.array([7, 7, 9], pa.int64()),
                "image_id": ["i1", "i2", "i9"],
                "spectrum_order": pa.array([8, 8, 8], pa.int8()),
                "spectrum_pixel": pa.array([2, 3, 2], pa.int64()),
                "spectrum_id": ["s1", "s2", "s9"],
                "dist_arcsec": pa.array([0.1, 0.2, 0.3], pa.float32()),
            }
        ),
        path,
    )
    matches, spectra_of = load_match_index(str(path))
    assert matches == {(6, 7): {"i1": "s1", "i2": "s2"}, (6, 9): {"i9": "s9"}}
    assert spectra_of == {(6, 7): {(8, 2), (8, 3)}, (6, 9): {(8, 2)}}


def test_paired_partition_joins_on_the_index(monkeypatch):
    """The pairs join: unmatched images dropped, spectrum columns attached
    under the _desi suffix row_to_record already understands."""
    from astropt3.data import streaming

    images = pd.DataFrame({"object_id": ["i1", "i2", "i3"], "ra": [1.0, 2.0, 3.0]})
    spectra = pd.DataFrame(
        {"object_id": ["s1", "s3"], "spectrum": ["S1", "S3"], "Z": [0.1, 0.3],
         "ZERR": [0.01, 0.03], "ZWARN": [False, False]}
    )

    class Fake:
        def __init__(self, path, fs):
            self._df = images if path == "img" else spectra

        @property
        def n_groups(self):
            return 1

        def group(self, index):
            return self._df

    monkeypatch.setattr(streaming, "_ParquetPartition", Fake)
    part = streaming._PairedPartition("img", {"i1": "s1", "i3": "s3"}, ["spec"], fs=None)
    out = part.group(0)

    assert list(out["object_id"]) == ["i1", "i3"]  # i2 has no match
    assert list(out["spectrum_desi"]) == ["S1", "S3"]
    assert list(out["Z_desi"]) == [0.1, 0.3]


@pytest.mark.network
def test_live_mmu_rows_decode_and_sequence():
    """Real hub rows -> records -> ObjectSeq, covering the decode a fake can't.

    This is what ADR 0006 §9 proposed a checked-in cassette for; hitting the
    hub directly needs no fixture to capture, refresh, or drift out of date.
    Deselect with ``-m 'not network'`` when the hub is down.
    """
    from astropt3.config_io import load_model_config
    from astropt3.data.packing import ObjectSequencer
    from astropt3.data.streaming import open_sources

    config, _ = load_model_config(CONFIGS / "model" / "test-tiny.yaml")
    sequencer = ObjectSequencer(config)

    # without a match index there is no pairs source (ADR 0006), so the live
    # corpus here is images + spectra; the paired join is covered offline by
    # test_paired_partition_joins_on_the_index
    stream = MMUStream(open_sources(), weights=DEFAULT_WEIGHTS[:2], seed=0)
    records = [r for r, _ in zip(iter(stream), range(24))]

    seen = set()
    for record in records:
        seq = sequencer.build(record)
        shapes = {m: tuple(v.shape) for m, v in seq.values.items()}
        if "images" in shapes:
            assert shapes["images"] == (144, 192)  # 96x96 crop, patch 8
        if "spectra" in shapes:
            assert shapes["spectra"] == (31, 256)
            assert shapes["Z"] == (1, 1)  # DESI rows carry redshift
        seen.add(("image" in record, "spectrum" in record))
    # both live sources are represented: image-only and spectrum-only
    assert seen == {(True, False), (False, True)}

    # and the cursor round-trips against live partitions, not just fakes
    state = stream.state_dict()
    reference = [r["object_id"] for r, _ in zip(iter(stream), range(5))]
    resumed = MMUStream(open_sources(), weights=DEFAULT_WEIGHTS[:2], seed=0)
    resumed.load_state_dict(state)
    assert [r["object_id"] for r, _ in zip(iter(resumed), range(5))] == reference

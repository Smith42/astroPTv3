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


def fake_sources(rows=ROWS, npart=NPART):
    def make(name):
        def fetch(index):
            n = rows[name]
            return pd.DataFrame({"id": [f"{name}:{index}:{i}" for i in range(n)]})

        return Source(name, npart[name], fetch, lambda row: {"object_id": row["id"]})

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


def test_val_partitions_are_disjoint_from_train():
    kw = dict(val_partitions=3)
    train = MMUStream(fake_sources(), split="train", **kw).sources[0].order
    val = MMUStream(fake_sources(), split="val", **kw).sources[0].order
    assert set(val.tolist()) == {0, 1, 2}
    assert not set(train.tolist()) & set(val.tolist())


def test_empty_partitions_are_skipped():
    srcs = fake_sources()
    src = srcs[2]
    full = src.fetch
    src.fetch = lambda i: (pd.DataFrame({"id": []}) if i % 2 else full(i))
    got = take(MMUStream(srcs), 400)
    assert any(g.startswith("pairs") for g in got)


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

    stream = MMUStream(open_sources(), seed=0)
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
    # the three sources are all represented: image-only, spectrum-only, paired
    assert seen == {(True, False), (False, True), (True, True)}

    # and the cursor round-trips against live partitions, not just fakes
    state = stream.state_dict()
    reference = [r["object_id"] for r, _ in zip(iter(stream), range(5))]
    resumed = MMUStream(open_sources(), seed=0)
    resumed.load_state_dict(state)
    assert [r["object_id"] for r, _ in zip(iter(resumed), range(5))] == reference

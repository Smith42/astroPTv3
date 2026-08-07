"""Crossmatch-only MMU stream: decode, ownership, splitting, and resume."""

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from astropt3.data.streaming import (
    _SOURCE_COLUMNS,
    _SPECTRUM_LEAVES,
    _partition_owner,
    _source_graph_examples,
    _spectrum_part,
    attach_source,
    decode_record,
    owned_by_rank,
    shuffled,
    source_only_record,
    split_files,
    split_of_cell,
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


def test_spectrum_leaf_projection_drops_ivar_and_decodes_identically(tmp_path):
    """ADR 0014 §6: projecting the leaves must not change a decoded record.

    ``ivar`` is 41% of a spectrum row's bytes and is read nowhere, so the
    projected read is pure saving — provided the surviving leaves decode
    byte-for-byte as they did from the whole struct.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    row = make_record(7, image_only_fraction=0.0, spectrum_only_fraction=1.0)
    spectrum = row["spectrum"]
    table = pa.Table.from_pylist(
        [
            {
                "object_id": "s1",
                "ra": 1.0,
                "dec": 2.0,
                "_healpix_29": 0,
                "spectrum": {
                    "flux": np.asarray(spectrum["flux"]).tolist(),
                    "lambda": np.asarray(spectrum["lambda"]).tolist(),
                    "ivar": np.asarray(spectrum["ivar"]).tolist(),
                    "mask": np.asarray(spectrum["mask"]).tolist(),
                },
                "Z": 0.5,
                "ZERR": 0.01,
                "ZWARN": 0,
            }
        ]
    )
    path = tmp_path / "desi.parquet"
    pq.write_table(table, path)

    parquet = pq.ParquetFile(path)
    whole = parquet.read_row_group(0).slice(0, 1).to_pylist()[0]
    projected = (
        parquet.read_row_group(0, columns=_SOURCE_COLUMNS["desi"])
        .slice(0, 1)
        .to_pylist()[0]
    )

    # the projection really did leave ivar on the wire
    assert "ivar" in whole["spectrum"]
    assert set(projected["spectrum"]) == set(_SPECTRUM_LEAVES)

    from_whole = _spectrum_part(whole)
    from_projected = _spectrum_part(projected)
    assert set(from_whole) == set(_SPECTRUM_LEAVES)  # whitelist, not passthrough
    for leaf in _SPECTRUM_LEAVES:
        assert from_whole[leaf].dtype == from_projected[leaf].dtype
        assert np.array_equal(from_whole[leaf], from_projected[leaf])
        assert from_whole[leaf].tobytes() == from_projected[leaf].tobytes()

    # the scalar columns the adapters need survive the projection too
    for key in ("object_id", "ra", "dec", "_healpix_29", "Z", "ZERR", "ZWARN"):
        assert projected[key] == whole[key]


def test_spectrum_part_rejects_a_row_missing_a_projected_leaf():
    with pytest.raises(ValueError, match="missing"):
        _spectrum_part({"spectrum": {"flux": [1.0], "lambda": [4000.0]}})


def test_source_adapters_keep_sdss_and_hsc_distinct():
    anchor = decode_record(make_record(1, image_only_fraction=1.0))
    spectrum_row = make_record(2, image_only_fraction=0.0, spectrum_only_fraction=1.0)
    spectrum_row.update({"Z_ERR": 0.01, "ZWARNING": 0})
    attach_source(anchor, "sdss", spectrum_row)
    assert list(anchor["sdss_spectrum"]["flux"].shape) == [7781]
    assert anchor["sdss_Z_ERR"] == 0.01
    assert anchor["sdss_ZWARNING"] is False
    assert "spectrum" not in anchor

    hsc_row = make_record(3, image_only_fraction=1.0)
    hsc_row["image"] = {
        "flux": [[[0.0] * 160 for _ in range(160)] for _ in range(5)],
        "band": ["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"],
    }
    standalone = source_only_record("hsc", hsc_row)
    assert standalone["object_id"].startswith("hsc:")
    assert list(standalone["hsc_image"]["flux"].shape) == [5, 160, 160]
    assert "image" not in standalone


def test_source_graph_assembles_all_spokes_and_emits_unmatched(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    def write(name, rows):
        path = tmp_path / f"{name}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        return str(path)

    def image(bands, side):
        return {
            "flux": np.zeros((len(bands), side, side), dtype=np.float32).tolist(),
            "band": bands,
        }

    spectrum = {
        "flux": [1.0, 2.0],
        "lambda": [4000.0, 4001.0],
        "ivar": [1.0, 1.0],
        "mask": [False, False],
    }
    anchors = write(
        "anchors",
        [
            {
                "object_id": object_id,
                "ra": 1.0,
                "dec": 2.0,
                "_healpix_29": i,
                "image": image(["des-g", "des-r", "des-z"], 152),
            }
            for i, object_id in enumerate(("a1", "a2"))
        ],
    )
    rows = {
        "desi": [
            {
                "object_id": value,
                "ra": 1.0,
                "dec": 2.0,
                "_healpix_29": i,
                "spectrum": spectrum,
            }
            for i, value in enumerate(("d1", "d2"))
        ],
        "sdss": [
            {
                "object_id": value.encode(),
                "ra": 1.0,
                "dec": 2.0,
                "_healpix_29": i,
                "spectrum": spectrum,
            }
            for i, value in enumerate(("s1", "s2"))
        ],
        "hsc": [
            {
                "object_id": value,
                "ra": 1.0,
                "dec": 2.0,
                "_healpix_29": i,
                "image": image(["hsc-g", "hsc-r", "hsc-i", "hsc-z", "hsc-y"], 160),
            }
            for i, value in enumerate(("h1", "h2"))
        ],
        "galaxies": [
            {
                "dr8_id": value,
                "ra": 1.0,
                "dec": 2.0,
                "_healpix_29": i,
                "smooth-or-featured_smooth_fraction": 0.7,
            }
            for i, value in enumerate(("g1", "g2"))
        ],
        "provabgs": [
            {
                "object_id": str(value),
                "ra": 1.0,
                "dec": 2.0,
                "_healpix_29": i,
                "LOG_MSTAR": 10.0,
                "TSNR2_BGS": 100.0,
            }
            for i, value in enumerate((11, 12))
        ],
    }
    paths = {source: write(source, source_rows) for source, source_rows in rows.items()}
    matches = {
        "a1": {
            "desi": "d1",
            "galaxies": "g1",
            "hsc": "h1",
            "provabgs": "11",
            "sdss": "s1",
        }
    }
    partition_specs = {
        source: [{"path": path, "order": 6, "pixel": i}]
        for i, (source, path) in enumerate(paths.items())
    }
    records = list(
        _source_graph_examples(
            [anchors],
            [json.dumps(matches)],
            [json.dumps(partition_specs)],
            [{source: [path] for source, path in paths.items()}],
            {source: [partner_id] for source, partner_id in matches["a1"].items()},
        )
    )
    paired = next(record for record in records if record["object_id"] == "a1")
    assert {"image", "spectrum", "sdss_spectrum", "hsc_image"} <= paired.keys()
    assert paired["gwh_smooth-or-featured_smooth_fraction"] == 0.7
    assert paired["provabgs_LOG_MSTAR"] == 10.0
    assert {
        record["object_id"] for record in records if ":" in record["object_id"]
    } == {
        "desi:d2",
        "hsc:h2",
        "sdss:s2",
    }


def test_source_graph_warns_and_skips_missing_partner(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    image = {
        "flux": np.zeros((3, 152, 152), dtype=np.float32).tolist(),
        "band": ["des-g", "des-r", "des-z"],
    }
    spectrum = {
        "flux": [1.0, 2.0],
        "lambda": [4000.0, 4001.0],
        "ivar": [1.0, 1.0],
        "mask": [False, False],
    }
    anchors = tmp_path / "anchors.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "object_id": "a1",
                    "ra": 1.0,
                    "dec": 2.0,
                    "_healpix_29": 0,
                    "image": image,
                    "ebv": 0.1,
                    "flux_g": 1.0,
                    "flux_r": 1.0,
                    "flux_z": 1.0,
                    "z_spec": 0.2,
                }
            ]
        ),
        anchors,
    )
    sdss = tmp_path / "sdss.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "object_id": "present",
                    "ra": 1.0,
                    "dec": 2.0,
                    "_healpix_29": 1,
                    "spectrum": spectrum,
                    "Z": 0.2,
                    "Z_ERR": 0.01,
                    "ZWARNING": 0,
                }
            ]
        ),
        sdss,
    )

    with pytest.warns(RuntimeWarning, match="missing sdss id 'missing'"):
        records = list(
            _source_graph_examples(
                [str(anchors)],
                [json.dumps({"a1": {"sdss": "missing"}})],
                [json.dumps({"sdss": [{"path": str(sdss), "order": 6, "pixel": 0}]})],
                [{"sdss": [str(sdss)]}],
                {"sdss": ["missing", "present"]},
            )
        )

    assert [record["object_id"] for record in records] == ["a1"]
    assert "sdss_spectrum" not in records[0]


# -- split + shuffle ---------------------------------------------------------


def test_common_split_uses_canonical_parent_cells():
    for pixel in range(16):
        assert split_of_cell((6, pixel << 4)) == split_of_cell((4, pixel))


def test_partner_owner_is_deterministic_and_requires_a_same_split_reference():
    references = [(6, 3), (6, 1), (6, 2)]
    assert _partition_owner("path", references, [(6, 1), (6, 2)]) in {
        (6, 1),
        (6, 2),
    }
    assert _partition_owner("path", list(reversed(references)), [(6, 1), (6, 2)]) == (
        _partition_owner("path", references, [(6, 1), (6, 2)])
    )
    assert _partition_owner("path", references, [(6, 9)]) is None


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


def test_dp_ranks_partition_the_corpus_without_dropping_any():
    """The DP deal must cover every cell exactly once at any rank count.

    split_dataset_by_node only assigns shards when the count divides evenly and
    otherwise reads everything on every rank; truncating to a multiple instead
    cost 37 of 165 cells an epoch at dp=64. Dealing the list does neither.
    """
    files = [f"f{i}" for i in range(165)]
    assert owned_by_rank(files, 0, 1) == files  # single rank: keep everything
    for dp in (2, 8, 32, 64):
        deals = [owned_by_rank(files, rank, dp) for rank in range(dp)]
        assert sorted(f for deal in deals for f in deal) == sorted(files)
        assert max(map(len, deals)) - min(map(len, deals)) <= 1


def test_crossmatch_generator_reads_survive_dataloader_workers():
    import torch

    stream = fake_open_stream()
    solo = sum(1 for _ in stream)
    assert solo > 0
    loader = torch.utils.data.DataLoader(
        cast(Any, fake_open_stream()), batch_size=None, num_workers=2
    )
    assert sum(1 for _ in loader) == solo


# -- crossmatch-only corpus --------------------------------------------------


def test_crossmatch_only_yields_all_kinds_from_one_scan():
    got = kinds(fake_open_stream(seed=0), 300)
    assert set(got) == {(True, True), (True, False), (False, True)}


def test_crossmatch_only_resume_and_ranks_are_disjoint():
    stream = fake_open_stream(seed=0)
    iterator = iter(stream)
    for _ in range(15):
        next(iterator)
    state = stream.state_dict()
    reference = [next(iterator)["object_id"] for _ in range(8)]

    resumed = fake_open_stream(seed=0)
    resumed.load_state_dict(state)
    assert [
        record["object_id"] for record, _ in zip(iter(resumed), range(8))
    ] == reference

    rank_ids = [
        {record["object_id"] for record in fake_open_stream(shard=rank, num_shards=2)}
        for rank in range(2)
    ]
    assert rank_ids[0] and rank_ids[1]
    assert rank_ids[0].isdisjoint(rank_ids[1])


def test_unmatched_buffer_budget_reorders_but_never_drops_records(monkeypatch):
    """The budget bounds memory only — the corpus must be byte-identical.

    A cell's unmatched spectra are strided into the image scan out of a
    bounded buffer; past the budget they are emitted as read. That moves
    records within a cell, so the guarantee is on the multiset, not the order.
    """
    baseline = sorted(record["object_id"] for record in fake_open_stream(seed=0))
    monkeypatch.setattr("astropt3.data.streaming.UNMATCHED_BUFFER_BYTES", 0)
    starved = [record["object_id"] for record in fake_open_stream(seed=0)]
    assert sorted(starved) == baseline
    assert len(starved) == len(set(starved)), "a record was emitted twice"


def test_spectrum_only_rows_are_disjoint_between_train_and_val():
    def spectrum_only_ids(split):
        return {
            record["object_id"]
            for record in fake_open_stream(split=split)
            if record.get("image") is None
        }

    assert spectrum_only_ids("train").isdisjoint(spectrum_only_ids("val"))


def test_crossmatch_only_requires_an_index(monkeypatch):
    from astropt3.data.streaming import MATCH_INDEX_ENV, open_stream

    monkeypatch.delenv(MATCH_INDEX_ENV, raising=False)
    with pytest.raises(ValueError, match="requires match_index"):
        open_stream()


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


# -- live hub (network) ------------------------------------------------------


@pytest.mark.network
def test_live_mmu_rows_decode_and_sequence():
    """Real hub rows -> records -> ObjectSeq, covering the decode a fake can't.

    Deselect with ``-m 'not network'`` when the hub is down.
    """
    from astropt3.config_io import load_model_config
    from astropt3.data.packing import ObjectSequencer
    from astropt3.data.streaming import open_stream, resolve_match_index

    match_index = resolve_match_index()
    if match_index is None:
        pytest.skip("live crossmatch test requires ASTROPT3_MATCH_INDEX")
    config, _ = load_model_config(CONFIGS / "model" / "test-tiny.yaml")
    sequencer = ObjectSequencer(config)
    stream = open_stream(seed=0, match_index=match_index)
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
    assert seen <= {(True, False), (False, True), (True, True)}
    assert seen

    # resume round-trips against live partitions (one iterator, mid-stream
    # snapshot, then compare the continuation to a fresh load_state_dict)
    state = stream.state_dict()
    reference = [next(it)["object_id"] for _ in range(5)]
    resumed = open_stream(seed=0, match_index=match_index)
    resumed.load_state_dict(state)
    rit = iter(resumed)
    assert [next(rit)["object_id"] for _ in range(5)] == reference

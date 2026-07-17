"""Crossmatch-row -> record conversion (offline; no lsdb needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from astropt3.data.mmu import IMAGE_SHAPE, SPECTRUM_LENGTH, normalize_record

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_pilot_data.py"
spec = importlib.util.spec_from_file_location("prepare_pilot_data", _SCRIPT)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def _image_struct_dict():
    return {
        "band": np.array(["des-g", "des-r", "des-z"], dtype=object),
        "flux": np.ones(IMAGE_SHAPE, dtype=np.float32),
        "psf_fwhm": np.array([1.2, 1.1, 1.0], dtype=np.float32),
        "scale": np.full(3, 0.262, dtype=np.float32),
    }


def _spectrum_struct_dict():
    return {
        "flux": np.linspace(1, 2, SPECTRUM_LENGTH, dtype=np.float32),
        "lambda": np.linspace(3600, 9824, SPECTRUM_LENGTH, dtype=np.float32),
        "ivar": np.ones(SPECTRUM_LENGTH, dtype=np.float32),
        "lsf_sigma": np.ones(SPECTRUM_LENGTH, dtype=np.float32),
        "mask": np.zeros(SPECTRUM_LENGTH, dtype=bool),
    }


def _matched_row():
    return pd.Series(
        {
            "object_id": "obj-1",
            "ra": 150.0,
            "dec": 2.2,
            "_healpix_29": 12345678901234,
            "image": _image_struct_dict(),
            "ebv": 0.03,
            "flux_g": 1.5,
            "flux_r": 2.5,
            "flux_z": 3.5,
            "z_spec": 0.42,
            "object_id_desi": "desi-9",
            "spectrum_desi": _spectrum_struct_dict(),
            "Z_desi": 0.43,
            "ZERR_desi": 0.001,
            "ZWARN_desi": False,
            "_dist_arcsec_desi": 0.3,
        }
    )


def test_matched_row_roundtrips_through_normalize():
    record = prepare.row_to_record(_matched_row())
    assert record["object_id"] == "obj-1"
    assert "spectrum" in record
    assert record["Z"] == pytest.approx(0.43)
    assert record["ZWARN"] is False
    assert record["match_dist_arcsec"] == pytest.approx(0.3)
    normalized = normalize_record(record)
    assert normalized["image"]["flux"].shape == IMAGE_SHAPE
    assert normalized["spectrum"]["flux"].shape == (SPECTRUM_LENGTH,)


def test_unmatched_left_row_has_no_spectrum():
    row = _matched_row()
    row["spectrum_desi"] = pd.NA  # nullable-dtype missing marker
    for key in ("Z_desi", "ZERR_desi", "ZWARN_desi", "_dist_arcsec_desi"):
        row[key] = np.float32("nan")
    record = prepare.row_to_record(row)
    assert "spectrum" not in record and "Z" not in record
    normalized = normalize_record(record)
    assert normalized["spectrum"] is None
    assert normalized["Z"] is None


def _desi_row(name, *, ra=150.0, dec=2.2, zwarn=False):
    # a plain DESI catalog row (pass 2, ADR 0005 — no crossmatch columns)
    return {
        "object_id": name,
        "ra": ra,
        "dec": dec,
        "_healpix_29": 12345678901234,
        "spectrum": _spectrum_struct_dict(),
        "Z": 0.43,
        "ZERR": 0.001,
        "ZWARN": zwarn,
    }


def test_spectra_only_records_filters_matched_and_zwarn():
    pytest.importorskip("scipy")
    arcsec = 1.0 / 3600.0
    df = pd.DataFrame(
        [
            _desi_row("keep"),
            _desi_row("matched", ra=200.0, dec=10.0),
            _desi_row("bad-zwarn", zwarn=True),
        ]
    )
    # one matched pair 0.5 arcsec from the "matched" row, far from the others
    index = prepare.position_index([200.0 + 0.5 * arcsec / np.cos(np.radians(10.0))], [10.0])
    records, n_zwarn, n_dup = prepare.spectra_only_records(df, index, radius_arcsec=1.0)
    assert [r["object_id"] for r in records] == ["keep"]
    assert n_zwarn == 1 and n_dup == 1
    record = records[0]
    assert "image" not in record and record["ZWARN"] is False
    assert record["Z"] == pytest.approx(0.43)
    normalized = normalize_record(record)
    assert normalized["image"] is None
    assert normalized["spectrum"]["flux"].shape == (SPECTRUM_LENGTH,)

    # no matched pairs at all (empty index): nothing is de-duplicated
    records, n_zwarn, n_dup = prepare.spectra_only_records(df, None, radius_arcsec=1.0)
    assert len(records) == 2 and n_dup == 0


def test_matched_positions_roundtrip(tmp_path):
    pytest.importorskip("scipy")
    from astropt3.data.mmu import write_shard
    from astropt3.data.synthetic import make_record

    matched = make_record(3)
    assert "spectrum" in matched
    matched["match_dist_arcsec"] = 0.3
    write_shard([matched, make_record(1)], tmp_path / "train" / "s0.parquet")
    ra, dec = prepare.load_matched_positions(tmp_path)
    assert len(ra) == 1 and ra[0] == pytest.approx(matched["ra"])
    # a spectrum at the matched position is dropped; 2 arcsec away is kept
    df = pd.DataFrame(
        [
            _desi_row("dup", ra=matched["ra"], dec=matched["dec"]),
            _desi_row("keep", ra=matched["ra"], dec=matched["dec"] + 2.0 / 3600.0),
        ]
    )
    records, _, n_dup = prepare.spectra_only_records(
        df, prepare.position_index(ra, dec), radius_arcsec=1.0
    )
    assert [r["object_id"] for r in records] == ["keep"] and n_dup == 1


def test_write_partition_spectra_subdir(tmp_path):
    from astropt3.data.synthetic import make_record

    records = [
        make_record(i, image_only_fraction=0.0, spectrum_only_fraction=1.0)
        for i in range(3)
    ]
    prepare.write_partition(
        {"train": records, "val": []}, tmp_path, 5, 42, shard_size=4,
        subdir=prepare.SPECTRA_SUBDIR,
    )
    assert [p.name for p in (tmp_path / "train" / "spectra").glob("*.parquet")] == [
        "part-05-0000042-000.parquet"
    ]
    prepare.clean_partition(tmp_path, 5, 42)  # top level: no-op
    assert len(list(tmp_path.glob("*/spectra/*.parquet"))) == 1
    prepare.clean_partition(tmp_path, 5, 42, subdir=prepare.SPECTRA_SUBDIR)
    assert list(tmp_path.glob("*/spectra/*.parquet")) == []


def _ragged_band(fill: float) -> np.ndarray:
    # arrow nested lists surface as object arrays of per-row float arrays
    rows = np.empty(152, dtype=object)
    for i in range(152):
        rows[i] = np.full(152, fill, dtype=np.float32)
    return rows


def test_nested_frame_struct_cells():
    # nested-pandas materializes struct columns as per-row sub-DataFrames,
    # with image flux as band -> object array of 152 row-arrays (verified
    # against a real crossmatch partition, 2026-07-07)
    row = _matched_row()
    row["image"] = pd.DataFrame(
        {
            "band": ["des-g", "des-r", "des-z"],
            "flux": [_ragged_band(float(b)) for b in range(3)],
            "psf_fwhm": np.array([1.2, 1.1, 1.0], dtype=np.float32),
            "scale": np.full(3, 0.262, dtype=np.float32),
        }
    )
    row["spectrum_desi"] = pd.DataFrame(_spectrum_struct_dict())
    record = prepare.row_to_record(row)
    normalized = normalize_record(record)
    flux = normalized["image"]["flux"]
    assert flux.shape == IMAGE_SHAPE
    assert np.all(flux[1] == 1.0)
    assert normalized["spectrum"]["flux"].shape == (SPECTRUM_LENGTH,)


def test_healpix_from_row_index():
    # lsdb result frames carry _healpix_29 as the index, not a column
    row = _matched_row().drop("_healpix_29")
    row.name = np.int64(98765)
    record = prepare.row_to_record(row)
    assert record["_healpix_29"] == 98765


def test_unsuffixed_columns_also_resolve():
    row = _matched_row().rename(
        {
            "spectrum_desi": "spectrum",
            "Z_desi": "Z",
            "ZERR_desi": "ZERR",
            "ZWARN_desi": "ZWARN",
            "_dist_arcsec_desi": "_dist_arcsec",
        }
    )
    record = prepare.row_to_record(row)
    assert "spectrum" in record and record["Z"] == pytest.approx(0.43)


def test_pixel_overlaps_nested_ancestry():
    # order-6 pixel 9145 descends from order-5 pixel 2286 (9145 >> 2) and
    # contains order-8 pixels 146320..146335 (9145 << 4 ..)
    assert prepare.pixel_overlaps((6, 9145), [(6, 9145)])
    assert prepare.pixel_overlaps((6, 9145), [(5, 2286)])  # coarser ancestor
    assert prepare.pixel_overlaps((5, 2286), [(6, 9145)])  # finer descendant
    assert prepare.pixel_overlaps((6, 9145), [(8, 146320)])
    assert not prepare.pixel_overlaps((6, 9145), [(6, 9146)])
    assert not prepare.pixel_overlaps((6, 9145), [(5, 2287)])
    assert not prepare.pixel_overlaps((6, 9145), [(8, 146336)])
    assert not prepare.pixel_overlaps((6, 9145), [])


def test_write_partition_shards_and_cleanup(tmp_path):
    from astropt3.data.synthetic import record_stream

    records = list(record_stream(10))
    prepare.write_partition(
        {"train": records[:9], "val": records[9:]}, tmp_path, 5, 42, shard_size=4
    )
    train = sorted(p.name for p in (tmp_path / "train").glob("*.parquet"))
    assert train == [
        "part-05-0000042-000.parquet",
        "part-05-0000042-001.parquet",
        "part-05-0000042-002.parquet",
    ]
    assert [p.name for p in (tmp_path / "val").glob("*.parquet")] == [
        "part-05-0000042-000.parquet"
    ]
    prepare.clean_partition(tmp_path, 5, 43)  # other partition: no-op
    assert len(list(tmp_path.glob("*/*.parquet"))) == 4
    prepare.clean_partition(tmp_path, 5, 42)
    assert list(tmp_path.glob("*/*.parquet")) == []

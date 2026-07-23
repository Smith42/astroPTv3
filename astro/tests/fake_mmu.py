"""An offline stand-in for :func:`astropt3.data.streaming.open_stream`.

Writes synthetic records to local parquet in the hub's nested schema, then
drives the real crossmatch-only generator over them. Only the partition paths
differ from production, so joins, ownership, node splitting, resume, and decode
are exercised offline.

Monkeypatch ``astropt3.data.streaming.open_stream`` with
:func:`fake_open_stream`; the loader imports it at call time.
"""

import tempfile
from pathlib import Path

import numpy as np
from datasets import Array3D, Dataset, Features, Sequence, Value

from astropt3.data.streaming import (
    _spectrum_owners,
    aligned,
    crossmatch_dataset,
    decode_record,
    shuffled,
    split_files,
    union_features,
)
from astropt3.data.synthetic import IMAGE_BANDS, make_record

_N = {"images": 24, "spectra": 24}  # per source; disjoint id bases below
_BASE = {"images": 0, "spectra": 1000}
_FILES_PER_SOURCE = 4

_IMAGE_FEATURES = Features(
    {
        "object_id": Value("string"),
        "ra": Value("float64"),
        "dec": Value("float64"),
        "_healpix_29": Value("int64"),
        "image": {
            "flux": Array3D((3, 152, 152), "float32"),
            "band": Sequence(Value("string")),
        },
        "ebv": Value("float32"),
        "flux_g": Value("float32"),
        "flux_r": Value("float32"),
        "flux_z": Value("float32"),
        "z_spec": Value("float32"),
    }
)
_SPECTRUM_FEATURES = Features(
    {
        "object_id": Value("string"),
        "ra": Value("float64"),
        "dec": Value("float64"),
        "_healpix_29": Value("int64"),
        "spectrum": {
            "flux": Sequence(Value("float32")),
            "lambda": Sequence(Value("float32")),
            "ivar": Sequence(Value("float32")),
            "lsf_sigma": Sequence(Value("float32")),
            "mask": Sequence(Value("bool")),
        },
        "Z": Value("float32"),
        "ZWARN": Value("bool"),
    }
)
_cache: dict | None = None


def _image_row(record):
    return {
        "object_id": record["object_id"],
        "ra": record["ra"],
        "dec": record["dec"],
        "_healpix_29": record["_healpix_29"],
        "image": {"flux": record["image"]["flux"], "band": IMAGE_BANDS},
        "ebv": record["ebv"],
        "flux_g": record["flux_g"],
        "flux_r": record["flux_r"],
        "flux_z": record["flux_z"],
        "z_spec": record["z_spec"],
    }


def _spectrum_row(record):
    return {
        "object_id": record["object_id"],
        "ra": record["ra"],
        "dec": record["dec"],
        "_healpix_29": record["_healpix_29"],
        "spectrum": {k: np.asarray(v) for k, v in record["spectrum"].items()},
        "Z": record["Z"],
        "ZWARN": record["ZWARN"],
    }


def _write(root, name):
    if name == "images":
        rows = [
            _image_row(make_record(_BASE[name] + i, image_only_fraction=1.0))
            for i in range(_N[name])
        ]
        feats = _IMAGE_FEATURES
    else:
        rows = [
            _spectrum_row(
                make_record(
                    _BASE[name] + i, image_only_fraction=0.0, spectrum_only_fraction=1.0
                )
            )
            for i in range(_N[name])
        ]
        feats = _SPECTRUM_FEATURES
    files = []
    for shard in range(_FILES_PER_SOURCE):
        part = rows[shard::_FILES_PER_SOURCE]
        path = root / f"{name}_{shard}.parquet"
        Dataset.from_list(part, features=feats).to_parquet(str(path))
        files.append(str(path))
    return files, [r["object_id"] for r in rows]


def _fixtures():
    global _cache
    if _cache is None:
        root = Path(tempfile.mkdtemp(prefix="fake_mmu_"))
        images, image_ids = _write(root, "images")
        spectra, spectrum_ids = _write(root, "spectra")
        _cache = {
            "images": images,
            "spectra": spectra,
            # an arbitrary id-to-id match; the fake join needs no spatial truth
            "match": dict(zip(image_ids[:8], spectrum_ids[:8])),
        }
    return _cache


def fixed_records(
    *, split="train", seed=0, epoch=0, shard=0, num_shards=1, match_index=None
):
    """A deterministic, guaranteed mix of record kinds — for eval-logic tests
    (probe, template selection) that need specific shapes present regardless of
    weighting. Cycles image-only / spectrum-only / bimodal; not the datasets
    machinery, just an iterable of records (which is all a consumer needs)."""
    recs = []
    for i in range(10):
        recs.append(make_record(i, image_only_fraction=1.0))  # image-only, no Z
        recs.append(
            make_record(1000 + i, image_only_fraction=0.0, spectrum_only_fraction=1.0)
        )
        recs.append(
            make_record(2000 + i, image_only_fraction=0.0, spectrum_only_fraction=0.0)
        )
    return iter(recs)


def fake_open_stream(
    *,
    split="train",
    seed=0,
    epoch=0,
    shard=0,
    num_shards=1,
    match_index: str | None = "present",
):
    """Open the local crossmatch-only fixture with production semantics."""
    import json

    from datasets.distributed import split_dataset_by_node

    if match_index is None:
        raise ValueError("crossmatch-only streaming requires a match index")

    fx = _fixtures()
    all_images = fx["images"]
    selected = aligned(
        shuffled(split_files(list(range(len(all_images))), split), seed, epoch),
        num_shards,
    )
    spectra_paths_by_cell = {cell: fx["spectra"] for cell in range(len(all_images))}
    owners = _spectrum_owners(spectra_paths_by_cell)
    stream = crossmatch_dataset(
        image_paths=[all_images[cell] for cell in selected],
        match_json=[json.dumps(fx["match"])] * len(selected),
        spectra_paths=[spectra_paths_by_cell[cell] for cell in selected],
        owned_spectra=[
            [path for path in spectra_paths_by_cell[cell] if owners[path] == cell]
            for cell in selected
        ],
        matched_spectra_ids={str(value) for value in fx["match"].values()},
        features=union_features(all_images[0], fx["spectra"][0]),
    )
    if num_shards > 1:
        stream = split_dataset_by_node(stream, rank=shard, world_size=num_shards)
    return stream.map(decode_record)

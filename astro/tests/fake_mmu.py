"""An offline stand-in for :func:`astropt3.data.streaming.open_stream`.

Writes synthetic records to local parquet in the HUB's nested schema (via
``datasets`` ``Array3D``/``Sequence`` features, which serialise a 3-D flux the
way the real catalog does), then drives the REAL streaming tail over them:
``_parquet_stream`` per source -> :func:`interleaved` -> ``.map(decode_record)``.
Only the file source differs from the hub path, so per-record weighting, node
splitting, resume, and the actual ``decode_record`` are all exercised.

Monkeypatch ``astropt3.data.streaming.open_stream`` with
:func:`fake_open_stream`; the loader imports it at call time.
"""

import tempfile
from pathlib import Path

import numpy as np
from datasets import Array3D, Dataset, Features, Sequence, Value

from astropt3.data.streaming import (
    DEFAULT_WEIGHTS,
    SOURCE_NAMES,
    _parquet_stream,
    _to_union,
    decode_record,
    interleaved,
    pairs_dataset,
    shuffled,
    split_files,
    union_features,
)
from astropt3.data.synthetic import IMAGE_BANDS, SPECTRUM_LENGTH, make_record

_N = {"images": 24, "spectra": 24}  # per source; disjoint id bases below
_BASE = {"images": 0, "spectra": 1000}
_FILES_PER_SOURCE = 4

_IMAGE_FEATURES = Features(
    {
        "object_id": Value("string"),
        "ra": Value("float64"),
        "dec": Value("float64"),
        "_healpix_29": Value("int64"),
        "image": {"flux": Array3D((3, 152, 152), "float32"), "band": Sequence(Value("string"))},
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
        rows = [_image_row(make_record(_BASE[name] + i, image_only_fraction=1.0)) for i in range(_N[name])]
        feats = _IMAGE_FEATURES
    else:
        rows = [
            _spectrum_row(make_record(_BASE[name] + i, image_only_fraction=0.0, spectrum_only_fraction=1.0))
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


def fixed_records(*, split="train", seed=0, epoch=0, shard=0, num_shards=1, match_index=None):
    """A deterministic, guaranteed mix of record kinds — for eval-logic tests
    (probe, template selection) that need specific shapes present regardless of
    weighting. Cycles image-only / spectrum-only / bimodal; not the datasets
    machinery, just an iterable of records (which is all a consumer needs)."""
    recs = []
    for i in range(10):
        recs.append(make_record(i, image_only_fraction=1.0))  # image-only, no Z
        recs.append(make_record(1000 + i, image_only_fraction=0.0, spectrum_only_fraction=1.0))
        recs.append(make_record(2000 + i, image_only_fraction=0.0, spectrum_only_fraction=0.0))
    return iter(recs)


def fake_open_stream(
    *,
    split="train",
    seed=0,
    epoch=0,
    shard=0,
    num_shards=1,
    match_index=None,
    only=None,
    skim_images=False,
):
    """``only`` restricts the corpus to a single source; ``match_index`` mirrors
    the real signature — None drops the pairs source, as it does live.
    ``skim_images`` mirrors ADR 0011: the scan feeds image-only draws and the
    standalone images source is dropped."""
    import json

    fx = _fixtures()

    image_files = shuffled(split_files(fx["images"], split), seed, epoch)
    spectra_files = shuffled(split_files(fx["spectra"], split), seed, epoch)
    features = union_features(image_files[0], spectra_files[0])

    if match_index is not None and skim_images:  # ADR 0011 image-only skim
        ipp = DEFAULT_WEIGHTS[0] / DEFAULT_WEIGHTS[2]
        scan = pairs_dataset(
            image_paths=image_files,
            match_json=[json.dumps(fx["match"])] * len(image_files),
            spectra_paths=[spectra_files] * len(image_files),
            features=features,
            images_per_pair=ipp,
        )
        spectra = _to_union(_parquet_stream(spectra_files), features, absent="image")
        weights = [DEFAULT_WEIGHTS[0] + DEFAULT_WEIGHTS[2], DEFAULT_WEIGHTS[1]]
        stream = interleaved([scan, spectra], weights, seed, shard, num_shards)
        return stream.map(decode_record)

    names = list(SOURCE_NAMES) if match_index is not None else list(SOURCE_NAMES[:2])
    weights = list(DEFAULT_WEIGHTS[: len(names)])
    if only is not None:
        weights = [1.0 if n == only else 0.0 for n in names]

    parts = [
        _to_union(_parquet_stream(image_files), features, absent="spectrum"),
        _to_union(_parquet_stream(spectra_files), features, absent="image"),
    ]
    if match_index is not None:
        parts.append(
            pairs_dataset(
                image_paths=[image_files[0]],
                match_json=[json.dumps(fx["match"])],
                spectra_paths=[spectra_files],
                features=features,
            )
        )

    stream = interleaved(parts, weights, seed, shard, num_shards)
    return stream.map(decode_record)

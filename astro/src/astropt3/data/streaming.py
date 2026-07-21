"""Stream MMU catalogs natively at train time (ADR 0006).

Replaces the local reshard: the MMU catalogs are streamed live from the HF
hub and interleaved per record, with no ``PILOT_FEATURES`` in the middle —
hub rows decode straight into the record dicts ``ObjectSequencer`` eats.

With a match index (``scripts/build_match_index.py``) the corpus is two
sources (ADR 0011, adopted after the 2026-07-21 A/B):

- **the crossmatch scan** — one pass over the matched LegacySurvey
  partitions demuxed into image x spectrum **pairs** plus **image-only**
  records skimmed from the otherwise-discarded unmatched rows, governed to
  the 0.60:0.25 images:pairs ratio (so there is no standalone images
  download),
- **spectra-only** — the DESI catalog as single-modality spectrum rows.

Without a match index the corpus degrades to images + spectra.

The transport is **HF ``datasets`` streaming**, not a hand-rolled reader.
``hats`` enumerates each catalog's parquet partitions (deterministic HEALPix
order, ``hf://`` resolution) and ``datasets`` does the rest: per-record
weighted interleave (``interleave_datasets(probabilities=...)``),
rank splitting (``split_dataset_by_node`` — the DataLoader-worker split is
datasets' own ``_iter_pytorch`` job; a manual worker split double-shards and
clamps the loader to one worker), exact resume
(``state_dict`` / ``load_state_dict``), and hub retry. An earlier hand-rolled
reader owned all of that and died at step 149 of the first real run on a
shared-httpx-client lifecycle bug — precisely the class of thing the library
already handles.

**Never call ``IterableDataset.shuffle()``**: in datasets 5.x it collapses
``n_shards`` to 1 and silently destroys rank/worker sharding. Partition order
is randomised by permuting the FILE LIST per epoch (:func:`shuffled`), which
is deterministic given ``(seed, epoch)`` and identical on every rank.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

IMAGES_CATALOG = "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north"
SPECTRA_CATALOG = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"
CROSSMATCH_RADIUS_ARCSEC = 1.0
IMAGE_SHAPE = (3, 152, 152)

SYNTHETIC_ROOT = "synthetic"
MMU_ROOT = "mmu"

SOURCE_NAMES = ("images", "spectra", "pairs")
# ADR 0006 §2, provisional: images dominate (bulk, cleanest signal), pairs are
# up-weighted ~5x over their natural share because cross-modal learning is the
# point, spectra sit between. Tuning these is the deferred mixing issue.
DEFAULT_WEIGHTS = (0.60, 0.15, 0.25)

# ADR 0006 §5: val reserves the first K partitions of every source in natural
# HEALPix order; whole partitions, so train/val are spatially disjoint. The
# pairs source is an inner join, so every val pair carries Z for the probe.
VAL_PARTITIONS = 8

MATCH_INDEX_ENV = "ASTROPT3_MATCH_INDEX"

# image-catalog columns carried onto the record
_IMAGE_SCALARS = ("ebv", "flux_g", "flux_r", "flux_z", "z_spec")


# -- decode: hub row -> record dict ------------------------------------------
# The sole adapter over MMU-native rows (ADR 0006 §7). No intermediate schema.
# datasets yields plain dicts, so `_healpix_29` is a real column and the struct
# fields are already dicts — none of the lsdb frame coercions apply.


def _stack_ragged(arr: np.ndarray) -> np.ndarray:
    """Recursively stack object arrays-of-arrays (arrow nested lists)."""
    if arr.dtype == object:
        return np.stack([_stack_ragged(np.asarray(x)) for x in arr])
    return arr


def _image_flux(value) -> np.ndarray:
    """Coerce nested lists / object arrays of band images to (3, 152, 152)."""
    arr = _stack_ragged(np.asarray(value)).astype(np.float32, copy=False)
    if arr.shape != IMAGE_SHAPE:
        raise ValueError(f"image flux has shape {arr.shape}, expected {IMAGE_SHAPE}")
    return arr


def _finite(value) -> bool:
    return value is not None and math.isfinite(float(value))


def _base(row) -> dict:
    return {
        "object_id": str(row["object_id"]),
        "ra": row["ra"],
        "dec": row["dec"],
        "_healpix_29": row["_healpix_29"],
    }


def _spectrum_part(row) -> dict:
    return {
        key: np.asarray(value, dtype=bool if key == "mask" else np.float32)
        for key, value in row["spectrum"].items()
    }


def _attach_spectrum(record: dict, row) -> None:
    record["spectrum"] = _spectrum_part(row)
    for key in ("Z", "ZERR"):
        if _finite(row.get(key)):
            record[key] = float(row[key])
    if row.get("ZWARN") is not None:
        record["ZWARN"] = bool(row["ZWARN"])


def _attach_image(record: dict, row) -> None:
    record["image"] = {
        "flux": _image_flux(row["image"]["flux"]),
        "band": [str(b) for b in row["image"]["band"]],
    }
    for key in _IMAGE_SCALARS:
        if _finite(row.get(key)):
            record[key] = float(row[key])


def decode_record(row) -> dict:
    """A hub row (possibly a null-unioned interleave row) -> record dict.

    Decode runs LAST, after ``interleave_datasets`` has already reconciled the
    three sources' schemas (image-only rows carry null spectrum columns and
    vice versa). Doing it here, not before interleave, is deliberate: the
    ``(3, 152, 152)`` flux is a numpy array once decoded, and datasets' feature
    inference cannot serialise a 3-D array — so the arrays must not exist until
    after interleave has resolved features from the raw (Arrow-native, nested
    list) parquet columns. Dispatch is on which struct is present.
    """
    record = _base(row)
    # after interleave a missing struct is filled with nulls, not dropped, so
    # presence is decided by a payload — and specifically by a NULLABLE field.
    # image.flux is Array3D, which is not nullable and fills to a zero array on
    # a spectrum-only row; image.band (a Sequence) fills to None, so it is the
    # reliable image signal. spectrum.flux (a Sequence) fills to None likewise.
    image = row.get("image")
    spectrum = row.get("spectrum")
    has_image = image is not None and image.get("band") is not None
    has_spectrum = spectrum is not None and spectrum.get("flux") is not None
    if has_image:
        _attach_image(record, row)
    if has_spectrum:
        _attach_spectrum(record, row)
    if not (has_image or has_spectrum):
        raise ValueError(f"row {record['object_id']!r} has neither image nor spectrum")
    return record


# -- catalog partitions ------------------------------------------------------


def catalog_files(url: str) -> tuple[list[str], dict]:
    """HEALPix-ordered ``hf://`` parquet paths, plus a ``(order, pixel) -> path`` map.

    The ordered list drives the images/spectra sources; the map resolves the
    match-index, which identifies partitions by HEALPix cell so the published
    artifact does not depend on this listing's ordering.
    """
    import hats
    from hats.io import paths

    collection = hats.read_hats(url)
    catalog = getattr(collection, "main_catalog", collection)
    files, by_cell = [], {}
    for pixel in catalog.get_healpix_pixels():
        rel = str(paths.pixel_catalog_file(catalog.catalog_base_dir, pixel)).replace(
            "hf://datasets/", "datasets/"
        )
        path = "hf://" + rel
        files.append(path)
        by_cell[(int(pixel.order), int(pixel.pixel))] = path
    return files, by_cell


def split_files(files: list, split: str, val_partitions: int = VAL_PARTITIONS) -> list:
    """Reserve the first K partitions for val; train gets the rest.

    Whole partitions, so the split is spatially disjoint and cannot leak. The
    cap keeps a small source (a partial match index) from being swallowed
    whole by the reservation.
    """
    if not val_partitions:
        return list(files)
    reserved = min(val_partitions, max(1, len(files) // 5))
    return files[:reserved] if split == "val" else files[reserved:]


def shuffled(files: list, seed: int, epoch: int) -> list:
    """Deterministic per-epoch order — identical on every rank.

    Replaces ``IterableDataset.shuffle()``, which in datasets 5.x collapses
    ``n_shards`` to 1 and silently destroys rank/worker sharding.
    """
    order = np.random.default_rng([seed, epoch]).permutation(len(files))
    return [files[i] for i in order]


def aligned(files: list, num_shards: int) -> list:
    """Truncate to a multiple of ``num_shards``.

    ``split_dataset_by_node`` only shard-splits when every source's shard
    count divides evenly; otherwise datasets falls back to example-stepping,
    which collapses ``n_shards`` to 1 and clamps the DataLoader to a single
    worker (the pairs source hit this: 165 train cells % dp 2 != 0). Call
    AFTER :func:`shuffled` so the <= num_shards-1 dropped partitions rotate
    with the epoch.
    """
    return files[: len(files) - len(files) % num_shards] if num_shards > 1 else files


# -- match index -------------------------------------------------------------


def resolve_match_index(match_index: str | None = None) -> str | None:
    """Explicit argument, else ``$ASTROPT3_MATCH_INDEX``, else None.

    The index is a corpus-level constant, so the env var mirrors the existing
    ``ASTROPT3_DATA_ROOT`` precedent and spares every eval entry point a
    parameter it would only forward. Training passes it explicitly from the
    nanotron config, which wins.
    """
    return match_index or os.environ.get(MATCH_INDEX_ENV) or None


def load_match_index(path: str):
    """The precomputed crossmatch, keyed by HEALPix cell.

    Built offline by ``scripts/build_match_index.py``; ids only, tens of MB,
    no pixels duplicated. ``path`` may be local or an ``hf://`` URL — pyarrow
    resolves the latter through huggingface_hub's fsspec registration, so a
    published index needs no extra plumbing.

    Returns ``(matches, spectra_of)`` keyed by the image partition's
    ``(order, pixel)``.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    matches: dict[tuple, dict] = {}
    spectra_of: dict[tuple, set] = {}
    for i in range(len(table["image_id"])):
        image_cell = (int(table["image_order"][i]), int(table["image_pixel"][i]))
        spectrum_cell = (int(table["spectrum_order"][i]), int(table["spectrum_pixel"][i]))
        matches.setdefault(image_cell, {})[table["image_id"][i]] = table["spectrum_id"][i]
        spectra_of.setdefault(image_cell, set()).add(spectrum_cell)
    return matches, spectra_of


# -- datasets construction ---------------------------------------------------


def _parquet_stream(files: list):
    from datasets import load_dataset

    return load_dataset("parquet", data_files=list(files), split="train", streaming=True)


def _pq_rows(path: str):
    """Plain pyarrow row-group reads, one row dict at a time.

    The pairs generator runs INSIDE a DataLoader worker, and a nested
    datasets ``IterableDataset`` iterated there gets worker-split again by
    ``_iter_pytorch`` — its single shard lands on worker 0 and every other
    worker silently reads an empty stream (no pairs, instantly-exhausted
    source). pyarrow has no worker magic, so the inner reads use it.
    """
    import fsspec
    import pyarrow.parquet as pq

    with fsspec.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for i in range(pf.num_row_groups):
            table = pf.read_row_group(i)
            # convert one row at a time: a bulk to_pylist of a 56 MB image row
            # group is ~0.5 GB of Python objects, and the job runs under a
            # 96 GiB slurm cgroup shared by 16 workers
            for j in range(table.num_rows):
                yield table.slice(j, 1).to_pylist()[0]


def _paired_examples(image_paths, match_json, spectra_paths, images_per_pair=0.0):
    """Generator for the demux scan: one image partition at a time.

    Yields RAW merged rows (the image row plus the matched spectrum's struct
    columns), NOT decoded records — decode runs after interleave, so the flux
    stays an Arrow-native nested list here. ``gen_kwargs`` lists are what
    datasets shards on, so the three are parallel — one entry per matched image
    partition. Matched spectrum rows for one image partition are bounded by the
    spectroscopic side (in practice one spectrum partition per image
    partition), so they are read once into an id -> row map — held as a
    FILTERED ARROW TABLE, not Python dicts: ~750 spectrum rows as Python
    objects is ~0.75 GB resident PER WORKER, which the 96 GiB slurm cgroup
    cannot afford 16 times over. Rows are converted one at a time on match.

    ADR 0011 image-only skim: with ``images_per_pair > 0`` the *unmatched*
    image rows — already downloaded by this scan, otherwise discarded — are
    also emitted as image-only records (null spectrum), which is what lets the
    standalone images-catalog download be dropped. A deterministic per-partition
    budget governor meters them to ``images_per_pair`` skims per matched pair:
    each pair funds ``images_per_pair`` of budget, each skim spends one. It
    self-adjusts to the unknown per-cell match density d — below the break-even
    it caps the skim at the ratio and drops the surplus discards; above it,
    supply runs short and every discard is emitted (no RNG, coarse determinism;
    ADR 0011 §Determinism model).
    """
    import fsspec
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    for image_path, raw, spectrum_paths in zip(image_paths, match_json, spectra_paths):
        wanted = json.loads(raw)  # image_id -> spectrum_id
        needed = pa.array({str(v) for v in wanted.values()}, type=pa.string())
        tables = []
        for path in spectrum_paths:
            with fsspec.open(path, "rb") as f:
                pf = pq.ParquetFile(f)
                for i in range(pf.num_row_groups):
                    t = pf.read_row_group(i)
                    keep = pc.is_in(pc.cast(t["object_id"], pa.string()), needed)
                    if pc.any(keep).as_py():
                        tables.append(t.filter(keep))
        matched = pa.concat_tables(tables) if tables else None
        spectra = (
            {}
            if matched is None
            else {
                matched["object_id"][j].as_py(): j for j in range(matched.num_rows)
            }
        )
        image_budget = 0.0  # leaky bucket: each pair funds images_per_pair skims
        for row in _pq_rows(image_path):
            spectrum_id = wanted.get(str(row["object_id"]))
            if spectrum_id is None:  # unmatched: skim as image-only if the ratio allows
                if image_budget >= 1.0:
                    image_budget -= 1.0
                    skimmed = dict(row)
                    skimmed["spectrum"] = None
                    for key in ("Z", "ZERR", "ZWARN"):
                        skimmed[key] = None
                    yield skimmed
                continue
            j = spectra.get(str(spectrum_id))
            if j is None:  # index newer than the catalog revision
                continue
            spectrum_row = matched.slice(j, 1).to_pylist()[0]
            merged = dict(row)
            merged["spectrum"] = spectrum_row["spectrum"]
            for key in ("Z", "ZERR", "ZWARN"):
                merged[key] = spectrum_row.get(key)
            image_budget += images_per_pair
            yield merged


def union_features(image_file: str, spectra_file: str):
    """The image ∪ spectrum feature schema — DERIVED from the parquet, not
    hand-written. Interleave needs every source aligned to one schema, and the
    3-D ``Array3D`` flux breaks datasets' inference, so all three sources are
    cast to this union (image-only rows carry a null spectrum struct and vice
    versa). Reading it from the catalogs keeps it out of the code as a
    parallel schema — it is exactly what the hub publishes.
    """
    from datasets import Features

    image = _parquet_stream([image_file]).features
    spectrum = _parquet_stream([spectra_file]).features
    return Features({**image, **spectrum})


def _to_union(source, features, absent: str):
    """Cast a single-modality parquet source to the union schema, setting the
    modality it lacks to an explicit null (a bare interleave scrambles the
    nested null-fill; an explicit None + fixed features is reliable)."""
    return source.map(lambda row: {**row, absent: None}, features=features)


def pairs_dataset(image_paths, match_json, spectra_paths, features, images_per_pair=0.0):
    from datasets import IterableDataset

    return IterableDataset.from_generator(
        _paired_examples,
        gen_kwargs={
            "image_paths": image_paths,
            "match_json": match_json,  # plain strings so datasets can shard them
            "spectra_paths": spectra_paths,
            "images_per_pair": images_per_pair,  # scalar: passed to every shard
        },
        features=features,
    )


def _pairs_dataset(
    match_index: str,
    split: str,
    seed: int,
    epoch: int,
    num_shards: int,
    images_per_pair: float = 0.0,
):
    image_files, image_by_cell = catalog_files(IMAGES_CATALOG)
    spectra_files, spectra_by_cell = catalog_files(SPECTRA_CATALOG)
    matches, spectra_of = load_match_index(match_index)

    cells = sorted(matches)
    missing = [c for c in cells if c not in image_by_cell]
    if missing:
        raise ValueError(
            f"match index references {len(missing)} image partitions absent from "
            f"{IMAGES_CATALOG} (first: {missing[0]}); the index was built against "
            "a different catalog revision — rebuild it"
        )
    cells = aligned(shuffled(split_files(cells, split), seed, epoch), num_shards)

    return pairs_dataset(
        image_paths=[image_by_cell[c] for c in cells],
        match_json=[json.dumps(matches[c]) for c in cells],
        spectra_paths=[[spectra_by_cell[s] for s in sorted(spectra_of[c])] for c in cells],
        features=union_features(image_files[0], spectra_files[0]),
        images_per_pair=images_per_pair,
    )


def interleaved(parts: list, weights: list, seed: int, shard: int, num_shards: int):
    """Weighted per-record interleave of decoded sources, then node-split.

    The shared core of :func:`open_stream` — factored out so an offline test
    can drive the identical datasets machinery (interleave probabilities,
    ``all_exhausted`` stopping, ``split_dataset_by_node``, resume) over local
    parquet instead of the hub.
    """
    from datasets import interleave_datasets
    from datasets.distributed import split_dataset_by_node

    total = sum(weights)
    stream = interleave_datasets(
        parts,
        probabilities=[w / total for w in weights],
        seed=seed,
        stopping_strategy="all_exhausted",
    )
    if num_shards > 1:
        stream = split_dataset_by_node(stream, rank=shard, world_size=num_shards)
    return stream


def open_stream(
    *,
    split: str = "train",
    seed: int = 0,
    epoch: int = 0,
    shard: int = 0,
    num_shards: int = 1,
    match_index: str | None = None,
):
    """The live corpus as one interleaved ``datasets.IterableDataset``.

    Deterministic given ``(seed, epoch, split, shard)`` — evaluation relies on
    it, so a fresh ``split="val"`` stream replays identical records on every
    checkpoint. Yields record dicts ready for ``ObjectSequencer``.

    One finite pass over the epoch's files. Callers that need an endless
    stream re-open at ``epoch + 1`` (which reshuffles the file order); the
    nanotron loader does exactly that.
    """
    match_index = resolve_match_index(match_index)

    image_files = aligned(
        shuffled(split_files(catalog_files(IMAGES_CATALOG)[0], split), seed, epoch),
        num_shards,
    )
    spectra_files = aligned(
        shuffled(split_files(catalog_files(SPECTRA_CATALOG)[0], split), seed, epoch),
        num_shards,
    )
    features = union_features(image_files[0], spectra_files[0])

    # every source is cast to one union schema (Arrow-native, image ∪ spectrum)
    # so interleave aligns cleanly; decode to numpy runs LAST, only on iteration
    spectra_part = _to_union(_parquet_stream(spectra_files), features, absent="image")
    if match_index is not None:
        # ADR 0011 (adopted after the 2026-07-21 A/B): one scan of the matched
        # partitions yields BOTH pairs and image-only records (from
        # otherwise-discarded unmatched rows), so the standalone images-catalog
        # download is dropped entirely. images:pairs is governed to the
        # 0.60:0.25 ratio inside the scan, so the scan carries their combined
        # weight; spectra stay single-sourced (no spectra skim — see
        # §Determinism model / Consequences).
        images_per_pair = DEFAULT_WEIGHTS[0] / DEFAULT_WEIGHTS[2]
        parts = [
            _pairs_dataset(match_index, split, seed, epoch, num_shards, images_per_pair),
            spectra_part,
        ]
        weights = [DEFAULT_WEIGHTS[0] + DEFAULT_WEIGHTS[2], DEFAULT_WEIGHTS[1]]
    else:
        parts = [
            _to_union(_parquet_stream(image_files), features, absent="spectrum"),
            spectra_part,
        ]
        weights = list(DEFAULT_WEIGHTS[:2])

    stream = interleaved(parts, weights, seed, shard, num_shards)
    # one line per (re)build: the silent failure mode here is a source whose
    # shard count breaks the rank/worker split and clamps the loader to one
    # worker, so make the counts visible where it happens
    print(
        f"[data] open_stream split={split} epoch={epoch} shard={shard}/{num_shards} "
        f"source n_shards={[p.n_shards for p in parts]} -> stream n_shards={stream.n_shards}",
        flush=True,
    )
    return stream.map(decode_record)

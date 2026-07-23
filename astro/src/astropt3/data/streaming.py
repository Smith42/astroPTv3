"""Stream the crossmatch-only MMU corpus natively at train time.

The precomputed match index (``scripts/build_match_index.py``) defines the
corpus. One pass over its LegacySurvey partitions emits every downloaded row:
matched image × spectrum pairs, unmatched images, and globally unmatched
spectra. There is no standalone source, weighting, skim governor, or local
cache; the modality mix follows the data.

``datasets.IterableDataset.from_generator`` owns worker sharding and resume.
DP ranks are split with ``split_dataset_by_node``. Never call
``IterableDataset.shuffle()``: datasets 5.x collapses ``n_shards`` to one.
Instead, partition paths are deterministically permuted per epoch.
"""

from __future__ import annotations

import importlib
import json
import math
import os
from typing import Any, cast

import numpy as np

IMAGES_CATALOG = "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north"
SPECTRA_CATALOG = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"
CROSSMATCH_RADIUS_ARCSEC = 1.0
IMAGE_SHAPE = (3, 152, 152)

SYNTHETIC_ROOT = "synthetic"
MMU_ROOT = "mmu"
SOURCE_ASSEMBLY = "crossmatch_only"

# Whole image cells are reserved, so image-bearing train/val objects are
# spatially disjoint. Stable global ownership assigns each unmatched spectrum
# partition to exactly one of those cells, preventing cross-split duplication.
VAL_PARTITIONS = 8
MATCH_INDEX_ENV = "ASTROPT3_MATCH_INDEX"

_IMAGE_SCALARS = ("ebv", "flux_g", "flux_r", "flux_z", "z_spec")


# -- decode: hub row -> record dict ------------------------------------------


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


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected a numeric value, got {value!r}") from error


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected an integer value, got {value!r}") from error


def _finite(value) -> bool:
    return value is not None and math.isfinite(_as_float(value))


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
            record[key] = _as_float(row[key])
    if row.get("ZWARN") is not None:
        record["ZWARN"] = bool(row["ZWARN"])


def _attach_image(record: dict, row) -> None:
    record["image"] = {
        "flux": _image_flux(row["image"]["flux"]),
        "band": [str(b) for b in row["image"]["band"]],
    }
    for key in _IMAGE_SCALARS:
        if _finite(row.get(key)):
            record[key] = _as_float(row[key])


def decode_record(row) -> dict:
    """Convert one raw union-schema row into an ``ObjectSequencer`` record."""
    record = _base(row)
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
    """Return HEALPix-ordered parquet paths and a cell-to-path mapping."""
    paths = cast(Any, importlib.import_module("hats.io.paths"))
    read_hats = cast(Any, importlib.import_module("hats.loaders.read_hats")).read_hats

    collection = read_hats(url)
    catalog = cast(Any, getattr(collection, "main_catalog", collection))
    files, by_cell = [], {}
    for pixel in catalog.get_healpix_pixels():
        rel = str(paths.pixel_catalog_file(catalog.catalog_base_dir, pixel)).replace(
            "hf://datasets/", "datasets/"
        )
        path = "hf://" + rel
        files.append(path)
        by_cell[(_as_int(pixel.order), _as_int(pixel.pixel))] = path
    return files, by_cell


def split_files(files: list, split: str, val_partitions: int = VAL_PARTITIONS) -> list:
    """Reserve the first K cells for validation and the remainder for train."""
    if not val_partitions:
        return list(files)
    reserved = min(val_partitions, max(1, len(files) // 5))
    return files[:reserved] if split == "val" else files[reserved:]


def shuffled(files: list, seed: int, epoch: int) -> list:
    """Return a deterministic per-epoch partition permutation."""
    order = np.random.default_rng([seed, epoch]).permutation(len(files))
    return [files[i] for i in order]


def aligned(files: list, num_shards: int) -> list:
    """Rotate then truncate paths to a multiple of the DP shard count."""
    return files[: len(files) - len(files) % num_shards] if num_shards > 1 else files


# -- match index -------------------------------------------------------------


def resolve_match_index(match_index: str | None = None) -> str | None:
    """Use the explicit index, then ``$ASTROPT3_MATCH_INDEX``."""
    return match_index or os.environ.get(MATCH_INDEX_ENV) or None


def load_match_index(path: str):
    """Load image→spectrum matches and referenced spectrum cells by image cell."""
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    matches: dict[tuple, dict] = {}
    spectra_of: dict[tuple, set] = {}
    for i in range(len(table["image_id"])):
        image_cell = (
            _as_int(table["image_order"][i]),
            _as_int(table["image_pixel"][i]),
        )
        spectrum_cell = (
            _as_int(table["spectrum_order"][i]),
            _as_int(table["spectrum_pixel"][i]),
        )
        matches.setdefault(image_cell, {})[table["image_id"][i]] = table["spectrum_id"][
            i
        ]
        spectra_of.setdefault(image_cell, set()).add(spectrum_cell)
    return matches, spectra_of


# -- crossmatch dataset ------------------------------------------------------


def _parquet_stream(files: list):
    """Open parquet through datasets only to derive its published features."""
    from datasets import load_dataset

    return load_dataset(
        "parquet", data_files=list(files), split="train", streaming=True
    )


def union_features(image_file: str, spectrum_file: str):
    """Derive the raw image ∪ spectrum schema from the published catalogs."""
    from datasets import Features

    image = _parquet_stream([image_file]).features
    spectrum = _parquet_stream([spectrum_file]).features
    if image is None or spectrum is None:
        raise ValueError("catalog parquet did not expose a feature schema")
    return Features({**cast(dict, image), **cast(dict, spectrum)})


def _rows(parquet_file):
    """Yield rows without materializing a row group as Python objects."""
    for i in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(i)
        for j in range(table.num_rows):
            yield table.slice(j, 1).to_pylist()[0]


def _crossmatch_examples(
    image_paths,
    match_json,
    spectra_paths,
    owned_spectra,
    matched_spectra_ids,
):
    """Yield pairs, unmatched images, and globally unmatched spectra once."""
    import fsspec
    import pyarrow as pa
    import pyarrow.parquet as pq

    pc = cast(Any, importlib.import_module("pyarrow.compute"))
    paired_globally = pa.array(sorted(map(str, matched_spectra_ids)), type=pa.string())
    for image_path, raw, spectrum_paths, owned in zip(
        image_paths, match_json, spectra_paths, owned_spectra
    ):
        try:
            wanted = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("invalid match-index partition JSON") from error

        needed = pa.array(sorted(map(str, wanted.values())), type=pa.string())
        owned = set(owned)
        matched_tables, unmatched_tables = [], []
        for path in spectrum_paths:
            with fsspec.open(path, "rb") as file:
                parquet = pq.ParquetFile(file)
                for i in range(parquet.num_row_groups):
                    table = parquet.read_row_group(i)
                    ids = table["object_id"].cast(pa.string())
                    matched_tables.append(table.filter(pc.is_in(ids, needed)))
                    if path in owned:
                        unmatched_tables.append(
                            table.filter(pc.invert(pc.is_in(ids, paired_globally)))
                        )

        matched = pa.concat_tables(matched_tables) if matched_tables else None
        spectra = (
            {}
            if matched is None
            else {
                str(matched["object_id"][i].as_py()): i for i in range(matched.num_rows)
            }
        )
        unmatched = pa.concat_tables(unmatched_tables) if unmatched_tables else None
        unmatched_count = 0 if unmatched is None else unmatched.num_rows
        unmatched_index = 0

        with fsspec.open(image_path, "rb") as file:
            parquet = pq.ParquetFile(file)
            stride = (
                max(1, parquet.metadata.num_rows // unmatched_count)
                if unmatched_count
                else 0
            )
            for row_position, row in enumerate(_rows(parquet)):
                if (
                    unmatched is not None
                    and unmatched_index < unmatched_count
                    and row_position % stride == 0
                ):
                    yield {
                        **unmatched.slice(unmatched_index, 1).to_pylist()[0],
                        "image": None,
                    }
                    unmatched_index += 1

                spectrum_id = wanted.get(str(row["object_id"]))
                if spectrum_id is None:
                    yield {
                        **row,
                        "spectrum": None,
                        "Z": None,
                        "ZERR": None,
                        "ZWARN": None,
                    }
                    continue

                index = spectra.get(str(spectrum_id))
                if index is None or matched is None:
                    continue  # match index and catalog revision disagree
                spectrum = matched.slice(index, 1).to_pylist()[0]
                yield {
                    **row,
                    "spectrum": spectrum["spectrum"],
                    "Z": spectrum.get("Z"),
                    "ZERR": spectrum.get("ZERR"),
                    "ZWARN": spectrum.get("ZWARN"),
                }

        while unmatched is not None and unmatched_index < unmatched_count:
            yield {
                **unmatched.slice(unmatched_index, 1).to_pylist()[0],
                "image": None,
            }
            unmatched_index += 1


def crossmatch_dataset(
    image_paths,
    match_json,
    spectra_paths,
    owned_spectra,
    matched_spectra_ids,
    features,
):
    """Build the sharded raw crossmatch dataset from resolved partition paths."""
    from datasets import IterableDataset

    return IterableDataset.from_generator(
        _crossmatch_examples,
        gen_kwargs={
            "image_paths": image_paths,
            "match_json": match_json,
            "spectra_paths": spectra_paths,
            "owned_spectra": owned_spectra,
            "matched_spectra_ids": matched_spectra_ids,
        },
        features=features,
    )


def _spectrum_owners(spectra_paths_by_cell: dict) -> dict:
    """Assign each spectrum partition to one stable image cell globally."""
    import zlib

    references: dict[str, list[tuple]] = {}
    for cell in sorted(spectra_paths_by_cell):
        for path in spectra_paths_by_cell[cell]:
            references.setdefault(path, []).append(cell)
    return {
        path: cells[zlib.crc32(path.encode()) % len(cells)]
        for path, cells in references.items()
    }


def _crossmatch_dataset(match_index, split, seed, epoch, num_shards):
    image_files, image_by_cell = catalog_files(IMAGES_CATALOG)
    spectrum_files, spectrum_by_cell = catalog_files(SPECTRA_CATALOG)
    matches, spectra_of = load_match_index(match_index)

    all_cells = sorted(matches)
    missing = [cell for cell in all_cells if cell not in image_by_cell]
    if missing:
        raise ValueError(
            f"match index references {len(missing)} image partitions absent from "
            f"{IMAGES_CATALOG} (first: {missing[0]}); rebuild the index"
        )

    paths_by_cell = {
        cell: [spectrum_by_cell[s] for s in sorted(spectra_of[cell])]
        for cell in all_cells
    }
    owners = _spectrum_owners(paths_by_cell)
    cells = aligned(shuffled(split_files(all_cells, split), seed, epoch), num_shards)
    return crossmatch_dataset(
        image_paths=[image_by_cell[cell] for cell in cells],
        match_json=[json.dumps(matches[cell]) for cell in cells],
        spectra_paths=[paths_by_cell[cell] for cell in cells],
        owned_spectra=[
            [path for path in paths_by_cell[cell] if owners[path] == cell]
            for cell in cells
        ],
        matched_spectra_ids={
            str(spectrum_id)
            for cell_matches in matches.values()
            for spectrum_id in cell_matches.values()
        },
        features=union_features(image_files[0], spectrum_files[0]),
    )


def open_stream(
    *,
    split: str = "train",
    seed: int = 0,
    epoch: int = 0,
    shard: int = 0,
    num_shards: int = 1,
    match_index: str | None = None,
):
    """Open one finite, deterministic epoch of the crossmatch-only corpus."""
    from datasets.distributed import split_dataset_by_node

    match_index = resolve_match_index(match_index)
    if match_index is None:
        raise ValueError(
            f"crossmatch-only MMU streaming requires match_index or ${MATCH_INDEX_ENV}"
        )

    stream = _crossmatch_dataset(match_index, split, seed, epoch, num_shards)
    if num_shards > 1:
        stream = split_dataset_by_node(stream, rank=shard, world_size=num_shards)
    print(
        f"[data] open_stream {SOURCE_ASSEMBLY} split={split} epoch={epoch} "
        f"shard={shard}/{num_shards} n_shards={stream.n_shards}",
        flush=True,
    )
    return stream.map(decode_record)

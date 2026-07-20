"""Stream MMU catalogs natively at train time (ADR 0006).

Replaces the local reshard: three sources are streamed live from the HF hub
and interleaved per record, with no ``PILOT_FEATURES`` in the middle —
lsdb rows decode straight into the record dicts ``ObjectSequencer`` eats.

- **images-only** — the LegacySurvey catalog as single-modality image rows,
- **spectra-only** — the DESI catalog as single-modality spectrum rows,
- **pairs** — the ``how="inner"`` crossmatch, as multimodal rows.

A matched object's image appears both standalone and paired; per ADR 0006 §1
that redundancy is accepted.

Two things carry the design:

*Weights are applied per record, not per partition draw.* Partitions hold
wildly different row counts (thousands of spectra against hundreds of pairs),
so drawing whole partitions by weight would realize a corpus mix nothing like
the configured one. Each source keeps its current partition buffered and only
fetches the next when that buffer drains.

*The draw order is a fixed pattern, not a sampler.* ``_pattern`` turns the
weights into a repeating length-``PATTERN_LEN`` sequence of source indices, so
resume state is a handful of ints and no RNG state is ever checkpointed.

``fetch`` is injectable so the cursor logic is testable without lsdb or
network; :func:`open_sources` builds the real lsdb-backed fetchers and is the
only place lsdb is imported.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

IMAGES_CATALOG = "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north"
SPECTRA_CATALOG = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"
CROSSMATCH_RADIUS_ARCSEC = 1.0
IMAGE_SHAPE = (3, 152, 152)

SYNTHETIC_ROOT = "synthetic"
MMU_ROOT = "mmu"
# ADR 0006 §5: val reserves the first K partitions of every source in natural
# HEALPix order; whole partitions, so train/val are spatially disjoint. The
# pairs source is an inner join, so every val pair carries Z for the probe.
VAL_PARTITIONS = 8

SOURCE_NAMES = ("images", "spectra", "pairs")
# ADR 0006 §2, provisional: images dominate (bulk, cleanest signal), pairs are
# up-weighted ~5x over their natural share because cross-modal learning is the
# point, spectra sit between. Tuning these is the deferred mixing issue.
DEFAULT_WEIGHTS = (0.60, 0.15, 0.25)
PATTERN_LEN = 20

# image-catalog columns carried onto the record; the DESI side contributes
# spectrum, Z, ZERR, ZWARN
_LEFT_SCALARS = ("ebv", "flux_g", "flux_r", "flux_z", "z_spec")


# -- decode: lsdb row -> record dict ---------------------------------------
# The sole adapter over MMU-native rows (ADR 0006 §7). No intermediate schema.


def _is_null(cell) -> bool:
    if cell is None or cell is pd.NA:
        return True
    return isinstance(cell, (float, np.floating)) and math.isnan(cell)


def _stack_ragged(arr: np.ndarray) -> np.ndarray:
    """Recursively stack object arrays-of-arrays (arrow nested lists) to one array."""
    if arr.dtype == object:
        return np.stack([_stack_ragged(np.asarray(x)) for x in arr])
    return arr


def _image_flux(value) -> np.ndarray:
    """Coerce nested lists / object arrays of band images to (3, 152, 152)."""
    arr = _stack_ragged(np.asarray(value)).astype(np.float32, copy=False)
    if arr.shape != IMAGE_SHAPE:
        raise ValueError(f"image flux has shape {arr.shape}, expected {IMAGE_SHAPE}")
    return arr


def _struct(cell) -> dict:
    """Coerce a struct-column cell (dict or nested-pandas frame) to a dict."""
    if isinstance(cell, dict):
        return cell
    if hasattr(cell, "columns"):  # nested-pandas: a per-row sub-DataFrame
        return {c: cell[c].to_numpy() for c in cell.columns}
    raise TypeError(f"cannot interpret struct cell of type {type(cell)}")


class _Row:
    """Suffix-tolerant column access on a crossmatch result row."""

    def __init__(self, row, suffixes=("", "_desi")):
        self._row = row
        self._suffixes = suffixes

    def get(self, name, side=0):
        for key in (name + self._suffixes[side], name):
            if key in self._row.index:
                return self._row[key]
        return None


def _finalize(record: dict) -> dict:
    """Coerce modality payloads to arrays; drop a modality that is absent.

    A missing key (rather than a null one) IS the modality-optional contract
    ``ObjectSequencer`` expects.
    """
    record["object_id"] = str(record["object_id"])
    image = record.get("image")
    if image is None or _is_null(image.get("flux")) or len(image.get("flux", ())) == 0:
        record.pop("image", None)
    else:
        record["image"] = {
            "flux": _image_flux(image["flux"]),
            "band": [str(b) for b in image["band"]],
        }
    spectrum = record.get("spectrum")
    if spectrum is None or len(spectrum.get("flux", ())) == 0:
        record.pop("spectrum", None)
    else:
        record["spectrum"] = {
            key: np.asarray(value, dtype=bool if key == "mask" else np.float32)
            for key, value in spectrum.items()
        }
    return record


def row_to_record(row, suffixes=("", "_desi")) -> dict:
    """One image-catalog or crossmatch row -> record dict."""
    r = _Row(row, suffixes)
    healpix = r.get("_healpix_29")
    if healpix is None:
        # lsdb frames carry _healpix_29 as the spatial index, not a column;
        # under iterrows() it surfaces as the row name
        healpix = row.name
    image = r.get("image")
    record = {
        "object_id": r.get("object_id"),
        "ra": r.get("ra"),
        "dec": r.get("dec"),
        "_healpix_29": healpix,
        "image": None if _is_null(image) else _struct(image),
    }
    for key in _LEFT_SCALARS:
        value = r.get(key)
        if not _is_null(value):
            record[key] = float(value)

    spectrum = r.get("spectrum", side=1)
    if not _is_null(spectrum):
        record["spectrum"] = _struct(spectrum)
        for key in ("Z", "ZERR", "ZWARN"):
            value = r.get(key, side=1)
            if not _is_null(value):
                record[key] = bool(value) if key == "ZWARN" else float(value)
    return _finalize(record)


def spectra_row_to_record(row) -> dict:
    """One DESI catalog row -> spectrum-only record dict."""
    r = _Row(row)
    healpix = r.get("_healpix_29")
    if healpix is None:
        healpix = row.name
    record = {
        "object_id": r.get("object_id"),
        "ra": r.get("ra"),
        "dec": r.get("dec"),
        "_healpix_29": healpix,
        "spectrum": _struct(r.get("spectrum")),
    }
    for key in ("Z", "ZERR"):
        value = r.get(key)
        if not _is_null(value):
            record[key] = float(value)
    zwarn = r.get("ZWARN")
    if not _is_null(zwarn):
        record["ZWARN"] = bool(zwarn)
    return _finalize(record)


# -- the draw pattern -------------------------------------------------------


def _pattern(weights, length: int = PATTERN_LEN) -> tuple[int, ...]:
    """Weights -> a repeating, evenly-spread sequence of source indices.

    Greedy largest-remainder: exact counts when ``w * length`` is integral,
    and the sources stay interleaved rather than arriving in blocks.
    """
    acc = [0.0] * len(weights)
    out = []
    for _ in range(length):
        acc = [a + w for a, w in zip(acc, weights)]
        i = max(range(len(acc)), key=acc.__getitem__)
        acc[i] -= 1.0
        out.append(i)
    return tuple(out)


# -- sources ----------------------------------------------------------------


@dataclass
class Source:
    """One streamed catalog: how many partitions, and how to fetch/decode one."""

    name: str
    npartitions: int
    fetch: Callable[[int], pd.DataFrame]
    decode: Callable[[object], dict]
    order: np.ndarray | None = None
    epoch: int = 0
    cursor: int = 0  # index into `order`
    row_off: int = 0  # index into the buffered partition
    _buf: pd.DataFrame | None = field(default=None, repr=False)


def partition_order(npartitions: int, seed: int, epoch: int) -> np.ndarray:
    """Deterministic per-epoch partition order — identical on every rank."""
    return np.random.default_rng([seed, epoch]).permutation(npartitions)


class MMUStream:
    """Endless, weighted, per-record interleave of the three MMU sources.

    Resume is partition-granularity by construction: the cursor is
    ``(draw, per-source (epoch, cursor, row_off))``, all ints. Restoring it
    re-derives the partition order from ``(seed, epoch)`` and re-fetches only
    the in-flight partition — skipping costs nothing and downloads nothing,
    because a source's partitions are addressed by index.
    """

    def __init__(
        self,
        sources: list[Source],
        *,
        weights=DEFAULT_WEIGHTS,
        seed: int = 0,
        shard: int = 0,
        num_shards: int = 1,
        val_partitions: int = 0,
        split: str = "train",
    ):
        if len(sources) != len(weights):
            raise ValueError(f"{len(sources)} sources but {len(weights)} weights")
        if split not in ("train", "val"):
            raise ValueError(f"split must be train|val, got {split!r}")
        self.sources = sources
        self.pattern = _pattern(weights)
        self.seed = seed
        self.shard = shard
        self.num_shards = num_shards
        self.val_partitions = val_partitions
        self.split = split
        self.draw = 0
        # epoch of the source that produced the most recent record — seeds the
        # ADR 0008 span shuffle. Reproducible on resume (source epochs are in
        # the state, and which source drew is pattern[(draw - 1) % PATTERN_LEN]).
        self.last_epoch = 0
        for src in self.sources:
            self._reset_order(src)

    # -- partition bookkeeping ---------------------------------------------

    def _reset_order(self, src: Source) -> None:
        """(Re)build this source's partition order for its current epoch.

        Val reserves the first ``val_partitions`` partitions in the catalog's
        natural HEALPix order and train excludes them (ADR 0006 §5) — whole
        partitions, so the split is spatially disjoint and cannot leak.
        """
        order = partition_order(src.npartitions, self.seed, src.epoch)
        if self.val_partitions:
            keep = order < self.val_partitions
            order = order[keep if self.split == "val" else ~keep]
        # partition-level DP/worker split: disjoint, no cross-rank coordination
        src.order = order[self.shard :: self.num_shards]
        if len(src.order) == 0:
            raise ValueError(
                f"source {src.name!r} has no partitions for shard {self.shard}"
                f"/{self.num_shards} (split={self.split})"
            )
        src.cursor = 0
        src.row_off = 0
        src._buf = None

    def _advance_partition(self, src: Source) -> None:
        src.cursor += 1
        src.row_off = 0
        src._buf = None
        if src.cursor >= len(src.order):
            src.epoch += 1
            self._reset_order(src)

    def _next_record(self, src: Source) -> dict | None:
        """Next decoded row from this source, fetching partitions as needed."""
        # bounded: every iteration either returns or consumes a partition, and
        # a fresh epoch cannot be empty (_reset_order rejects an empty order)
        for _ in range(len(src.order) + 1):
            if src._buf is None:
                src._buf = src.fetch(int(src.order[src.cursor]))
            if src.row_off < len(src._buf):
                row = src._buf.iloc[src.row_off]
                src.row_off += 1
                self.last_epoch = src.epoch  # before any rollover below
                if src.row_off >= len(src._buf):
                    self._advance_partition(src)
                return src.decode(row)
            self._advance_partition(src)  # empty partition
        return None

    # -- iteration ----------------------------------------------------------

    def __iter__(self):
        while True:
            src = self.sources[self.pattern[self.draw % PATTERN_LEN]]
            self.draw += 1
            record = self._next_record(src)
            if record is not None:
                yield record

    # -- checkpoint state ---------------------------------------------------

    def state_dict(self) -> dict:
        """Position after the last yielded record. All ints — no RNG state."""
        return {
            "draw": self.draw,
            "seed": self.seed,
            "split": self.split,
            "sources": {
                s.name: {"epoch": s.epoch, "cursor": s.cursor, "row_off": s.row_off}
                for s in self.sources
            },
        }

    def load_state_dict(self, state: dict) -> None:
        if state["seed"] != self.seed or state["split"] != self.split:
            raise ValueError(
                f"state was saved for seed={state['seed']} split={state['split']!r}, "
                f"this stream is seed={self.seed} split={self.split!r}"
            )
        self.draw = state["draw"]
        for src in self.sources:
            saved = state["sources"][src.name]
            src.epoch = saved["epoch"]
            self._reset_order(src)  # re-derives order for the saved epoch
            src.cursor = saved["cursor"]
            src.row_off = saved["row_off"]


# -- lsdb wiring ------------------------------------------------------------


def open_sources(
    images_catalog: str = IMAGES_CATALOG,
    spectra_catalog: str = SPECTRA_CATALOG,
    radius_arcsec: float = CROSSMATCH_RADIUS_ARCSEC,
    client=None,
) -> list[Source]:
    """Open the three live MMU sources. The only place lsdb is imported.

    ``client`` is a dask client; with one, partition fetches prefetch in the
    background, without one they are synchronous.
    """
    import lsdb
    from lsdb.streams import CatalogStream

    images = lsdb.open_catalog(images_catalog)
    spectra = lsdb.open_catalog(spectra_catalog)
    pairs = images.crossmatch(
        spectra,
        n_neighbors=1,
        radius_arcsec=radius_arcsec,
        suffixes=("", "_desi"),
        # pinned: lsdb's default flips to "overlapping_columns" in a future
        # release, and row_to_record's _desi handling assumes all_columns
        suffix_method="all_columns",
        how="inner",
    )

    def fetcher(catalog):
        # CatalogStream is built once for its culled per-partition dask graphs
        # (it avoids O(N^2) graph transmission); we drive the partition order
        # ourselves, so shuffle is off and its own iterator is never used.
        stream = CatalogStream(catalog, client=client, shuffle=False, seed=0)
        return lambda index: stream.submit_next_partitions(np.array([index])).result()

    return [
        Source("images", images.npartitions, fetcher(images), row_to_record),
        Source("spectra", spectra.npartitions, fetcher(spectra), spectra_row_to_record),
        Source("pairs", pairs.npartitions, fetcher(pairs), row_to_record),
    ]


def open_stream(
    *,
    split: str = "train",
    seed: int = 0,
    shard: int = 0,
    num_shards: int = 1,
    client=None,
) -> MMUStream:
    """The live corpus stream. Deterministic given ``(seed, split, shard)``.

    Evaluation relies on that determinism: a fresh ``split="val"`` stream
    replays the identical records on every checkpoint.
    """
    return MMUStream(
        open_sources(client=client),
        seed=seed,
        shard=shard,
        num_shards=num_shards,
        val_partitions=VAL_PARTITIONS,
        split=split,
    )

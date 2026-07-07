"""Pilot-corpus parquet shards: canonical schema, shard writer, streaming reader.

``scripts/prepare_pilot_data.py`` (login node, ``[data]`` env) crossmatches the
MMU HATS catalogs and writes records through :func:`write_shard` into ~256MB
parquet shards under ``{root}/{train,val}/``. Training machines stream them
back with :class:`MMUIterableDataset` — fully offline
(``HF_DATASETS_OFFLINE=1``), sharded by DP rank (``split_dataset_by_node``)
and by DataLoader worker (HF datasets splits a node's shards across workers
at iteration time). Decoded records match the synthetic fixtures in
``synthetic.py``, so ``ObjectSequencer`` consumes either interchangeably.

Train/val assignment happens at write time via :func:`assign_split`: whole
coarse HEALPix tiles go to one split, so the sets are spatially disjoint and
near-duplicates cannot leak across.
"""

import hashlib
from pathlib import Path

import numpy as np
import torch
from datasets import Array3D, Dataset, Features, Sequence, Value, load_dataset
from datasets.distributed import split_dataset_by_node

IMAGE_SHAPE = (3, 152, 152)
SPECTRUM_LENGTH = 7781
N_BANDS = IMAGE_SHAPE[0]

# Canonical shard schema. Verified against the HATS ``_common_metadata`` of
# both catalogs (2026-07-07): the image struct uses ``band`` with per-band
# ``psf_fwhm``/``scale`` lists, and DESI's ``ZWARN`` is a bool. ``spectrum``,
# ``Z``, ``ZERR``, ``ZWARN`` and ``match_dist_arcsec`` are null for the ~13/14
# of objects without a DESI match. Scalars beyond the model inputs (``z_spec``,
# ``Z``, ``ebv``, ``flux_*``) ride along for the Phase 4 linear probes.
PILOT_FEATURES = Features(
    {
        "object_id": Value("string"),
        "ra": Value("float64"),
        "dec": Value("float64"),
        "_healpix_29": Value("int64"),
        "image": {
            "flux": Array3D(IMAGE_SHAPE, "float32"),
            "band": Sequence(Value("string")),
            "psf_fwhm": Sequence(Value("float32")),
            "scale": Sequence(Value("float32")),
        },
        "ebv": Value("float32"),
        "flux_g": Value("float32"),
        "flux_r": Value("float32"),
        "flux_z": Value("float32"),
        "z_spec": Value("float32"),
        "spectrum": {
            "flux": Sequence(Value("float32")),
            "lambda": Sequence(Value("float32")),
            "ivar": Sequence(Value("float32")),
            "lsf_sigma": Sequence(Value("float32")),
            "mask": Sequence(Value("bool")),
        },
        "Z": Value("float32"),
        "ZERR": Value("float32"),
        "ZWARN": Value("bool"),
        "match_dist_arcsec": Value("float32"),
    }
)

_SCALAR_KEYS = ("ebv", "flux_g", "flux_r", "flux_z", "z_spec", "Z", "ZERR")


def _as_band_list(value, dtype=np.float32) -> list:
    """Coerce a per-band field (scalar or length-3 sequence) to a 3-list."""
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        arr = np.repeat(arr, N_BANDS)
    return arr.tolist()


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


def normalize_record(rec: dict) -> dict:
    """Coerce a raw record (synthetic fixture or crossmatch row) to PILOT_FEATURES.

    Tolerates the schema drift between sources: ``bands`` vs ``band``, scalar
    vs per-band ``psf_fwhm``/``scale``, missing probe scalars.
    """
    image = rec["image"]
    out = {
        "object_id": str(rec["object_id"]),
        "ra": float(rec["ra"]),
        "dec": float(rec["dec"]),
        "_healpix_29": int(rec["_healpix_29"]),
        "image": {
            "flux": _image_flux(image["flux"]),
            "band": [str(b) for b in image.get("band", image.get("bands", []))],
            "psf_fwhm": _as_band_list(image.get("psf_fwhm", np.nan)),
            "scale": _as_band_list(image.get("scale", np.nan)),
        },
    }
    for key in _SCALAR_KEYS:
        value = rec.get(key)
        out[key] = None if value is None or np.isnan(value) else float(value)
    zwarn = rec.get("ZWARN")
    out["ZWARN"] = None if zwarn is None else bool(zwarn)
    dist = rec.get("match_dist_arcsec")
    out["match_dist_arcsec"] = None if dist is None or np.isnan(dist) else float(dist)

    spec = rec.get("spectrum")
    if spec is None:
        out["spectrum"] = None
    else:
        flux = np.asarray(spec["flux"], dtype=np.float32)
        if flux.shape != (SPECTRUM_LENGTH,):
            raise ValueError(
                f"spectrum has shape {flux.shape}, expected ({SPECTRUM_LENGTH},)"
            )
        out["spectrum"] = {
            "flux": flux,
            "lambda": np.asarray(spec["lambda"], dtype=np.float32),
            "ivar": np.asarray(spec["ivar"], dtype=np.float32),
            "lsf_sigma": np.asarray(spec["lsf_sigma"], dtype=np.float32),
            "mask": np.asarray(spec["mask"], dtype=bool),
        }
    return out


def _spectrum_is_null(spec) -> bool:
    if spec is None:
        return True
    flux = spec.get("flux")
    return flux is None or len(flux) == 0


def decode_record(example: dict) -> dict:
    """Shard example -> record dict shaped like ``synthetic.make_record`` output.

    Image flux comes back as float32 (3, 152, 152); an unmatched object's
    ``spectrum`` key is dropped entirely (the ``ObjectSequencer`` convention
    for a missing modality).
    """
    rec = dict(example)
    image = dict(example["image"])
    image["flux"] = np.asarray(image["flux"], dtype=np.float32)
    rec["image"] = image
    rec["object_id"] = str(rec["object_id"])
    spec = rec.get("spectrum")
    if _spectrum_is_null(spec):
        rec.pop("spectrum", None)
    else:
        rec["spectrum"] = {
            key: np.asarray(value, dtype=bool if key == "mask" else np.float32)
            for key, value in spec.items()
        }
    return rec


def assign_split(
    healpix_29: int, *, val_fraction: float = 0.02, order: int = 7, salt: int = 0
) -> str:
    """Deterministic, spatially coherent train/val assignment.

    Hashes the order-``order`` ancestor of the nested ``_healpix_29`` index
    (drop ``2*(29-order)`` bits), so entire ~0.2 deg^2 tiles (at the default
    order 7) land in one split. Crossmatched pairs sit well inside a tile
    (1 arcsec vs ~0.4 deg), so no pair straddles the split boundary.
    """
    pixel = int(healpix_29) >> (2 * (29 - order))
    digest = hashlib.sha256(f"{salt}:{pixel}".encode()).digest()
    fraction = int.from_bytes(digest[:8], "big") / 2**64
    return "val" if fraction < val_fraction else "train"


def write_shard(records: list[dict], path: str | Path) -> None:
    """Normalize records and write one parquet shard atomically (tmp + rename).

    The ``.tmp`` staging name does not match the ``*.parquet`` glob
    :class:`MMUIterableDataset` reads, so a crash mid-write can never leave a
    half-shard in the training stream.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Dataset.from_list(
        [normalize_record(r) for r in records], features=PILOT_FEATURES
    ).to_parquet(tmp)
    tmp.rename(path)


class MMUIterableDataset(torch.utils.data.IterableDataset):
    """Stream decoded pilot records from local parquet shards.

    A thin wrapper over ``datasets`` streaming that yields
    :func:`decode_record` output ready for ``ObjectSequencer``. Sharding:

    - across DP ranks via ``split_dataset_by_node`` (shard-level when
      ``n_shards % world_size == 0``, else example-stride);
    - across DataLoader workers by HF datasets itself — its ``__iter__``
      checks ``torch.utils.data.get_worker_info()`` and gives each worker a
      disjoint subset of this rank's shards, so keep
      ``num_workers <= n_shards // world_size``.

    Shuffling is deliberately NOT ``datasets.IterableDataset.shuffle``: in
    datasets 5.x that wraps the stream in a cycling iterable with
    ``num_shards == 1``, which destroys shard-level rank/worker splitting.
    Instead the shard order is shuffled once per (seed, epoch) — identically
    on every rank, so the node split stays consistent — and each worker runs
    a seeded buffer shuffle over its own stream.

    ``state_dict``/``load_state_dict`` pass through to the underlying
    stateful iterable for Phase 4 checkpoint-resume (like HF's own shuffle,
    the in-flight buffer is not part of the state).
    """

    def __init__(
        self,
        data_files: str | Path | list,
        *,
        rank: int = 0,
        world_size: int = 1,
        shuffle_buffer_size: int = 0,
        seed: int = 0,
    ):
        if isinstance(data_files, (str, Path)):
            root = Path(data_files)
            data_files = sorted(root.glob("*.parquet")) if root.is_dir() else [root]
        if not data_files:
            raise ValueError("no parquet shards found")
        self._files = [str(p) for p in data_files]
        self._rank = rank
        self._world_size = world_size
        self._buffer_size = shuffle_buffer_size
        self._seed = seed
        self._epoch = 0
        self._load()

    def _load(self) -> None:
        files = list(self._files)
        if self._buffer_size:
            np.random.default_rng((self._seed, self._epoch)).shuffle(files)
        ds = load_dataset(
            "parquet", data_files=files, split="train", streaming=True
        )
        if self._world_size > 1:
            ds = split_dataset_by_node(
                ds, rank=self._rank, world_size=self._world_size
            )
        self._hf = ds.with_format("numpy")

    @property
    def n_shards(self) -> int:
        return self._hf.n_shards

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch
        self._load()  # reshuffle the shard order for the new epoch

    def state_dict(self) -> dict:
        return self._hf.state_dict()

    def load_state_dict(self, state: dict) -> None:
        self._hf.load_state_dict(state)

    def __iter__(self):
        stream = (decode_record(example) for example in self._hf)
        if not self._buffer_size:
            yield from stream
            return
        worker = torch.utils.data.get_worker_info()
        rng = np.random.default_rng(
            (self._seed, self._epoch, self._rank, worker.id if worker else 0)
        )
        buffer: list[dict] = []
        for record in stream:
            if len(buffer) < self._buffer_size:
                buffer.append(record)
                continue
            j = int(rng.integers(self._buffer_size))
            yield buffer[j]
            buffer[j] = record
        for j in rng.permutation(len(buffer)):
            yield buffer[j]

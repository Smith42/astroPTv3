"""Adapter from the LSDB LegacySurvey stream to nanotron micro-batches.

Each DataLoader worker opens the fixed catalog and owns one synchronous
``lsdb.streams.InfiniteStream``. Items yielded by this module are already
whole micro-batches, flattened because nanotron only moves top-level tensors
to the device.
"""

from __future__ import annotations

import gc
import math
import time
import zlib
from collections.abc import Mapping
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import httpx
import lsdb
import numpy as np
import torch
from lsdb.streams.catalog_streams import InfiniteStream

from ..configuration_astropt3 import AstroPT3Config
from .band_registry import _DIV_FACTOR
from .packing import ObjectSequencer, PackedCollator, span_order
from .spectral import _DIV_FACTOR as _SPECTRA_DIV_FACTOR
from .telemetry import install_byte_probe, instrument

LEGACY_CATALOG = "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north"
IMAGE_SHAPE = (3, 152, 152)
_MAX_NET_RETRIES = 60
_MAX_NET_RETRY_WAIT = 120
_MAX_REPLICA_ATTEMPTS = 32


def hf_config_from_modalities(
    modalities, tokeniser: str = "affine", **extra
) -> AstroPT3Config:
    """Build the HF-side config used by the shared sequencer and collator."""
    return AstroPT3Config(
        modalities=[dict(modality) for modality in modalities],
        tokeniser=tokeniser,
        **extra,
    )


def flatten_packed_batch(batch: dict, config: AstroPT3Config, seq_len: int) -> dict:
    """Convert ``PackedCollator`` output to nanotron's flat input contract."""
    registry = config.modality_registry()
    batch_size = batch["input_ids"].shape[0]
    flat = {
        "input_ids": batch["input_ids"],
        "position_ids": batch["position_ids"],
    }
    for name in registry.names():
        modality = registry.get_config(name)
        if name in batch["modality_masks"]:
            flat[f"{name}_mask"] = batch["modality_masks"][name]
            flat[f"{name}_values"] = batch["modality_values"][name]
            flat[f"{name}_positions"] = batch["modality_positions"][name]
        else:
            flat[f"{name}_mask"] = torch.zeros(
                (batch_size, seq_len), dtype=torch.bool
            )
            flat[f"{name}_values"] = torch.empty(
                (0, modality.input_size), dtype=torch.float32
            )
            if modality.pos_type == "index":
                flat[f"{name}_positions"] = torch.empty((0,), dtype=torch.long)
            else:
                flat[f"{name}_positions"] = torch.empty(
                    (0, modality.pos_input_size), dtype=torch.float32
                )
    return flat


def regroup_micro_batch(flat: dict, names) -> dict:
    """Convert a flat nanotron micro-batch to HF model keyword arguments."""
    return {
        "input_ids": flat["input_ids"],
        "position_ids": flat["position_ids"],
        "modality_values": {
            name: flat[f"{name}_values"]
            for name in names
            if flat[f"{name}_values"].shape[0]
        },
        "modality_masks": {
            name: flat[f"{name}_mask"]
            for name in names
            if flat[f"{name}_mask"].any()
        },
        "modality_positions": {
            name: flat[f"{name}_positions"]
            for name in names
            if flat[f"{name}_values"].shape[0]
        },
    }


def _as_mapping(value: Any) -> Mapping:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "as_py"):
        value = value.as_py()
        if isinstance(value, Mapping):
            return value
    if hasattr(value, "to_dict"):
        try:
            value = value.to_dict(orient="list")
        except TypeError:
            value = value.to_dict()
        if isinstance(value, Mapping):
            return value
    raise ValueError(f"image payload has unsupported type {type(value).__name__}")


def _stack_nested(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        return np.stack([_stack_nested(item) for item in array])
    return array


def _finite_scalar(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def decode_legacy_row(row: Mapping) -> dict:
    """Decode one LegacySurvey row to the ``ObjectSequencer`` contract.

    Duck-typed: any mapping with ``image``/scalar columns works (pandas
    ``Series``, plain dicts, test fixtures).
    """
    image = _as_mapping(row.get("image"))
    bands = [str(band) for band in image.get("band", ())]
    flux = _stack_nested(image.get("flux")).astype(np.float32, copy=False)
    if flux.shape != IMAGE_SHAPE:
        raise ValueError(f"image flux has shape {flux.shape}, expected {IMAGE_SHAPE}")
    if len(bands) != IMAGE_SHAPE[0]:
        raise ValueError(f"image has {len(bands)} bands, expected {IMAGE_SHAPE[0]}")

    object_id = row.get("object_id")
    if object_id is None:
        raise ValueError("LegacySurvey row has no object_id")
    record = {
        "object_id": str(object_id),
        "image": {"flux": flux, "band": bands},
    }
    for key, value in row.items():
        if key in {"object_id", "image"}:
            continue
        number = _finite_scalar(value)
        if number is not None:
            record[str(key)] = number

    fwhm = image.get("psf_fwhm")
    if fwhm is not None and len(fwhm) == len(bands):
        for band, value in zip(bands, fwhm):
            number = _finite_scalar(value)
            if number is not None and 0 < number < 5:
                record[f"psf_fwhm_{band}"] = number
    return record


def _catalog_columns(config: AstroPT3Config) -> list[str]:
    """Project the fixed catalog to active Legacy image/scalar fields only."""
    columns = {"object_id"}
    has_image = False
    for name in config.modality_registry().names():
        modality = config.modality_registry().get_config(name)
        if modality.source != "legacy":
            continue
        if modality.family == "image":
            if tuple(modality.record_keys) != ("image",):
                raise ValueError(
                    f"Legacy image modality {name!r} must use record key 'image'"
                )
            has_image = True
            columns.add("image")
        elif modality.family == "scalar":
            for key in modality.record_keys:
                columns.add("image" if key.startswith("psf_fwhm_") else key)
    if not has_image:
        raise ValueError("LSDB training requires an active Legacy image modality")
    return ["object_id", "image", *sorted(columns - {"object_id", "image"})]


def consumer_seed(seed: int, rank: int, worker: int, retry_generation: int) -> int:
    """Stable independent seed for one DP-rank/worker/fresh-stream consumer."""
    return zlib.crc32(f"{seed}:{rank}:{worker}:{retry_generation}".encode())


def _retryable(error: Exception) -> bool:
    if isinstance(error, (httpx.HTTPError, OSError, TimeoutError)):
        return True
    return isinstance(error, RuntimeError) and "client has been closed" in str(error)


def _lsdb_version() -> str:
    try:
        return version("lsdb")
    except PackageNotFoundError:
        return "unknown"


def _log_provenance(catalog, columns: list[str], rank: int, worker: int) -> None:
    info = getattr(getattr(catalog, "hc_structure", None), "catalog_info", None)
    metadata = {}
    for key in ("catalog_name", "catalog_type", "hats_builder", "hats_version"):
        value = getattr(info, key, None)
        if value is not None:
            metadata[key] = str(value)
    suffix = f" provenance={metadata}" if metadata else ""
    print(
        f"[data] dp={rank} worker={worker} lsdb={_lsdb_version()} "
        f"catalog={LEGACY_CATALOG} columns={columns}{suffix}",
        flush=True,
    )


class PackedMicroBatches(torch.utils.data.IterableDataset):
    """Endless LSDB records packed into fixed-shape nanotron micro-batches."""

    def __init__(
        self,
        config: AstroPT3Config,
        micro_batch_size: int,
        seq_len: int,
        *,
        rank: int = 0,
        seed: int = 0,
        ar_replicas: int = 1,
        replica_placement: str = "decorrelated",
    ):
        super().__init__()
        if ar_replicas < 1:
            raise ValueError(f"ar_replicas must be >= 1, got {ar_replicas}")
        if replica_placement not in ("decorrelated", "adjacent"):
            raise ValueError(
                "replica_placement must be 'decorrelated' or 'adjacent', "
                f"got {replica_placement!r}"
            )
        if replica_placement == "decorrelated" and ar_replicas > micro_batch_size:
            raise ValueError(
                f"ar_replicas={ar_replicas} exceeds micro_batch_size="
                f"{micro_batch_size}; each replica needs its own packed row"
            )
        self.config = config
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.seed = seed
        self.ar_replicas = ar_replicas
        self.replica_placement = replica_placement
        self.columns = _catalog_columns(config)
        self._epoch = 0  # InfiniteStream partition-draw nonce for span ordering
        self.sequencer = ObjectSequencer(config)
        self.collator = PackedCollator(config, seq_len=seq_len)

    def _open_records(self, worker):
        worker_id = worker.id if worker else 0
        retry_generation = 0
        consecutive_failures = 0
        draw = 0
        while True:
            catalog = stream = iterator = None
            try:
                catalog = getattr(lsdb, "open_catalog")(
                    LEGACY_CATALOG, columns=self.columns
                )
                _log_provenance(catalog, self.columns, self.rank, worker_id)
                stream = InfiniteStream(
                    catalog,
                    client=None,
                    partitions_per_chunk=1,
                    seed=consumer_seed(
                        self.seed, self.rank, worker_id, retry_generation
                    ),
                )
                iterator = iter(stream)
            except Exception as error:
                if not _retryable(error):
                    raise
                consecutive_failures += 1
                if consecutive_failures > _MAX_NET_RETRIES:
                    raise
                wait = min(
                    5 * 2 ** (consecutive_failures - 1), _MAX_NET_RETRY_WAIT
                )
                print(
                    f"[data] {type(error).__name__}: opening a fresh LSDB stream "
                    f"in {wait}s (retry {consecutive_failures}/{_MAX_NET_RETRIES})",
                    flush=True,
                )
                time.sleep(wait)
                retry_generation += 1
                continue

            if iterator is None:
                raise AssertionError("LSDB stream opened without an iterator")
            active_iterator: Any = iterator
            reopen = False
            while not reopen:
                try:
                    frame: Any = next(active_iterator)
                except StopIteration:
                    reopen = True
                    retry_generation += 1
                    continue
                except Exception as error:
                    if not _retryable(error):
                        raise
                    consecutive_failures += 1
                    if consecutive_failures > _MAX_NET_RETRIES:
                        raise
                    wait = min(
                        5 * 2 ** (consecutive_failures - 1), _MAX_NET_RETRY_WAIT
                    )
                    print(
                        f"[data] {type(error).__name__}: discarding the LSDB "
                        f"iterator and opening a fresh stream in {wait}s "
                        f"(retry {consecutive_failures}/{_MAX_NET_RETRIES})",
                        flush=True,
                    )
                    active_iterator = iter(())
                    iterator = stream = catalog = None
                    gc.collect()
                    time.sleep(wait)
                    retry_generation += 1
                    reopen = True
                    continue

                consecutive_failures = 0
                self._epoch = draw
                draw += 1
                for _, row in frame.iterrows():
                    yield decode_legacy_row(row)

    def _replica_objects(self, record: dict) -> list:
        """Build the base sequence and distinct autoregressive reorderings."""
        base_id = str(record.get("object_id", ""))
        first = self.sequencer.build(record, epoch=self._epoch)
        names = sorted(first.order)
        if self.ar_replicas == 1 or len(names) < 2:
            return [first]

        wanted = min(self.ar_replicas, math.factorial(len(names)))
        objects, seen = [first], {first.order}
        attempt = 1
        while len(objects) < wanted and attempt <= _MAX_REPLICA_ATTEMPTS:
            replica_id = f"{base_id}#{attempt}"
            order = tuple(span_order(names, replica_id, self._epoch))
            attempt += 1
            if order in seen:
                continue
            seen.add(order)
            objects.append(
                self.sequencer.build(
                    {**record, "object_id": replica_id},
                    epoch=self._epoch,
                    modality_order=list(order),
                )
            )
        return objects

    def _place(self, objects: list, used: list[int]) -> list[int] | None:
        """Assign a record's base sequence and replicas to packed rows."""
        order = sorted(range(len(used)), key=lambda row: (used[row], row))
        if self.replica_placement == "adjacent":
            span = sum(len(obj) for obj in objects)
            for row_index in order:
                if used[row_index] + span <= self.seq_len:
                    return [row_index] * len(objects)
            return None
        chosen: list[int] = []
        for obj in objects:
            for row_index in order:
                if row_index not in chosen and used[row_index] + len(obj) <= self.seq_len:
                    chosen.append(row_index)
                    break
            else:
                return None
        return chosen

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        install_byte_probe(self.rank, worker.id if worker else 0)
        records = self._open_records(worker)
        rows: list[list] = [[] for _ in range(self.micro_batch_size)]
        used = [0] * self.micro_batch_size

        for record in records:
            objects = self._replica_objects(record)
            for obj in objects:
                if len(obj) > self.seq_len:
                    raise ValueError(
                        f"object of length {len(obj)} exceeds seq_len {self.seq_len}"
                    )
            placement = self._place(objects, used)
            if placement is None:
                batch = self.collator.collate_rows(rows)
                expected_shape = (self.micro_batch_size, self.seq_len)
                if batch["input_ids"].shape != expected_shape:
                    raise AssertionError(
                        f"packing mismatch: {batch['input_ids'].shape} != {expected_shape}"
                    )
                yield flatten_packed_batch(batch, self.config, self.seq_len)
                rows = [[] for _ in range(self.micro_batch_size)]
                used = [0] * self.micro_batch_size
                placement = self._place(objects, used)
                if placement is None:
                    raise AssertionError(
                        f"record {record.get('object_id')!r} cannot be placed "
                        f"into {self.micro_batch_size} rows x {self.seq_len} tokens"
                    )
            for obj, row_index in zip(objects, placement):
                rows[row_index].append(obj)
                used[row_index] += len(obj)


def build_astropt3_dataloader(
    dataset_args,
    model_config,
    micro_batch_size: int,
    sequence_length: int,
    dp_rank: int,
    dp_size: int,
    num_workers: int = 0,
    seed: int = 0,
    multiprocessing_context: str | None = None,
) -> Any:
    """Build the plain DataLoader used by nanotron's astropt3 dataset type."""
    config = hf_config_from_modalities(
        model_config.modalities,
        getattr(model_config, "tokeniser", "affine"),
        **{
            field: getattr(model_config, field, default)
            for field, default in [
                ("jetformer_flow_steps", 4),
                ("jetformer_flow_hidden", 128),
                ("jetformer_gmm_k", 4),
                ("jetformer_noise_max", 0.1),
                ("jetformer_noise_min", 0.0),
                ("scalar_gmm_k", 5),
                ("image_norm_divisor", _DIV_FACTOR),
                ("spectra_norm_divisor", _SPECTRA_DIV_FACTOR),
                ("spiral", True),
            ]
        },
    )
    dataset = PackedMicroBatches(
        config,
        micro_batch_size,
        sequence_length,
        rank=dp_rank,
        seed=seed,
        ar_replicas=getattr(dataset_args, "ar_replicas", 1) or 1,
        replica_placement=getattr(dataset_args, "replica_placement", None)
        or "decorrelated",
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context=multiprocessing_context,
    )
    return instrument(loader)

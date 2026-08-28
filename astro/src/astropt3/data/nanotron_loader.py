"""Adapter: astro data pipeline -> nanotron ``astropt3_streaming`` micro-batches.

Turns the record sources (:func:`mmu_stream.streaming.open_stream` over the
live MMU catalogs, or the synthetic stream) into an endless stream of
fixed-shape micro-batch dicts for the nanotron fork's ``AstroPT3ForTraining``:

- ``input_ids``      long  [micro_batch_size, sequence_length]
- ``position_ids``   long  [micro_batch_size, sequence_length]  (restart at 0
  per object; pads at 0 — the packed-document boundary signal)
- per modality ``m`` (all modalities always present, zero-length if absent
  from the micro-batch):
  - ``{m}_values``    float32 [n_m, input_size]   (row-major flattened)
  - ``{m}_positions`` long [n_m] or float32 [n_m, pos_input_size]
  - ``{m}_mask``      bool  [micro_batch_size, sequence_length]

The dict is flat because nanotron's device mover
(``nanotron.data.dataloader.sanity_check_dataloader``) only transfers
top-level tensors. This module must stay importable WITHOUT nanotron: the CPU
test suite exercises it against the HF model, and only
``nanotron/run_train.py`` calls :func:`build_astropt3_dataloader`
(``dataset_args`` is duck-typed, never isinstance-checked).

Sharding: the object stream is split by DP rank (identical within a TP
group — nanotron passes the dp process-group rank/size) inside
``streaming.owned_by_rank``, which deals whole partitions; the split across
DataLoader workers is then done by ``datasets`` itself — its ``_iter_pytorch``
shards the stream per worker whenever it detects one. Splitting manually by
``world_size x num_workers`` on top of that DOUBLE-shards and clamps the
loader to one worker, so don't. ``datasets`` only WARNS when a rank owns
fewer partitions than it has workers, so :meth:`_mmu_records` raises instead.
(The synthetic stream, which has no datasets machinery underneath, still
strides over record indices by ``world_size x num_workers`` itself.)

Checkpoint-resume (Phase 4): ``state_dict()`` returns the stream position at
the START of the current partial packing row — everything already drawn into
that row has not been trained on, so resume re-draws it and continues with
exactly the micro-batch sequence an uninterrupted run would have produced.
The synthetic state is a record counter; the MMU state is the ``datasets``
generator's own ``state_dict()`` (shard index + example index) alongside the
epoch, tagged with ``source_assembly`` so a state written by a different
record ORDER is rejected rather than resumed onto the wrong row. Both are
exact: ADR 0006 §4 budgeted for replaying the in-flight partition, but the
row-group cursor makes the offset exact, so the no-replay guarantee survives
streaming.

With ``num_workers == 0`` the dataset object itself carries the state. With
``num_workers > 0`` each DataLoader worker's dataset copy keeps its own
state, and :func:`build_astropt3_dataloader` returns a torchdata
``StatefulDataLoader`` whose ``state_dict()`` gathers the per-worker
snapshots consistent with the last micro-batch actually yielded to the
caller (worker-prefetched batches are accounted for by torchdata). The
trainer captures either kind of loader through :func:`loader_state_dict`.
"""

from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any, cast

import httpx
import torch
from mmu_stream.streaming import assembly_and_revisions

from ..configuration_astropt3 import AstroPT3Config
from .band_registry import _DIV_FACTOR
from .packing import (
    IMAGE_CROP,
    SPAN_ORDER_VERSION,
    ObjectSequencer,
    PackedCollator,
    span_order,
)
from .spectral import _DIV_FACTOR as _SPECTRA_DIV_FACTOR
from .synthetic import make_record
from .telemetry import install_byte_probe, instrument

MMU_ROOT = "mmu"
SYNTHETIC_ROOT = "synthetic"
STATE_FILE_TEMPLATE = "dp_{rank}.pt"
STATE_SUBDIR = "dataset_state"
LOADER_STATE_FORMAT = "stateful_dataloader"
# transient hub/network failures (an overloaded DNS resolver returns "Name or
# service not known" for seconds at a time) must not kill a multi-day run:
# rebuild the stream from the last per-record snapshot instead. The counter
# resets on any successful draw, so the budget only burns during a SUSTAINED
# outage: 60 x 120s cap ≈ 2h, after which the network is truly down and the
# run should die loudly rather than hang looking like a stall
_MAX_NET_RETRIES = 60
_MAX_NET_RETRY_WAIT = 120

# ADR 0014 §7a/7b: bumped whenever the replica SELECTION or PLACEMENT rule
# changes, since either changes the emitted sequence stream without changing
# record order. Folded into the §5 fingerprint below.
REPLICA_POLICY_VERSION = "distinct_orders_decorrelated_v1"
# candidate replica orders are cheap to test (span_order, no tokenisation) and
# a repeat is just skipped, so a small cap suffices: at the hardest case,
# 2 spans, each draw is a coin flip, and 32 tries fail with p ~ 2e-10
_MAX_REPLICA_ATTEMPTS = 32


def sequence_fingerprint(
    *,
    assembly: str,
    revisions: dict,
    config: AstroPT3Config,
    seq_len: int,
    ar_replicas: int,
    replica_placement: str = "decorrelated",
) -> str:
    """ADR 0014 §5: a resume tag over the whole sequence-assembly policy.

    ``source_assembly`` alone protects record ORDER. A checkpoint saved at
    ``ar_replicas: 1`` and resumed at 2 passed that check while silently
    changing the emitted sequence stream — as would a crop change, a
    per-band switch (§8), or a new span-order algorithm. Fingerprinting all
    of them means a mismatched stream state is REJECTED (weights still load,
    exactly like an assembly bump) instead of resuming onto the wrong
    sequences and contaminating an A/B.

    Deliberately excluded: normalization divisors and the spiral flag. They
    change token VALUES, not the sequence stream, and are already
    checkpoint-incompatible — a weights load is what stops you there.
    """
    registry = config.modality_registry()
    payload = {
        "assembly": assembly,
        "revisions": dict(sorted(revisions.items())),
        "modalities": [registry.get_config(n).to_dict() for n in registry.names()],
        "image_crop": IMAGE_CROP,
        "span_order": SPAN_ORDER_VERSION,
        "replica_policy": REPLICA_POLICY_VERSION,
        "replica_placement": replica_placement,
        "ar_replicas": ar_replicas,
        "seq_len": seq_len,
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:12]
    return f"{assembly}:{digest}"


def hf_config_from_modalities(
    modalities, tokeniser: str = "affine", **extra
) -> AstroPT3Config:
    """Build the (tiny) HF-side config the sequencer/collator machinery wants.

    ``modalities`` may come from either implementation's config — both carry
    the same list of dicts. ``extra`` passes tokeniser-specific fields
    (e.g. the ``jetformer_*`` knobs) straight through to ``AstroPT3Config``.
    """
    return AstroPT3Config(
        modalities=[dict(m) for m in modalities], tokeniser=tokeniser, **extra
    )


def flatten_packed_batch(batch: dict, config: AstroPT3Config, seq_len: int) -> dict:
    """PackedCollator output -> flat nanotron micro-batch dict.

    Modalities absent from the batch get correctly-typed zero-length tensors
    so the model's forward signature (and DDP's used-parameter accounting)
    stays fixed.
    """
    registry = config.modality_registry()
    b = batch["input_ids"].shape[0]
    flat = {
        "input_ids": batch["input_ids"],
        "position_ids": batch["position_ids"],
    }
    for name in registry.names():
        mod = registry.get_config(name)
        if name in batch["modality_masks"]:
            flat[f"{name}_mask"] = batch["modality_masks"][name]
            flat[f"{name}_values"] = batch["modality_values"][name]
            flat[f"{name}_positions"] = batch["modality_positions"][name]
        else:
            flat[f"{name}_mask"] = torch.zeros((b, seq_len), dtype=torch.bool)
            flat[f"{name}_values"] = torch.empty(
                (0, mod.input_size), dtype=torch.float32
            )
            if mod.pos_type == "index":
                flat[f"{name}_positions"] = torch.empty((0,), dtype=torch.long)
            else:
                flat[f"{name}_positions"] = torch.empty(
                    (0, mod.pos_input_size), dtype=torch.float32
                )
    return flat


def regroup_micro_batch(flat: dict, names) -> dict:
    """Flat nanotron micro-batch -> HF ``AstroPT3Model`` forward kwargs."""
    return {
        "input_ids": flat["input_ids"],
        "position_ids": flat["position_ids"],
        "modality_values": {
            n: flat[f"{n}_values"] for n in names if flat[f"{n}_values"].shape[0]
        },
        "modality_masks": {
            n: flat[f"{n}_mask"] for n in names if flat[f"{n}_mask"].any()
        },
        "modality_positions": {
            n: flat[f"{n}_positions"] for n in names if flat[f"{n}_values"].shape[0]
        },
    }


class PackedMicroBatches(torch.utils.data.IterableDataset):
    """Endless stream of fixed-shape nanotron micro-batches.

    Objects are packed greedily into rows of ``seq_len`` (never split), rows
    are grouped ``micro_batch_size`` at a time, and each group is collated by
    the shared :class:`PackedCollator` — the greedy repack of whole rows is
    deterministic, so the collator reproduces exactly the grouped rows.

    Use with ``DataLoader(batch_size=None)``; each item IS a micro-batch.

    ``object_id_log`` appends one ``object_id`` line per object as its
    micro-batch is YIELDED (a partial row lost to a kill is never logged),
    to ``{object_id_log}.dp{rank}`` — the no-replay audit trail for the
    Phase 4 kill/resume gate.
    """

    def __init__(
        self,
        config: AstroPT3Config,
        micro_batch_size: int,
        seq_len: int,
        *,
        data_root: str = SYNTHETIC_ROOT,
        match_index: str | None = None,
        synthetic_image_only_fraction: float = 0.3,
        synthetic_spectrum_only_fraction: float = 0.0,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        split: str = "train",
        object_id_log: str | Path | None = None,
        stateful: bool = True,
        ar_replicas: int = 1,
        replica_placement: str = "decorrelated",
    ):
        super().__init__()
        self.config = config
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len
        self.data_root = str(data_root)
        if self.data_root not in (SYNTHETIC_ROOT, MMU_ROOT):
            # a stale path to the deleted local reshard must fail loudly, not
            # silently stream something else (ADR 0006 §7)
            raise ValueError(
                f"data_root must be {SYNTHETIC_ROOT!r} or {MMU_ROOT!r}, got "
                f"{self.data_root!r}; the local parquet corpus was removed by "
                "ADR 0006 — the MMU catalogs are streamed live"
            )
        # The MMU branch has one assembly and requires its precomputed index.
        self.match_index = match_index
        if self.data_root == MMU_ROOT:
            assembly, revisions = assembly_and_revisions(match_index)
        else:
            assembly, revisions = SYNTHETIC_ROOT, {}
        self.synthetic_image_only_fraction = synthetic_image_only_fraction
        self.synthetic_spectrum_only_fraction = synthetic_spectrum_only_fraction
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.split = split
        self.object_id_log = None if object_id_log is None else str(object_id_log)
        if ar_replicas < 1:
            raise ValueError(f"ar_replicas must be >= 1, got {ar_replicas}")
        if replica_placement not in ("decorrelated", "adjacent"):
            raise ValueError(
                f"replica_placement must be 'decorrelated' or 'adjacent', "
                f"got {replica_placement!r}"
            )
        if replica_placement == "decorrelated" and ar_replicas > micro_batch_size:
            # ADR 0014 §7b puts each replica in its own packed row, so more
            # replicas than rows can never be placed — fail here rather than
            # deadlock the packing loop
            raise ValueError(
                f"ar_replicas={ar_replicas} exceeds micro_batch_size="
                f"{micro_batch_size}; each replica needs its own packed row"
            )
        self.ar_replicas = ar_replicas
        self.replica_placement = replica_placement
        # ADR 0014 §5: the resume tag is the whole sequence-assembly policy,
        # not just record order — built here so it covers ar_replicas AND the
        # replica-separation policy (an A/B between the two placements must
        # not be resumable across arms)
        self._source_assembly_tag = sequence_fingerprint(
            assembly=assembly,
            revisions=revisions,
            config=config,
            seq_len=seq_len,
            ar_replicas=ar_replicas,
            replica_placement=replica_placement,
        )
        self._stateful = stateful
        self._resume_state: dict | None = None  # applied on next __iter__
        self._ckpt_state: dict | None = None  # updated at every yield
        self._stream = None  # the live datasets stream, once iteration starts
        self._epoch = 0

        self.sequencer = ObjectSequencer(config)
        self.collator = PackedCollator(config, seq_len=seq_len)

    # -- checkpoint state ---------------------------------------------------

    @property
    def _source_assembly(self) -> str:
        return self._source_assembly_tag

    def state_dict(self) -> dict | None:
        """Stream position at the start of the current partial row.

        Returns None only from the never-iterated main-process copy of a
        plain ``num_workers > 0`` DataLoader (``stateful=False`` and nothing
        consumed): the real state lives in the worker copies, which a
        StatefulDataLoader collects through this same method.
        """
        if self._ckpt_state is not None:
            return dict(self._ckpt_state)
        if self._resume_state is not None:
            return dict(self._resume_state)
        if not self._stateful:
            return None
        return {
            "records": 0,
            "epoch": 0,
            "stream_state": None,
            "data_root": self.data_root,
            "source_assembly": self._source_assembly,
        }

    def load_state_dict(self, state: dict | None) -> None:
        if state is None:  # a worker snapshotted before its first yield
            return
        if state.get("data_root") not in (None, self.data_root):
            raise ValueError(
                f"dataset state was saved for data_root={state['data_root']!r}, "
                f"this stream reads {self.data_root!r}"
            )
        saved_assembly = state.get("source_assembly")
        if saved_assembly != self._source_assembly:
            raise ValueError(
                f"dataset state uses source_assembly={saved_assembly!r}; "
                f"this branch requires {self._source_assembly!r}. Start a new "
                "run rather than restoring a checkpoint from another corpus."
            )
        self._resume_state = dict(state)

    def _snapshot(self, records: int) -> dict:
        """State AFTER ``records`` records have been consumed by the packer."""
        return {
            "records": records,
            "epoch": self._epoch,
            "stream_state": None if self._stream is None else self._stream.state_dict(),
            "data_root": self.data_root,
            "source_assembly": self._source_assembly,
        }

    # -- record sources -----------------------------------------------------

    def _synthetic_records(self, start_count: int, worker):
        """Endless deterministic stream; index striding keeps ranks/workers disjoint."""
        n_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        offset = self.rank * n_workers + worker_id
        stride = self.world_size * n_workers
        for k in itertools.count(start_count):
            yield make_record(
                offset + k * stride,
                image_only_fraction=self.synthetic_image_only_fraction,
                spectrum_only_fraction=self.synthetic_spectrum_only_fraction,
            )

    def _mmu_records(self, start_epoch, stream_state, worker):
        """Repeat finite crossmatch-only epochs and restore their exact state."""
        import itertools

        from mmu_stream.streaming import open_stream

        for epoch in itertools.count(start_epoch):
            self._epoch = epoch  # seeds the ADR 0008 span shuffle
            stream = open_stream(
                split=self.split,
                seed=self.seed,
                epoch=epoch,
                shard=self.rank,
                num_shards=self.world_size,
                match_index=self.match_index,
            )
            # datasets splits this rank's shards across the loader workers and
            # only WARNS when it runs short ("Stopping N-M dataloader
            # workers"), so an over-subscribed run silently trains on a
            # fraction of its workers. The crossmatch-only corpus is ~165 cells
            # total, not the ~5.5k of the retired images source, so this binds
            # at pilot dp.
            if worker is not None and worker.num_workers > stream.n_shards:
                raise ValueError(
                    f"dp rank {self.rank} owns {stream.n_shards} crossmatch "
                    f"train partitions but num_loading_workers is "
                    f"{worker.num_workers} — reduce num_loading_workers to "
                    f"<= {stream.n_shards}, or reduce dp({self.world_size}) "
                    f"so each rank owns more partitions"
                )
            if epoch == start_epoch and stream_state is not None:
                stream.load_state_dict(stream_state)
            self._stream = stream
            iterator = iter(stream)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    break

    def _open_records(self, state, worker):
        """(Re)open the record source at ``state`` (None = fresh start)."""
        if self.data_root == SYNTHETIC_ROOT:
            return self._synthetic_records(state["records"] if state else 0, worker)
        return self._mmu_records(
            state["epoch"] if state else 0,
            state.get("stream_state") if state else None,
            worker,
        )

    # -- replay (ADR 0014 §7) -------------------------------------------------

    def _replica_objects(self, record: dict) -> list:
        """Build this record's sequences: replica 0 plus DISTINCT re-orderings.

        ``ar_replicas > 1`` re-emits the same downloaded record under a
        different ADR 0008 span order. The corpus is transfer-bound, so extra
        factorisations of a record already in memory are close to free, and
        every conditional among its spans gets trained rather than one sample
        of them.

        ADR 0014 §7a is what keeps that honest. Suffixing the object id
        reseeds the shuffle but does NOT guarantee a different permutation,
        and identical duplicates would raise MFU and ``E_AR`` while training
        on nothing new. So: one-span records get no replica (there is no
        other order), replicas are capped at the number of distinct
        permutations, and a candidate whose order repeats one already emitted
        is skipped. Candidates are rejected using :func:`span_order` alone —
        no tokenisation is paid for an order we then discard.

        Replica 0 keeps the original id (replicas get ``#n``), so
        ``object_id_log`` stays one unique line per emitted sequence and the
        no-replay audit still catches accidental duplication.
        """
        base_id = str(record.get("object_id", ""))
        first = self.sequencer.build(record, epoch=self._epoch)
        names = sorted(first.order)  # canonical input: a pure function of the SET
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
            record["object_id"] = replica_id
            objects.append(
                self.sequencer.build(
                    record, epoch=self._epoch, modality_order=list(order)
                )
            )
        record["object_id"] = base_id
        return objects

    def _place(self, objects: list, used: list[int]) -> list[int] | None:
        """Assign each of a record's sequences to a packed row.

        ADR 0014 §7b: under ``decorrelated`` (the default) a base object's
        replicas land in DIFFERENT rows — document masking prevents
        cross-attention but not gradient repetition within the batch.
        ``adjacent`` keeps them in one row, which is the pre-§7b behaviour
        and exists only so the B1 benchmark arm reproduces the measured
        3.1x result on the same footing.

        Emptiest row first (ties by index) keeps the rows balanced and makes
        the assignment a pure function of ``used``, which resume rebuilds
        exactly by replaying the batch from its saved start. Returns None
        when no assignment fits, which is the signal to close the batch.
        """
        order = sorted(range(len(used)), key=lambda b: (used[b], b))
        if self.replica_placement == "adjacent":
            span = sum(len(obj) for obj in objects)
            for row_index in order:
                if used[row_index] + span <= self.seq_len:
                    return [row_index] * len(objects)
            return None
        chosen: list[int] = []
        for obj in objects:
            for row_index in order:
                if row_index in chosen:
                    continue
                if used[row_index] + len(obj) <= self.seq_len:
                    chosen.append(row_index)
                    break
            else:
                return None
        return chosen

    # -- iteration ------------------------------------------------------------

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        # ADR 0014 §3: the byte probe must live where the fetches happen —
        # this runs in each loader worker process, so patch there.
        install_byte_probe(self.rank, worker.id if worker else 0)
        # Worker copies are always stateful: a StatefulDataLoader snapshots
        # them via state_dict() inside the worker process (under a plain
        # DataLoader the bookkeeping is dead weight but harmless). The
        # main-process copy honors the ctor flag as before.
        stateful = self._stateful or worker is not None
        state = self._resume_state if stateful else None

        count = state["records"] if state else 0
        start_epoch = state["epoch"] if state else 0
        self._epoch = start_epoch
        self._stream = None

        records = self._open_records(state, worker)

        log = None
        if self.object_id_log is not None:
            worker_suffix = f".w{worker.id}" if worker else ""
            log_path = Path(f"{self.object_id_log}.dp{self.rank}{worker_suffix}")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                log = open(log_path, "a")
            except OSError as error:
                raise RuntimeError(
                    f"cannot open object audit log {log_path}"
                ) from error

        # prev_state = stream position BEFORE the record about to be drawn;
        # batch_start = position at the first record of the current OPEN
        # micro-batch. On resume that position is the loaded state itself (the
        # MMU dataset object does not exist until the record generator first
        # runs). ADR 0014 §7b opens all micro_batch_size rows at once (a
        # record's replicas must land in different rows), so the checkpoint
        # unit moved from the partial row to the partial micro-batch —
        # equally exact, since nothing in an open batch has been yielded.
        prev_state = (
            dict(state) if state else (self._snapshot(count) if stateful else None)
        )
        batch_start = prev_state
        rows: list[list] = [[] for _ in range(self.micro_batch_size)]
        used = [0] * self.micro_batch_size

        net_retries = 0
        try:
            while True:
                error = None
                record = None
                try:
                    record = next(records)
                except StopIteration:
                    break
                except httpx.HTTPError as caught:
                    error = caught
                except OSError as caught:
                    error = caught
                except RuntimeError as caught:
                    error = caught
                if error is not None:
                    # hub's http_backoff close_session() on a ConnectError
                    # races datasets' prefetch threads sharing the global
                    # httpx client -> plain RuntimeError; a rebuild gets a
                    # fresh client from get_session(). Match the message:
                    # a blanket RuntimeError catch would mask real bugs.
                    if isinstance(
                        error, RuntimeError
                    ) and "client has been closed" not in str(error):
                        raise error
                    # transient network failure: rebuild the stream from the
                    # last per-record snapshot — exact, nothing replayed or
                    # skipped (prev_state is the position BEFORE the record
                    # that failed to draw). Only possible when stateful.
                    if prev_state is None:
                        raise error
                    net_retries += 1
                    if net_retries > _MAX_NET_RETRIES:
                        raise error
                    wait = min(5 * 2 ** (net_retries - 1), _MAX_NET_RETRY_WAIT)
                    print(
                        f"[data] {type(error).__name__}: rebuilding the stream from "
                        f"the last record snapshot in {wait}s "
                        f"(retry {net_retries}/{_MAX_NET_RETRIES})",
                        flush=True,
                    )
                    time.sleep(wait)
                    # Reclaim the abandoned stream before rebuilding, or its
                    # datasets/pyarrow prefetch buffers leak per rebuild and RSS
                    # climbs to the cgroup OOM. Drop our last live reference
                    # (self._stream — the dead generator's own frame is already
                    # cleared by the exception) and force a collection. An offline
                    # 3-arm probe over the real datasets machinery: +79 MiB / 40
                    # rebuilds abandoned vs +2 MiB with gc.collect(); an explicit
                    # records.close() adds nothing (the generator died by
                    # exception, so close() is a no-op here).
                    self._stream = None
                    self._epoch = prev_state["epoch"]
                    gc.collect()
                    records = self._open_records(prev_state, worker)
                    continue
                if record is None:
                    raise AssertionError("record source returned None")
                net_retries = 0  # a successful draw resets the blip budget
                # live epoch feeds the modality-order parity (ADR 0005
                # amendment); pure function of (object_id, epoch), so the
                # resumed stream rebuilds identical sequences
                #
                # ar_replicas > 1 re-emits the SAME downloaded record under a
                # different ADR 0008 span order. The corpus is transfer-bound
                # (~94% of wall waits for bytes), so extra factorisations of a
                # record already in memory are close to free, and every
                # conditional among its spans gets trained rather than one
                # sample of them. The span order is a pure function of
                # (object_id, epoch), so suffixing the id IS the reseed — no
                # extra RNG state to checkpoint. Replica 0 keeps the original
                # id, so object_id_log stays one unique line per emitted
                # sequence and the no-replay audit still catches accidental
                # duplication.
                objects = self._replica_objects(record)
                for obj in objects:
                    if len(obj) > self.seq_len:
                        raise ValueError(
                            f"object of length {len(obj)} exceeds seq_len {self.seq_len}"
                        )

                placement = self._place(objects, used)
                if placement is None:
                    # No assignment keeps this record's replicas in separate
                    # rows with room to spare, so the micro-batch is done.
                    # Nothing in it has been yielded, so the record we could
                    # not place simply opens the next one — exactly-once holds
                    # and the saved position is this record's.
                    batch = self.collator.collate_rows(rows)
                    expected_shape = (self.micro_batch_size, self.seq_len)
                    if batch["input_ids"].shape != expected_shape:
                        raise AssertionError(
                            f"packing mismatch: "
                            f"{batch['input_ids'].shape} != {expected_shape}"
                        )
                    # the saved position is the first UNTRAINED record: this
                    # one, which could not be placed and so opens the next
                    # micro-batch. Everything in the batch being yielded has
                    # been consumed by the packer and is about to be trained.
                    batch_start = prev_state
                    if stateful:
                        self._ckpt_state = batch_start
                    if log is not None:
                        log.writelines(f"{o.object_id}\n" for r in rows for o in r)
                        log.flush()
                    yield flatten_packed_batch(batch, self.config, self.seq_len)
                    rows = [[] for _ in range(self.micro_batch_size)]
                    used = [0] * self.micro_batch_size
                    placement = self._place(objects, used)
                    if placement is None:
                        raise AssertionError(
                            f"record {record.get('object_id')!r} cannot be placed "
                            f"into an empty micro-batch of {self.micro_batch_size} "
                            f"rows x {self.seq_len} tokens"
                        )
                for obj, row_index in zip(objects, placement):
                    rows[row_index].append(obj)
                    used[row_index] += len(obj)
                count += 1
                if stateful:
                    prev_state = self._snapshot(count)
        finally:
            if log is not None:
                log.close()


def loader_state_dict(dataloader) -> dict | None:
    """Checkpointable stream state of a :func:`build_astropt3_dataloader` loader.

    A StatefulDataLoader's state (which embeds every worker's row-start
    snapshot plus torchdata's prefetch/round-robin bookkeeping) is wrapped
    with the worker count so resume can insist on the same layout. A plain
    DataLoader defers to its dataset, which returns None when it holds no
    state — the caller skips saving in that case.
    """
    if hasattr(dataloader, "state_dict"):  # torchdata StatefulDataLoader
        return {
            "format": LOADER_STATE_FORMAT,
            "num_workers": dataloader.num_workers,
            "loader": dataloader.state_dict(),
        }
    return dataloader.dataset.state_dict()


def build_astropt3_dataloader(
    dataset_args,
    model_config,
    micro_batch_size: int,
    sequence_length: int,
    dp_rank: int,
    dp_size: int,
    num_workers: int = 0,
    seed: int = 0,
    resume_state_dir: str | Path | None = None,
    multiprocessing_context: str | None = None,
) -> Any:
    """Entry point called by the fork's ``run_train.py`` (astropt3_streaming).

    ``dataset_args`` is nanotron's ``AstroPT3StreamingDatasetsArgs`` and
    ``model_config`` its ``AstroPT3Config`` — both duck-typed so this module
    never imports nanotron. ``resume_state_dir`` points at a checkpoint's
    ``dataset_state/`` directory. Loader-format states (written via
    :func:`loader_state_dict` from a StatefulDataLoader) restore per-worker
    stream positions and require the same ``num_workers`` as the saving run;
    legacy dataset-format states require ``num_workers == 0``.
    """
    config = hf_config_from_modalities(
        model_config.modalities,
        getattr(model_config, "tokeniser", "affine"),
        # getattr with defaults so older fork configs still load
        **{
            f: getattr(model_config, f, d)
            for f, d in [
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
        data_root=dataset_args.data_root,
        match_index=getattr(dataset_args, "match_index", None),
        synthetic_image_only_fraction=getattr(
            dataset_args, "synthetic_image_only_fraction", 0.3
        ),
        synthetic_spectrum_only_fraction=getattr(
            dataset_args, "synthetic_spectrum_only_fraction", 0.0
        ),
        rank=dp_rank,
        world_size=dp_size,
        seed=seed,
        object_id_log=getattr(dataset_args, "object_id_log", None),
        stateful=num_workers == 0,
        ar_replicas=getattr(dataset_args, "ar_replicas", 1) or 1,
        replica_placement=getattr(dataset_args, "replica_placement", None)
        or "decorrelated",
    )
    try:
        from torchdata.stateful_dataloader import StatefulDataLoader as loader_cls
    except ImportError:
        if num_workers > 0:
            # never train unresumable: with workers the stream position lives
            # in the worker processes and only a StatefulDataLoader can save it
            raise ImportError(
                "num_loading_workers > 0 requires torchdata's StatefulDataLoader "
                "to checkpoint the stream position (`uv pip install torchdata`); "
                "either install it or set num_loading_workers: 0"
            )
        loader_cls = torch.utils.data.DataLoader
    # persistent_workers deliberately unset: the stream is endless, so the
    # loader is never re-iterated and the flag only adds state-restore risk
    loader = loader_cls(
        dataset,
        batch_size=None,  # items are already whole micro-batches
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        multiprocessing_context=multiprocessing_context,
    )
    if resume_state_dir is not None:
        state_file = Path(resume_state_dir) / STATE_FILE_TEMPLATE.format(rank=dp_rank)
        if state_file.exists():
            state = torch.serialization.load(state_file, weights_only=True)
            if isinstance(state, dict) and state.get("format") == LOADER_STATE_FORMAT:
                if state["num_workers"] != num_workers:
                    raise ValueError(
                        f"stream state was saved with num_loading_workers="
                        f"{state['num_workers']}, this run uses {num_workers}; "
                        "per-worker stream positions only map onto the same count"
                    )
                cast(Any, loader).load_state_dict(state["loader"])
            else:  # legacy dataset-format state (pre-StatefulDataLoader)
                if num_workers != 0:
                    raise ValueError(
                        "resuming a dataset-format stream state requires "
                        f"num_loading_workers == 0 (got {num_workers})"
                    )
                dataset.load_state_dict(state)
    # ADR 0014 §3: wraps only when $ASTROPT3_TELEMETRY_DIR is set, and
    # proxies state_dict/num_workers, so resume is unaffected either way
    return instrument(loader)

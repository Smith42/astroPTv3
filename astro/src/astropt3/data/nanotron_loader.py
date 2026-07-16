"""Adapter: astro data pipeline -> nanotron ``astropt3_streaming`` micro-batches.

Turns the record sources (:class:`~astropt3.data.mmu.MMUIterableDataset` or
the synthetic stream) into an endless stream of fixed-shape micro-batch dicts
for the nanotron fork's ``AstroPT3ForTraining``:

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
group — nanotron passes the dp process-group rank/size) and further across
DataLoader workers (HF datasets shard-splits MMU parquet; the synthetic
stream strides over record indices).

Checkpoint-resume (Phase 4): ``state_dict()`` returns the stream position at
the START of the current partial packing row — everything already drawn into
that row has not been trained on, so resume re-draws it and continues with
exactly the micro-batch sequence an uninterrupted run would have produced.
The synthetic state is a record counter (exact resume); the MMU state is the
HF-datasets stream state snapshotted at row starts, which is exact for
``shuffle_buffer_size == 0`` and skips at most the in-flight shuffle buffer
otherwise (never replays a trained record — matching HF's own
shuffle-resume semantics).

With ``num_workers == 0`` the dataset object itself carries the state. With
``num_workers > 0`` each DataLoader worker's dataset copy keeps its own
state, and :func:`build_astropt3_dataloader` returns a torchdata
``StatefulDataLoader`` whose ``state_dict()`` gathers the per-worker
snapshots consistent with the last micro-batch actually yielded to the
caller (worker-prefetched batches are accounted for by torchdata). The
trainer captures either kind of loader through :func:`loader_state_dict`.
"""

import itertools
from pathlib import Path

import torch

from ..configuration_astropt3 import AstroPT3Config
from .band_registry import _DIV_FACTOR
from .mmu import MMUIterableDataset
from .packing import ObjectSequencer, PackedCollator
from .synthetic import make_record

SYNTHETIC_ROOT = "synthetic"
STATE_FILE_TEMPLATE = "dp_{rank}.pt"
STATE_SUBDIR = "dataset_state"
LOADER_STATE_FORMAT = "stateful_dataloader"


def hf_config_from_modalities(modalities, tokeniser: str = "affine", **extra) -> AstroPT3Config:
    """Build the (tiny) HF-side config the sequencer/collator machinery wants.

    ``modalities`` may come from either implementation's config — both carry
    the same list of dicts. ``extra`` passes tokeniser-specific fields
    (e.g. the ``jetformer_*`` knobs) straight through to ``AstroPT3Config``.
    """
    return AstroPT3Config(modalities=[dict(m) for m in modalities], tokeniser=tokeniser, **extra)


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
            flat[f"{name}_values"] = torch.empty((0, mod.input_size), dtype=torch.float32)
            if mod.pos_type == "index":
                flat[f"{name}_positions"] = torch.empty((0,), dtype=torch.long)
            else:
                flat[f"{name}_positions"] = torch.empty((0, mod.pos_input_size), dtype=torch.float32)
    return flat


def regroup_micro_batch(flat: dict, names) -> dict:
    """Flat nanotron micro-batch -> HF ``AstroPT3Model`` forward kwargs."""
    return {
        "input_ids": flat["input_ids"],
        "position_ids": flat["position_ids"],
        "modality_values": {n: flat[f"{n}_values"] for n in names if flat[f"{n}_values"].shape[0]},
        "modality_masks": {n: flat[f"{n}_mask"] for n in names if flat[f"{n}_mask"].any()},
        "modality_positions": {n: flat[f"{n}_positions"] for n in names if flat[f"{n}_values"].shape[0]},
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
        shuffle_buffer_size: int = 0,
        synthetic_image_only_fraction: float = 0.3,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        object_id_log: str | Path | None = None,
        stateful: bool = True,
    ):
        super().__init__()
        self.config = config
        self.micro_batch_size = micro_batch_size
        self.seq_len = seq_len
        self.data_root = str(data_root)
        self.shuffle_buffer_size = shuffle_buffer_size
        self.synthetic_image_only_fraction = synthetic_image_only_fraction
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.object_id_log = None if object_id_log is None else str(object_id_log)
        self._stateful = stateful
        self._resume_state: dict | None = None  # applied on next __iter__
        self._ckpt_state: dict | None = None  # updated at every yield
        self._mmu_dataset: MMUIterableDataset | None = None
        self._epoch = 0

        self.sequencer = ObjectSequencer(config)
        self.collator = PackedCollator(config, seq_len=seq_len)

    # -- checkpoint state ---------------------------------------------------

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
        return {"records": 0, "epoch": 0, "hf_state": None, "data_root": self.data_root}

    def load_state_dict(self, state: dict | None) -> None:
        if state is None:  # a worker snapshotted before its first yield
            return
        if state.get("data_root") not in (None, self.data_root):
            raise ValueError(
                f"dataset state was saved for data_root={state['data_root']!r}, "
                f"this stream reads {self.data_root!r}"
            )
        self._resume_state = dict(state)

    def _snapshot(self, records: int) -> dict:
        """State AFTER ``records`` records have been consumed by the packer."""
        hf_state = None
        if self._mmu_dataset is not None:
            hf_state = self._mmu_dataset.state_dict()
        return {
            "records": records,
            "epoch": self._epoch,
            "hf_state": hf_state,
            "data_root": self.data_root,
        }

    # -- record sources -----------------------------------------------------

    def _synthetic_records(self, start_count: int, worker):
        """Endless deterministic stream; index striding keeps ranks/workers disjoint."""
        n_workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        offset = self.rank * n_workers + worker_id
        stride = self.world_size * n_workers
        for k in itertools.count(start_count):
            yield make_record(offset + k * stride, image_only_fraction=self.synthetic_image_only_fraction)

    def _mmu_records(self, start_epoch: int, hf_state):
        """Endless stream over the shards, reshuffled per epoch.

        On resume the first epoch is fast-forwarded to the saved HF stream
        state; if that state was end-of-stream the epoch yields nothing and
        the loop rolls into the next one.
        """
        dataset = MMUIterableDataset(
            self.data_root,
            rank=self.rank,
            world_size=self.world_size,
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed,
        )
        self._mmu_dataset = dataset
        for epoch in itertools.count(start_epoch):
            dataset.set_epoch(epoch)
            if epoch == start_epoch and hf_state is not None:
                dataset.load_state_dict(hf_state)
            self._epoch = epoch
            yield from dataset

    # -- iteration ------------------------------------------------------------

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        # Worker copies are always stateful: a StatefulDataLoader snapshots
        # them via state_dict() inside the worker process (under a plain
        # DataLoader the bookkeeping is dead weight but harmless). The
        # main-process copy honors the ctor flag as before.
        stateful = self._stateful or worker is not None
        state = self._resume_state if stateful else None

        count = state["records"] if state else 0
        start_epoch = state["epoch"] if state else 0
        hf_state = state["hf_state"] if state else None
        self._epoch = start_epoch
        self._mmu_dataset = None

        if self.data_root == SYNTHETIC_ROOT:
            records = self._synthetic_records(count, worker)
        else:
            records = self._mmu_records(start_epoch, hf_state)

        log = None
        if self.object_id_log is not None:
            worker_suffix = f".w{worker.id}" if worker else ""
            log_path = Path(f"{self.object_id_log}.dp{self.rank}{worker_suffix}")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = open(log_path, "a")

        # prev_state = stream position BEFORE the record about to be drawn;
        # row_start = position at the first record of the current partial row.
        # On resume that position is the loaded state itself (the MMU dataset
        # object does not exist until the record generator first runs).
        prev_state = dict(state) if state else (self._snapshot(count) if stateful else None)
        row_start = prev_state
        rows: list[list] = []
        row: list = []
        used = 0

        try:
            for record in records:
                obj = self.sequencer.build(record)
                if len(obj) > self.seq_len:
                    raise ValueError(f"object of length {len(obj)} exceeds seq_len {self.seq_len}")
                if used + len(obj) > self.seq_len:
                    rows.append(row)
                    row, used = [], 0
                    row_start = prev_state  # the new row starts at this record
                    if len(rows) == self.micro_batch_size:
                        batch = self.collator([o for r in rows for o in r])
                        assert batch["input_ids"].shape == (self.micro_batch_size, self.seq_len), (
                            f"greedy repack mismatch: {batch['input_ids'].shape} != "
                            f"({self.micro_batch_size}, {self.seq_len})"
                        )
                        if stateful:
                            self._ckpt_state = row_start
                        if log is not None:
                            log.writelines(f"{o.object_id}\n" for r in rows for o in r)
                            log.flush()
                        rows = []
                        yield flatten_packed_batch(batch, self.config, self.seq_len)
                row.append(obj)
                used += len(obj)
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
) -> torch.utils.data.DataLoader:
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
                ("image_norm_divisor", _DIV_FACTOR),
                ("spiral", False),
            ]
        },
    )
    dataset = PackedMicroBatches(
        config,
        micro_batch_size,
        sequence_length,
        data_root=dataset_args.data_root,
        shuffle_buffer_size=getattr(dataset_args, "shuffle_buffer_size", 0),
        synthetic_image_only_fraction=getattr(dataset_args, "synthetic_image_only_fraction", 0.3),
        rank=dp_rank,
        world_size=dp_size,
        seed=seed,
        object_id_log=getattr(dataset_args, "object_id_log", None),
        stateful=num_workers == 0,
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
    )
    if resume_state_dir is not None:
        state_file = Path(resume_state_dir) / STATE_FILE_TEMPLATE.format(rank=dp_rank)
        if state_file.exists():
            state = torch.load(state_file, weights_only=False)
            if isinstance(state, dict) and state.get("format") == LOADER_STATE_FORMAT:
                if state["num_workers"] != num_workers:
                    raise ValueError(
                        f"stream state was saved with num_loading_workers="
                        f"{state['num_workers']}, this run uses {num_workers}; "
                        "per-worker stream positions only map onto the same count"
                    )
                loader.load_state_dict(state["loader"])
            else:  # legacy dataset-format state (pre-StatefulDataLoader)
                if num_workers != 0:
                    raise ValueError(
                        "resuming a dataset-format stream state requires "
                        f"num_loading_workers == 0 (got {num_workers})"
                    )
                dataset.load_state_dict(state)
    return loader

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
stream strides over record indices). Checkpoint-resume of the stream is
Phase 4 (``state_dict`` plumbing exists on ``MMUIterableDataset``).
"""

import itertools
from pathlib import Path

import torch

from ..config_io import load_data_config, sequencer_kwargs_from_data_config
from ..configuration_astropt3 import AstroPT3Config
from .mmu import MMUIterableDataset
from .packing import ObjectSequencer, PackedCollator
from .synthetic import make_record

SYNTHETIC_ROOT = "synthetic"


def hf_config_from_modalities(modalities, tokeniser: str = "affine") -> AstroPT3Config:
    """Build the (tiny) HF-side config the sequencer/collator machinery wants.

    ``modalities`` may come from either implementation's config — both carry
    the same list of dicts.
    """
    return AstroPT3Config(modalities=[dict(m) for m in modalities], tokeniser=tokeniser)


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


def _synthetic_records(rank: int, stride: int, image_only_fraction: float):
    """Endless deterministic record stream, disjoint across (rank, stride)."""
    for i in itertools.count(rank, stride):
        yield make_record(i, image_only_fraction=image_only_fraction)


def _mmu_records(dataset: MMUIterableDataset):
    """Endless stream over the shards, reshuffled per epoch."""
    for epoch in itertools.count():
        dataset.set_epoch(epoch)
        yield from dataset


class PackedMicroBatches(torch.utils.data.IterableDataset):
    """Endless stream of fixed-shape nanotron micro-batches.

    Objects are packed greedily into rows of ``seq_len`` (never split), rows
    are grouped ``micro_batch_size`` at a time, and each group is collated by
    the shared :class:`PackedCollator` — the greedy repack of whole rows is
    deterministic, so the collator reproduces exactly the grouped rows.

    Use with ``DataLoader(batch_size=None)``; each item IS a micro-batch.
    """

    def __init__(
        self,
        config: AstroPT3Config,
        micro_batch_size: int,
        seq_len: int,
        *,
        data_root: str = SYNTHETIC_ROOT,
        norm_stats: str | Path | None = None,
        shuffle_buffer_size: int = 0,
        synthetic_image_only_fraction: float = 0.3,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
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

        sequencer_kwargs = {}
        if norm_stats is not None:
            sequencer_kwargs = sequencer_kwargs_from_data_config(load_data_config(norm_stats))
        self.sequencer = ObjectSequencer(config, **sequencer_kwargs)
        self.collator = PackedCollator(config, seq_len=seq_len)

    def _records(self):
        worker = torch.utils.data.get_worker_info()
        if self.data_root == SYNTHETIC_ROOT:
            # stride over record indices: disjoint across ranks and workers
            n_workers = worker.num_workers if worker else 1
            worker_id = worker.id if worker else 0
            offset = self.rank * n_workers + worker_id
            stride = self.world_size * n_workers
            return _synthetic_records(offset, stride, self.synthetic_image_only_fraction)
        # MMU parquet shards: DP split here, worker split inside HF datasets
        dataset = MMUIterableDataset(
            self.data_root,
            rank=self.rank,
            world_size=self.world_size,
            shuffle_buffer_size=self.shuffle_buffer_size,
            seed=self.seed,
        )
        return _mmu_records(dataset)

    def _rows(self):
        """Greedy whole-object packing into rows of exactly seq_len budget."""
        row, used = [], 0
        for record in self._records():
            obj = self.sequencer.build(record)
            if len(obj) > self.seq_len:
                raise ValueError(f"object of length {len(obj)} exceeds seq_len {self.seq_len}")
            if used + len(obj) > self.seq_len:
                yield row
                row, used = [], 0
            row.append(obj)
            used += len(obj)

    def __iter__(self):
        rows = self._rows()
        while True:
            group = list(itertools.islice(rows, self.micro_batch_size))
            batch = self.collator([obj for row in group for obj in row])
            assert batch["input_ids"].shape == (self.micro_batch_size, self.seq_len), (
                f"greedy repack mismatch: {batch['input_ids'].shape} != "
                f"({self.micro_batch_size}, {self.seq_len})"
            )
            yield flatten_packed_batch(batch, self.config, self.seq_len)


def build_astropt3_dataloader(
    dataset_args,
    model_config,
    micro_batch_size: int,
    sequence_length: int,
    dp_rank: int,
    dp_size: int,
    num_workers: int = 0,
    seed: int = 0,
) -> torch.utils.data.DataLoader:
    """Entry point called by the fork's ``run_train.py`` (astropt3_streaming).

    ``dataset_args`` is nanotron's ``AstroPT3StreamingDatasetsArgs`` and
    ``model_config`` its ``AstroPT3Config`` — both duck-typed so this module
    never imports nanotron.
    """
    config = hf_config_from_modalities(model_config.modalities, getattr(model_config, "tokeniser", "affine"))
    dataset = PackedMicroBatches(
        config,
        micro_batch_size,
        sequence_length,
        data_root=dataset_args.data_root,
        norm_stats=getattr(dataset_args, "norm_stats", None),
        shuffle_buffer_size=getattr(dataset_args, "shuffle_buffer_size", 0),
        synthetic_image_only_fraction=getattr(dataset_args, "synthetic_image_only_fraction", 0.3),
        rank=dp_rank,
        world_size=dp_size,
        seed=seed,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,  # items are already whole micro-batches
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

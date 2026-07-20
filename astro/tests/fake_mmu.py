"""A fake live-MMU stream: synthetic records behind the ADR 0006 Source API.

Lets the loader's MMU path (sharding, epoch rollover, resume) be tested
offline, without lsdb or the hub. Monkeypatch
``astropt3.data.streaming.open_stream`` with :func:`fake_open_stream` —
``nanotron_loader`` imports it at call time, so the patch takes effect.
"""

import pandas as pd

from astropt3.data.streaming import MMUStream, Source
from astropt3.data.synthetic import make_record

# small enough that a few micro-batches roll a source past its epoch boundary
N_PARTITIONS = 6
ROWS_PER_PARTITION = 4
# disjoint index ranges keep object_ids unique across the three sources
_SPEC = {
    "images": (0, {"image_only_fraction": 1.0}),
    "spectra": (1000, {"image_only_fraction": 0.0, "spectrum_only_fraction": 1.0}),
    "pairs": (2000, {"image_only_fraction": 0.0, "spectrum_only_fraction": 0.0}),
}


ROWS_PER_GROUP = 2  # so the fakes exercise the multi-row-group path


class _FakePartition:
    def __init__(self, name: str, index: int):
        base, kwargs = _SPEC[name]
        start = base + index * ROWS_PER_PARTITION
        self._recs = [make_record(start + j, **kwargs) for j in range(ROWS_PER_PARTITION)]

    @property
    def n_groups(self) -> int:
        return max(1, -(-len(self._recs) // ROWS_PER_GROUP))

    def group(self, index: int) -> pd.DataFrame:
        chunk = self._recs[index * ROWS_PER_GROUP : (index + 1) * ROWS_PER_GROUP]
        return pd.DataFrame({"rec": chunk})


def _source(name: str) -> Source:
    return Source(
        name,
        N_PARTITIONS,
        lambda index, name=name: _FakePartition(name, index),
        lambda row: row["rec"],
    )


def fake_open_stream(
    *, split="train", seed=0, shard=0, num_shards=1, match_index=None, only=None
) -> MMUStream:
    """``only`` restricts the corpus to one source (all weight on it), for
    testing what happens when a modality shape never arrives. ``match_index``
    is accepted and ignored: the fake always has all three sources, so the
    signature just has to match the real ``open_stream``."""
    names = ("images", "spectra", "pairs")
    weights = None
    if only is not None:
        weights = tuple(1.0 if n == only else 0.0 for n in names)
    return MMUStream(
        [_source(n) for n in names],
        seed=seed,
        shard=shard,
        num_shards=num_shards,
        split=split,
        **({"weights": weights} if weights else {}),
    )

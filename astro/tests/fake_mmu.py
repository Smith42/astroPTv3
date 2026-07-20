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


def _source(name: str) -> Source:
    base, kwargs = _SPEC[name]

    def fetch(index: int) -> pd.DataFrame:
        start = base + index * ROWS_PER_PARTITION
        return pd.DataFrame(
            {"rec": [make_record(start + j, **kwargs) for j in range(ROWS_PER_PARTITION)]}
        )

    return Source(name, N_PARTITIONS, fetch, lambda row: row["rec"])


def fake_open_stream(
    *, split="train", seed=0, shard=0, num_shards=1, client=None, only=None
) -> MMUStream:
    """``only`` restricts the corpus to one source (all weight on it), for
    testing what happens when a modality shape never arrives."""
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

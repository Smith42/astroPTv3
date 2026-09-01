"""Bounded live integration check: the real catalog through InfiniteStream.

``pytest -m 'not network'`` (the default in CI gates marked-out runs) skips
this; it needs hub access and the normal suite must pass offline.
"""

import itertools

import pytest
import torch

pytestmark = pytest.mark.network

ROWS_TO_CHECK = 5


def test_live_lsdb_stream_decodes_and_selects(tiny_config, tiny_model):
    from astropt3.data.nanotron_loader import (
        LEGACY_CATALOG,
        _catalog_columns,
        decode_legacy_row,
    )

    from lsdb.loaders.hats.read_hats import open_catalog
    from lsdb.streams.catalog_streams import InfiniteStream

    catalog = open_catalog(LEGACY_CATALOG, columns=_catalog_columns(tiny_config))
    stream = InfiniteStream(
        catalog, client=None, partitions_per_chunk=1, seed=0
    )
    frame = next(iter(stream))
    assert len(frame) >= ROWS_TO_CHECK

    sequencer_runs = 0
    for _, row in itertools.islice(frame.iterrows(), ROWS_TO_CHECK):
        record = decode_legacy_row(dict(row.items()))
        flux = torch.as_tensor(record["image"]["flux"], dtype=torch.float32)
        assert flux.shape == (3, 152, 152)
        assert torch.isfinite(flux).all()
        assert record["image"]["band"] == ["des-g", "des-r", "des-z"]
        from astropt3.data.packing import ObjectSequencer

        obj = ObjectSequencer(tiny_config).build(record)
        assert obj.input_ids[0] == 1
        assert torch.isfinite(obj.values["images"]).all()
        sequencer_runs += 1
    assert sequencer_runs == ROWS_TO_CHECK

    # a few more draws must keep yielding rows (the stream is endless)
    for _ in range(2):
        more = next(iter(stream))  # fresh draw
        assert len(more) > 0

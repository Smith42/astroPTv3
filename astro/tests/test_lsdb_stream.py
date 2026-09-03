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


def test_live_desi_crossmatch_stream_decodes_and_selects(tiny_config, tiny_model):
    """ADR 0015 spectra experiment: the real DESI-left ``desi x legacy`` join,
    extended by ``OuterKdTreeCrossmatch`` to also recover Legacy rows with no
    DESI match (image-only) from bytes the plain left-join already fetches
    and discards -- see outer_crossmatch.py."""
    from astropt3.data.nanotron_loader import (
        DESI_CATALOG,
        LEGACY_CATALOG,
        _CROSSMATCH_LEGACY_SUFFIX,
        _CROSSMATCH_NESTED,
        _CROSSMATCH_RADIUS_ARCSEC,
        _catalog_columns,
        _desi_columns,
        _map_rows_columns,
        _row_from_map_rows,
        decode_crossmatch_row,
    )
    from astropt3.data.outer_crossmatch import OuterKdTreeCrossmatch
    from astropt3.data.packing import ObjectSequencer

    from lsdb.loaders.hats.read_hats import open_catalog
    from lsdb.streams.catalog_streams import InfiniteStream

    legacy_cat = open_catalog(
        LEGACY_CATALOG, columns=_catalog_columns(tiny_config, include_position=True)
    )
    desi_cat = open_catalog(DESI_CATALOG, columns=_desi_columns(tiny_config))
    catalog = desi_cat.crossmatch(
        legacy_cat,
        algorithm=OuterKdTreeCrossmatch(radius_arcsec=_CROSSMATCH_RADIUS_ARCSEC),
        how="left",
        suffixes=("", _CROSSMATCH_LEGACY_SUFFIX),
        suffix_method="all_columns",
    )
    stream = InfiniteStream(catalog, client=None, partitions_per_chunk=1, seed=0)
    frame = next(iter(stream))
    assert len(frame) > 0

    columns = _map_rows_columns(frame, _CROSSMATCH_NESTED)

    def decode(mapped):
        return {"record": decode_crossmatch_row(_row_from_map_rows(mapped, _CROSSMATCH_NESTED))}

    decoded = frame.map_rows(decode, columns=columns, infer_nesting=False)
    records = list(itertools.islice(decoded["record"], ROWS_TO_CHECK))

    sequencer = ObjectSequencer(tiny_config)
    matched_images = 0
    image_only = 0
    for record in records:
        assert "spectrum" in record or "image" in record  # never neither
        if "spectrum" in record:
            flux = torch.as_tensor(record["spectrum"]["flux"], dtype=torch.float32)
            assert flux.shape == (7781,)
        if "image" in record:
            image_flux = torch.as_tensor(record["image"]["flux"], dtype=torch.float32)
            assert image_flux.shape == (3, 152, 152)
            if "spectrum" in record:
                matched_images += 1
            else:
                image_only += 1
        obj = sequencer.build(record)
        assert obj.input_ids[0] == 1
    # not asserting matched_images/image_only > 0: a single small partition
    # draw may legitimately contain zero of either

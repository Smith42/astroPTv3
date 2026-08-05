"""Offline checks for the shared ADR 0013 match-index builder."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_match_index.py"
SPEC = importlib.util.spec_from_file_location("build_match_index", SCRIPT)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_reciprocal_index_schema_and_matching():
    assert builder.normalize_id(b"    42", strip=True) == "42"
    assert builder.normalize_id("b'    42'", strip=True) == "42"
    assert builder.SCHEMA.names == [
        "index_schema_version",
        "anchor_source",
        "anchor_revision",
        "anchor_order",
        "anchor_pixel",
        "anchor_id",
        "partner_source",
        "partner_revision",
        "partner_order",
        "partner_pixel",
        "partner_id",
        "join_kind",
        "separation_arcsec",
        "match_radius_arcsec",
        "epoch_treatment",
        "via_source",
        "via_revision",
        "via_order",
        "via_pixel",
        "via_id",
    ]

    matches = builder.reciprocal_indices(
        anchor_ra=[0.0, 0.01],
        anchor_dec=[0.0, 0.0],
        partner_ra=[0.0001, 0.0101],
        partner_dec=[0.0, 0.0],
        central_indices=[0, 1],
        radius_arcsec=1.0,
    )
    assert [(anchor, partner) for anchor, partner, _ in matches] == [(0, 0), (1, 1)]
    assert all(abs(separation - 0.36) < 1e-3 for _, _, separation in matches)


def test_v2_spoke_directory_loads_as_one_source_graph(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    from astropt3.data.match_index import load_source_graph
    from astropt3.data.streaming import SOURCE_GRAPH_ASSEMBLY, source_assembly_for_index

    directory = tmp_path / "indexes"
    directory.mkdir()
    common = {
        "index_schema_version": 2,
        "anchor_source": "legacy_north",
        "anchor_revision": "north-sha",
        "anchor_order": 6,
        "anchor_pixel": 7,
        "anchor_id": "anchor-1",
        "partner_revision": "partner-sha",
        "partner_order": 8,
        "partner_pixel": 9,
        "separation_arcsec": 0.2,
        "match_radius_arcsec": 1.0,
        "epoch_treatment": "icrs_j2000_static",
        "join_kind": "positional",
        "via_source": None,
        "via_revision": None,
        "via_order": None,
        "via_pixel": None,
        "via_id": None,
    }
    for source, partner_id in (("desi", "spectrum-1"), ("galaxies", "galaxy-1")):
        row = {**common, "partner_source": source, "partner_id": partner_id}
        pq.write_table(
            pa.Table.from_pylist([row], schema=builder.SCHEMA),
            directory / f"{source}.parquet",
        )

    graph = load_source_graph(directory)
    assert graph.schema_version == 2
    assert graph.anchor_source == "legacy_north"
    assert graph.matches == {
        (6, 7): {"anchor-1": {"desi": "spectrum-1", "galaxies": "galaxy-1"}}
    }
    assert graph.partner_cells == {(6, 7): {"desi": {(8, 9)}, "galaxies": {(8, 9)}}}
    assert source_assembly_for_index(str(directory)) == SOURCE_GRAPH_ASSEMBLY
    with pytest.raises(ValueError, match="revision"):
        builder._via_edges(directory, "desi", "wrong-sha")
    assert builder._via_edges(directory, "desi", "partner-sha") == {
        "spectrum-1": {
            "anchor_source": "legacy_north",
            "anchor_revision": "north-sha",
            "anchor_order": 6,
            "anchor_pixel": 7,
            "anchor_id": "anchor-1",
            "via_revision": "partner-sha",
            "via_order": 8,
            "via_pixel": 9,
        }
    }

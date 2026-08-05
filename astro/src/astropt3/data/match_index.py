"""Read legacy and ADR 0013 match indexes through one small interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


Cell = tuple[int, int]


@dataclass
class MatchGraph:
    """Pointer graph grouped in anchor stream order.

    ``matches[cell][anchor_id][source]`` is the partner id and
    ``partner_cells[cell][source]`` is the set of partner partitions that must
    be fetched while scanning that anchor cell.
    """

    schema_version: int
    anchor_source: str
    anchor_revision: str
    partner_revisions: dict[str, str]
    matches: dict[Cell, dict[str, dict[str, str]]]
    partner_cells: dict[Cell, dict[str, set[Cell]]]


def _integer(value, field: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"match index has invalid {field}") from error


def _legacy(table: dict) -> MatchGraph:
    matches: dict[Cell, dict[str, dict[str, str]]] = {}
    partner_cells: dict[Cell, dict[str, set[Cell]]] = {}
    for index, anchor_id in enumerate(table["image_id"]):
        anchor_cell = (
            _integer(table["image_order"][index], "image_order"),
            _integer(table["image_pixel"][index], "image_pixel"),
        )
        partner_cell = (
            _integer(table["spectrum_order"][index], "spectrum_order"),
            _integer(table["spectrum_pixel"][index], "spectrum_pixel"),
        )
        matches.setdefault(anchor_cell, {}).setdefault(str(anchor_id), {})["desi"] = (
            str(table["spectrum_id"][index])
        )
        partner_cells.setdefault(anchor_cell, {}).setdefault("desi", set()).add(
            partner_cell
        )
    return MatchGraph(1, "legacy_north", "", {"desi": ""}, matches, partner_cells)


def _v2(table: dict) -> MatchGraph:
    versions = {
        _integer(value, "index_schema_version")
        for value in table["index_schema_version"]
    }
    if versions != {2}:
        raise ValueError(f"match index mixes schema versions {sorted(versions)}")
    anchors = set(map(str, table["anchor_source"]))
    anchor_revisions = set(map(str, table["anchor_revision"]))
    if len(anchors) != 1 or len(anchor_revisions) != 1:
        raise ValueError("match index must have one pinned anchor source/revision")

    matches: dict[Cell, dict[str, dict[str, str]]] = {}
    partner_cells: dict[Cell, dict[str, set[Cell]]] = {}
    revisions: dict[str, str] = {}
    for index, anchor_id in enumerate(table["anchor_id"]):
        join_kind = str(table["join_kind"][index])
        separation = table["separation_arcsec"][index]
        radius = table["match_radius_arcsec"][index]
        via_values = [
            table[name][index]
            for name in (
                "via_source",
                "via_revision",
                "via_order",
                "via_pixel",
                "via_id",
            )
        ]
        if join_kind == "positional":
            try:
                separation_value, radius_value = float(separation), float(radius)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "positional index rows require separation and radius"
                ) from error
            if (
                not math.isfinite(separation_value)
                or not math.isfinite(radius_value)
                or radius_value <= 0
                or not 0 <= separation_value <= radius_value
                or any(value is not None for value in via_values)
            ):
                raise ValueError("invalid positional match-index row")
        elif join_kind == "lineage":
            if (
                separation is not None
                or radius is not None
                or str(table["epoch_treatment"][index]) != "exact_source_id"
                or any(value is None for value in via_values)
            ):
                raise ValueError("invalid lineage match-index row")
        else:
            raise ValueError(f"unknown join kind {join_kind!r}")

        source = str(table["partner_source"][index])
        revision = str(table["partner_revision"][index])
        if source in revisions and revisions[source] != revision:
            raise ValueError(f"partner {source!r} mixes revisions")
        revisions[source] = revision
        anchor_cell = (
            _integer(table["anchor_order"][index], "anchor_order"),
            _integer(table["anchor_pixel"][index], "anchor_pixel"),
        )
        partner_cell = (
            _integer(table["partner_order"][index], "partner_order"),
            _integer(table["partner_pixel"][index], "partner_pixel"),
        )
        partners = matches.setdefault(anchor_cell, {}).setdefault(str(anchor_id), {})
        partner_id = str(table["partner_id"][index])
        if source in partners and partners[source] != partner_id:
            raise ValueError(f"anchor {anchor_id!r} has multiple {source!r} partners")
        partners[source] = partner_id
        partner_cells.setdefault(anchor_cell, {}).setdefault(source, set()).add(
            partner_cell
        )
    return MatchGraph(
        2,
        anchors.pop(),
        anchor_revisions.pop(),
        revisions,
        matches,
        partner_cells,
    )


def load_source_graph(path: str | Path) -> MatchGraph:
    """Load one parquet file or a directory of same-schema spoke parquets."""
    import pyarrow.parquet as pq

    table = pq.read_table(path).to_pydict()
    names = set(table)
    if {
        "image_order",
        "image_pixel",
        "image_id",
        "spectrum_order",
        "spectrum_pixel",
        "spectrum_id",
    } <= names:
        return _legacy(table)
    required = {
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
    }
    missing = required - names
    if missing:
        raise ValueError(f"match index is missing columns {sorted(missing)}")
    return _v2(table)

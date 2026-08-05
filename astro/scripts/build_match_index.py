"""Build one pointer-only ADR 0013 spoke index off the single Legacy anchor.

LSDB is confined to this offline builder, and the positional join is LSDB's
own default: ``anchor.crossmatch(spoke)`` — KdTree, one neighbour, 1 arcsec.
One anchor, one spoke per run; spokes never match each other.

The MMU catalogs are HATS collections with a 10-arcsec ``default_margin``,
which lsdb attaches on open, so those joins are exact at partition edges.
galaxies-with-hats publishes no margin and is lossy there; each build prints
which of the two it got.

Pinned North × galaxies-with-hats train example::

    uv run --extra data python scripts/build_match_index.py \
      --anchor-catalog hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north@f634744d3c44dd4fde0dee3172d4887c5e3c31c0 \
      --anchor-source legacy_north \
      --anchor-revision f634744d3c44dd4fde0dee3172d4887c5e3c31c0 \
      --partner-catalog hf://datasets/Smith42/galaxies-with-hats@c0188b776c4ce6312a805a04cbc25c891a075933/train \
      --partner-source galaxies_train \
      --partner-revision c0188b776c4ce6312a805a04cbc25c891a075933 \
      --partner-id-column dr8_id \
      --out galaxies_train.parquet

The output stores source names, revisions, both HATS cells/ids, separation,
radius, epoch treatment, and schema version. A directory of same-schema spoke
parquets is one cumulative index consumable by ``load_source_graph``.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import sys
import time
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from hats.pixel_math.healpix_shim import radec2pix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

INDEX_SCHEMA_VERSION = 2
HEALPIX_ORDER = 12
# lsdb suffixes the right side; the left keeps its own column names
PARTNER_SUFFIX = "_spoke"
DISTANCE_COLUMN = "_dist_arcsec"
SCHEMA = pa.schema(
    [
        ("index_schema_version", pa.int16()),
        ("anchor_source", pa.string()),
        ("anchor_revision", pa.string()),
        ("anchor_order", pa.int8()),
        ("anchor_pixel", pa.int64()),
        ("anchor_id", pa.string()),
        ("partner_source", pa.string()),
        ("partner_revision", pa.string()),
        ("partner_order", pa.int8()),
        ("partner_pixel", pa.int64()),
        ("partner_id", pa.string()),
        ("join_kind", pa.string()),
        ("separation_arcsec", pa.float32()),
        ("match_radius_arcsec", pa.float32()),
        ("epoch_treatment", pa.string()),
        ("via_source", pa.string()),
        ("via_revision", pa.string()),
        ("via_order", pa.int8()),
        ("via_pixel", pa.int64()),
        ("via_id", pa.string()),
    ]
)


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"expected an integer, got {value!r}") from error


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"expected a number, got {value!r}") from error


def normalize_id(value: Any, strip: bool = False) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    if strip and text.startswith("b'"):
        try:
            decoded = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            decoded = text
        if isinstance(decoded, bytes):
            text = decoded.decode("utf-8")
    return text.strip() if strip else text


def partition_cells(catalog) -> set[tuple[int, int]]:
    """Return the published ``(order, pixel)`` cells for a HATS catalog."""
    try:
        return {
            (as_int(pixel.order), as_int(pixel.pixel))
            for pixel in catalog.get_ordered_healpix_pixels()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("catalog has invalid HEALPix partitions") from error


def containing_partition(
    order: int, pixel: int, cells: set[tuple[int, int]]
) -> tuple[int, int]:
    """Find the published partition containing a finer HEALPix cell."""
    order, pixel = as_int(order), as_int(pixel)
    while order >= 0:
        if (order, pixel) in cells:
            return order, pixel
        order -= 1
        pixel >>= 2
    raise KeyError("no catalog partition contains the HEALPix cell")


def in_cell(order: int, pixel: int, cell: tuple[int, int], cells: set) -> bool:
    """Is this crossmatch partition inside the given anchor partition?"""
    try:
        return containing_partition(order, pixel, cells) == cell
    except KeyError:
        return False


def _cells_of(frame, suffix: str, published: set) -> list[tuple[int, int]]:
    """Published partition of every row, from its own coordinates."""
    pixels = cast(Any, radec2pix)(
        HEALPIX_ORDER,
        frame[f"ra{suffix}"].to_numpy(dtype="float64"),
        frame[f"dec{suffix}"].to_numpy(dtype="float64"),
    )
    return [containing_partition(HEALPIX_ORDER, as_int(p), published) for p in pixels]


def _rows(pairs, args, anchor_cells, partner_cells):
    rows = {name: [] for name in SCHEMA.names}
    if len(pairs) == 0:
        return rows
    if DISTANCE_COLUMN not in pairs:
        raise ValueError(f"crossmatch result has no {DISTANCE_COLUMN} column")
    anchors = _cells_of(pairs, "", anchor_cells)
    partners = _cells_of(pairs, PARTNER_SUFFIX, partner_cells)
    for position in range(len(pairs)):
        row = pairs.iloc[position]
        values = {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "anchor_source": args.anchor_source,
            "anchor_revision": args.anchor_revision,
            "anchor_order": anchors[position][0],
            "anchor_pixel": anchors[position][1],
            "anchor_id": normalize_id(row[args.anchor_id_column], args.anchor_strip_id),
            "partner_source": args.partner_source,
            "partner_revision": args.partner_revision,
            "partner_order": partners[position][0],
            "partner_pixel": partners[position][1],
            "partner_id": normalize_id(
                row[f"{args.partner_id_column}{PARTNER_SUFFIX}"], args.partner_strip_id
            ),
            "join_kind": "positional",
            "separation_arcsec": as_float(row[DISTANCE_COLUMN]),
            "match_radius_arcsec": args.radius_arcsec,
            "epoch_treatment": args.epoch_treatment,
            "via_source": None,
            "via_revision": None,
            "via_order": None,
            "via_pixel": None,
            "via_id": None,
        }
        for name, value in values.items():
            rows[name].append(value)
    return rows


def parse_cell(value: str) -> tuple[int, int]:
    try:
        order, pixel = value.split("/", 1)
        return as_int(order), as_int(pixel)
    except ValueError as error:
        raise argparse.ArgumentTypeError("cell must be ORDER/PIXEL") from error


def _via_edges(path: str | Path, via_source: str, via_revision: str):
    table = pq.read_table(path).to_pydict()
    edges = {}
    if "index_schema_version" in table:
        for index, partner_id in enumerate(table["partner_id"]):
            if str(table["partner_source"][index]) != via_source:
                continue
            revision = str(table["partner_revision"][index])
            if revision != via_revision:
                raise ValueError(
                    f"via index has {via_source} revision {revision}, expected {via_revision}"
                )
            edge = {
                "anchor_source": str(table["anchor_source"][index]),
                "anchor_revision": str(table["anchor_revision"][index]),
                "anchor_order": as_int(table["anchor_order"][index]),
                "anchor_pixel": as_int(table["anchor_pixel"][index]),
                "anchor_id": str(table["anchor_id"][index]),
                "via_revision": revision,
                "via_order": as_int(table["partner_order"][index]),
                "via_pixel": as_int(table["partner_pixel"][index]),
            }
            key = str(partner_id)
            # lsdb's default join is one-directional, so one spectrum can be the
            # nearest neighbour of two anchors; keep the lowest anchor id so the
            # lineage spoke stays one-to-one and the build is reproducible
            if key not in edges or edge["anchor_id"] < edges[key]["anchor_id"]:
                edges[key] = edge
        if not edges:
            raise ValueError(f"via index has no {via_source!r} edges")
        return edges
    if via_source != "desi":
        raise ValueError("legacy indexes contain only DESI edges")
    for index, partner_id in enumerate(table["spectrum_id"]):
        edges[str(partner_id)] = {
            "anchor_source": "legacy_north",
            "anchor_revision": "",
            "anchor_order": as_int(table["image_order"][index]),
            "anchor_pixel": as_int(table["image_pixel"][index]),
            "anchor_id": str(table["image_id"][index]),
            "via_revision": via_revision,
            "via_order": as_int(table["spectrum_order"][index]),
            "via_pixel": as_int(table["spectrum_pixel"][index]),
        }
    return edges


def _build_lineage(args, lsdb) -> int:
    if args.via_index is None or not args.via_source or not args.via_revision:
        raise ValueError("lineage joins require via-index/source/revision")
    edges = _via_edges(args.via_index, args.via_source, args.via_revision)
    partner = lsdb.open_catalog(
        args.partner_catalog,
        columns=[args.partner_id_column, "ra", "dec"],
    )
    cells = sorted(partition_cells(partner))
    if args.cell is not None:
        if args.cell not in cells:
            raise ValueError(f"partner catalog has no partition {args.cell}")
        cells = [args.cell]
    elif args.limit_partitions is not None:
        cells = cells[: args.limit_partitions]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, SCHEMA)
    count = 0
    try:
        for number, (order, pixel) in enumerate(cells, 1):
            frame = partner.get_partition(order, pixel).compute()
            rows = {name: [] for name in SCHEMA.names}
            for raw_partner_id in frame[args.partner_id_column]:
                partner_id = normalize_id(raw_partner_id, args.partner_strip_id)
                edge = edges.get(partner_id)
                if edge is None:
                    continue
                if edge["anchor_source"] != args.anchor_source:
                    raise ValueError("via index anchor source disagrees with arguments")
                if edge["anchor_revision"] not in {"", args.anchor_revision}:
                    raise ValueError(
                        "via index anchor revision disagrees with arguments"
                    )
                if edge["via_revision"] != args.via_revision:
                    raise ValueError(
                        "via index partner revision disagrees with arguments"
                    )
                values = {
                    "index_schema_version": INDEX_SCHEMA_VERSION,
                    "anchor_source": args.anchor_source,
                    "anchor_revision": args.anchor_revision,
                    "anchor_order": edge["anchor_order"],
                    "anchor_pixel": edge["anchor_pixel"],
                    "anchor_id": edge["anchor_id"],
                    "partner_source": args.partner_source,
                    "partner_revision": args.partner_revision,
                    "partner_order": order,
                    "partner_pixel": pixel,
                    "partner_id": partner_id,
                    "join_kind": "lineage",
                    "separation_arcsec": None,
                    "match_radius_arcsec": None,
                    "epoch_treatment": "exact_source_id",
                    "via_source": args.via_source,
                    "via_revision": args.via_revision,
                    "via_order": edge["via_order"],
                    "via_pixel": edge["via_pixel"],
                    "via_id": partner_id,
                }
                for name, value in values.items():
                    rows[name].append(value)
            table = pa.table(rows, schema=SCHEMA)
            writer.write_table(table)
            count += table.num_rows
            print(
                f"[{number}/{len(cells)}] Norder={order} Npix={pixel}: "
                f"{table.num_rows} lineage | total {count}",
                flush=True,
            )
    finally:
        writer.close()
    return count


def build(args) -> int:
    if args.radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")
    if not args.anchor_revision or not args.partner_revision:
        raise ValueError("source revisions must be pinned")

    lsdb = cast(Any, importlib.import_module("lsdb"))
    if args.join_kind == "lineage":
        return _build_lineage(args, lsdb)
    if not args.anchor_catalog:
        raise ValueError("positional joins require anchor-catalog")
    anchor = lsdb.open_catalog(
        args.anchor_catalog,
        columns=[args.anchor_id_column, "ra", "dec"],
    )
    partner = lsdb.open_catalog(
        args.partner_catalog,
        columns=[args.partner_id_column, "ra", "dec"],
    )
    anchor_cells = partition_cells(anchor)
    partner_cells = partition_cells(partner)

    # the MMU collections carry a 10-arcsec default margin that lsdb attaches
    # here; a spoke without one (galaxies-with-hats) silently loses pairs whose
    # partner sits across a partition edge, so say which one this build got
    margin = getattr(partner, "margin", None)
    print(
        f"{args.partner_source}: margin cache "
        + (
            f"{margin.hc_structure.catalog_info.margin_threshold} arcsec"
            if margin is not None
            else "MISSING — matches are lossy at partition edges"
        ),
        flush=True,
    )

    matched = anchor.crossmatch(
        partner,
        radius_arcsec=args.radius_arcsec,
        suffixes=("", PARTNER_SUFFIX),
        # pinned: lsdb's default flips to "overlapping_columns" in a future
        # release, and the suffix is what identifies the spoke side here
        suffix_method="all_columns",
    )
    pixels = matched.get_ordered_healpix_pixels()
    if args.cell is not None:
        if args.cell not in anchor_cells:
            raise ValueError(f"anchor catalog has no partition {args.cell}")
        pixels = [
            p for p in pixels if in_cell(p.order, p.pixel, args.cell, anchor_cells)
        ]
    elif args.limit_partitions is not None:
        pixels = pixels[: args.limit_partitions]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, SCHEMA)
    count = 0
    started = time.time()
    try:
        for number, pixel in enumerate(pixels, 1):
            pairs = matched.get_partition(pixel.order, pixel.pixel).compute()
            table = pa.table(
                _rows(pairs, args, anchor_cells, partner_cells), schema=SCHEMA
            )
            writer.write_table(table)
            count += table.num_rows
            elapsed = time.time() - started
            print(
                f"[{number}/{len(pixels)}] Norder={pixel.order} Npix={pixel.pixel}: "
                f"{table.num_rows} matched | total {count} ({elapsed:.0f}s)",
                flush=True,
            )
    finally:
        writer.close()
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--join-kind", choices=["positional", "lineage"], default="positional"
    )
    parser.add_argument("--anchor-catalog")
    parser.add_argument("--anchor-source", required=True)
    parser.add_argument("--anchor-revision", required=True)
    parser.add_argument("--anchor-id-column", default="object_id")
    parser.add_argument("--anchor-strip-id", action="store_true")
    parser.add_argument("--partner-catalog", required=True)
    parser.add_argument("--partner-source", required=True)
    parser.add_argument("--partner-revision", required=True)
    parser.add_argument("--partner-id-column", default="object_id")
    parser.add_argument("--partner-strip-id", action="store_true")
    parser.add_argument("--via-index")
    parser.add_argument("--via-source")
    parser.add_argument("--via-revision")
    parser.add_argument("--radius-arcsec", type=float, default=1.0)
    parser.add_argument("--epoch-treatment", default="icrs_j2000_static")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--cell", type=parse_cell)
    selection.add_argument("--limit-partitions", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = build(args)
    print(f"wrote {count} matches -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

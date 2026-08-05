"""Build one reciprocal, pointer-only ADR 0013 positional-spoke index.

LSDB is confined to this offline builder. Each anchor partition is matched with
its HEALPix neighbourhood, so catalogs without a published HATS margin (notably
galaxies-with-hats) remain exact at partition edges without a full scan or
pixel materialization.

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

import astropy.units as u
import cdshealpix
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
from hats.pixel_math.healpix_shim import radec2pix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

INDEX_SCHEMA_VERSION = 2
HEALPIX_ORDER = 12
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


def _neighbourhood(order: int, pixel: int) -> list[tuple[int, int]]:
    neighbours = cdshealpix.neighbours(np.asarray([pixel], dtype=np.uint64), order)[0]
    return [(order, as_int(value)) for value in neighbours]


def _central_indices(frame, order: int, pixel: int) -> np.ndarray:
    pixels = cast(Any, radec2pix)(
        order,
        frame["ra"].to_numpy(dtype="float64"),
        frame["dec"].to_numpy(dtype="float64"),
    )
    return np.flatnonzero(pixels == pixel)


def reciprocal_indices(
    anchor_ra,
    anchor_dec,
    partner_ra,
    partner_dec,
    central_indices,
    radius_arcsec: float,
) -> list[tuple[int, int, float]]:
    """Return mutual nearest neighbours for central anchors within radius."""
    if len(central_indices) == 0 or len(partner_ra) == 0:
        return []
    anchors = cast(
        Any,
        SkyCoord(ra=np.asarray(anchor_ra) * u.deg, dec=np.asarray(anchor_dec) * u.deg),
    )
    partners = cast(
        Any,
        SkyCoord(
            ra=np.asarray(partner_ra) * u.deg,
            dec=np.asarray(partner_dec) * u.deg,
        ),
    )
    central = anchors[central_indices]
    partner_of, separation, _ = central.match_to_catalog_sky(partners)
    anchor_of, _, _ = partners.match_to_catalog_sky(anchors)
    accepted = []
    for offset, partner_index in enumerate(partner_of):
        anchor_index = as_int(central_indices[offset])
        partner_index = as_int(partner_index)
        distance = as_float(separation[offset].arcsec)
        if anchor_of[partner_index] == anchor_index and distance <= radius_arcsec:
            accepted.append((anchor_index, partner_index, distance))
    return accepted


def _partition_frame(catalog, order: int, pixel: int):
    try:
        return catalog.pixel_search(_neighbourhood(order, pixel), fine=False).compute()
    except ValueError:
        return None


def _rows(anchor, partner, accepted, args, anchor_cell, partner_cells):
    rows = {name: [] for name in SCHEMA.names}
    if not accepted:
        return rows
    partner_pixels = cast(Any, radec2pix)(
        HEALPIX_ORDER,
        partner["ra"].to_numpy(dtype="float64"),
        partner["dec"].to_numpy(dtype="float64"),
    )
    for anchor_index, partner_index, separation in accepted:
        partner_cell = containing_partition(
            HEALPIX_ORDER, as_int(partner_pixels[partner_index]), partner_cells
        )
        values = {
            "index_schema_version": INDEX_SCHEMA_VERSION,
            "anchor_source": args.anchor_source,
            "anchor_revision": args.anchor_revision,
            "anchor_order": anchor_cell[0],
            "anchor_pixel": anchor_cell[1],
            "anchor_id": normalize_id(
                anchor.iloc[anchor_index][args.anchor_id_column], args.anchor_strip_id
            ),
            "partner_source": args.partner_source,
            "partner_revision": args.partner_revision,
            "partner_order": partner_cell[0],
            "partner_pixel": partner_cell[1],
            "partner_id": normalize_id(
                partner.iloc[partner_index][args.partner_id_column],
                args.partner_strip_id,
            ),
            "join_kind": "positional",
            "separation_arcsec": separation,
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
            if key in edges and edges[key] != edge:
                raise ValueError(f"via source id {key!r} maps to multiple anchor edges")
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
    cells = sorted(anchor_cells)
    if args.cell is not None:
        if args.cell not in anchor_cells:
            raise ValueError(f"anchor catalog has no partition {args.cell}")
        cells = [args.cell]
    elif args.limit_partitions is not None:
        cells = cells[: args.limit_partitions]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(args.out, SCHEMA)
    count = 0
    started = time.time()
    try:
        for number, (order, pixel) in enumerate(cells, 1):
            anchor_frame = _partition_frame(anchor, order, pixel)
            partner_frame = _partition_frame(partner, order, pixel)
            if anchor_frame is None or partner_frame is None:
                accepted = []
                rows = {name: [] for name in SCHEMA.names}
            else:
                central = _central_indices(anchor_frame, order, pixel)
                accepted = reciprocal_indices(
                    anchor_frame["ra"],
                    anchor_frame["dec"],
                    partner_frame["ra"],
                    partner_frame["dec"],
                    central,
                    args.radius_arcsec,
                )
                rows = _rows(
                    anchor_frame,
                    partner_frame,
                    accepted,
                    args,
                    (order, pixel),
                    partner_cells,
                )
            table = pa.table(rows, schema=SCHEMA)
            writer.write_table(table)
            count += len(accepted)
            elapsed = time.time() - started
            print(
                f"[{number}/{len(cells)}] Norder={order} Npix={pixel}: "
                f"{len(accepted)} reciprocal | total {count} ({elapsed:.0f}s)",
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

"""Shared offline HATS reciprocal image--spectrum match-index builder."""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from hats.pixel_math.healpix_shim import radec2pix

HEALPIX_ORDER = 12
SCHEMA = pa.schema(
    [
        ("image_order", pa.int8()),
        ("image_pixel", pa.int64()),
        ("image_id", pa.string()),
        ("spectrum_order", pa.int8()),
        ("spectrum_pixel", pa.int64()),
        ("spectrum_id", pa.string()),
    ]
)


def partition_cells(catalog) -> set[tuple[int, int]]:
    """Return published ``(order, pixel)`` cells for ``catalog``."""
    try:
        return {
            (int(pixel.order), int(pixel.pixel))
            for pixel in catalog.get_ordered_healpix_pixels()
        }
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("catalog has invalid HEALPix partitions") from error


def containing_partition(
    order: int, pixel: int, cells: set[tuple[int, int]]
) -> tuple[int, int]:
    """Find the published partition containing a HEALPix cell."""
    while order >= 0:
        if (order, pixel) in cells:
            return order, pixel
        order -= 1
        pixel >>= 2
    raise KeyError("no catalog partition contains the HEALPix cell")


def catalog_cell(order, pixel, cells: set[tuple[int, int]]) -> tuple[int, int]:
    """Resolve a crossmatch cell to a published catalog cell."""
    try:
        return containing_partition(int(order), int(pixel), cells)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("crossmatch references an unknown catalog cell") from error


def reciprocal_pairs(
    image_spectrum_pairs: Iterable[tuple[str, str]],
    spectrum_image_pairs: set[tuple[str, str]],
) -> set[tuple[str, str]]:
    """Keep image→spectrum pairs confirmed by the reverse nearest-neighbor pass."""
    return {
        pair
        for pair in image_spectrum_pairs
        if (pair[1], pair[0]) in spectrum_image_pairs
    }


def _crossmatch(left, right, suffix: str, radius_arcsec: float):
    return left.crossmatch(
        right,
        n_neighbors=1,
        radius_arcsec=radius_arcsec,
        suffixes=("", suffix),
        suffix_method="all_columns",
        how="inner",
    )


def build_match_index(
    *,
    image_catalog_url: str,
    spectrum_catalog_url: str,
    spectrum_suffix: str,
    radius_arcsec: float,
    out: Path,
    limit_partitions: int | None = None,
) -> int:
    """Write an AION-style reciprocal, pointer-only image--spectrum index."""
    if radius_arcsec <= 0:
        raise ValueError("radius_arcsec must be positive")

    import lsdb

    open_catalog = cast(Any, getattr(lsdb, "open_catalog"))
    images = open_catalog(image_catalog_url)
    spectra = open_catalog(spectrum_catalog_url)
    image_cells = partition_cells(images)
    spectrum_cells_available = partition_cells(spectra)

    reverse_suffix = "_image"
    reverse = _crossmatch(spectra, images, reverse_suffix, radius_arcsec)
    reverse_pixels = reverse.get_ordered_healpix_pixels()
    reverse_pairs: set[tuple[str, str]] = set()
    reverse_started = time.time()
    for n, pixel in enumerate(reverse_pixels):
        frame = reverse.get_partition(pixel.order, pixel.pixel).compute()
        reverse_pairs.update(
            (str(row["object_id"]), str(row[f"object_id{reverse_suffix}"]))
            for _, row in frame.iterrows()
        )
        if n == 0 or (n + 1) % 10 == 0 or n + 1 == len(reverse_pixels):
            elapsed = time.time() - reverse_started
            eta = elapsed / (n + 1) * (len(reverse_pixels) - n - 1)
            print(
                f"[reverse {n + 1}/{len(reverse_pixels)}] {len(reverse_pairs)} pairs "
                f"({elapsed:.0f}s elapsed, ~{eta / 60:.0f}m left)",
                flush=True,
            )
    print(f"loaded {len(reverse_pairs)} reverse nearest-neighbor pairs", flush=True)

    forward = _crossmatch(images, spectra, spectrum_suffix, radius_arcsec)
    pixels = forward.get_ordered_healpix_pixels()
    if limit_partitions is not None:
        pixels = pixels[:limit_partitions]

    rows = {name: [] for name in SCHEMA.names}
    started = time.time()
    for n, pixel in enumerate(pixels):
        frame = forward.get_partition(pixel.order, pixel.pixel).compute()
        if len(frame) == 0:
            continue
        image_order, image_pixel = catalog_cell(pixel.order, pixel.pixel, image_cells)
        spectrum_cells = cast(Any, radec2pix)(
            HEALPIX_ORDER,
            frame[f"ra{spectrum_suffix}"].to_numpy(dtype="float64"),
            frame[f"dec{spectrum_suffix}"].to_numpy(dtype="float64"),
        )
        pairs = [
            (str(row["object_id"]), str(row[f"object_id{spectrum_suffix}"]))
            for _, row in frame.iterrows()
        ]
        accepted = reciprocal_pairs(pairs, reverse_pairs)
        for i, (image_id, spectrum_id) in enumerate(pairs):
            if (image_id, spectrum_id) not in accepted:
                continue
            spectrum_order, spectrum_pixel = catalog_cell(
                HEALPIX_ORDER, spectrum_cells[i], spectrum_cells_available
            )
            rows["image_order"].append(image_order)
            rows["image_pixel"].append(image_pixel)
            rows["image_id"].append(image_id)
            rows["spectrum_order"].append(spectrum_order)
            rows["spectrum_pixel"].append(spectrum_pixel)
            rows["spectrum_id"].append(spectrum_id)
        elapsed = time.time() - started
        eta = elapsed / (n + 1) * (len(pixels) - n - 1)
        print(
            f"[{n + 1}/{len(pixels)}] Norder={pixel.order} Npix={pixel.pixel}: "
            f"{len(accepted)} reciprocal matches | total {len(rows['image_id'])} "
            f"({elapsed:.0f}s elapsed, ~{eta / 60:.0f}m left)",
            flush=True,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows, schema=SCHEMA), out)
    return len(rows["image_id"])

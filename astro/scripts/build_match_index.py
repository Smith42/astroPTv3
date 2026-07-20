"""Precompute the image x spectrum crossmatch into a match-index (ADR 0006).

Runs offline on a login node in the ``[data]`` env (lsdb + network). This is
the ONLY place lsdb is used: the train-time stream reads the index and joins
by id, so no worker ever runs a spatial join.

The artifact is pointers, not pixels — ``(image_partition, image_id,
spectrum_partition, spectrum_id, dist_arcsec)``, ~0.7M rows and tens of MB.
Joined records would be ~280 GB of imagery already hosted in the
LegacySurvey catalog, plus a second bespoke schema; this keeps ADR 0006's
"one native schema" win and stays bounded by the spectroscopic side, so an
arbitrarily larger imaging corpus never inflates it.

    uv run --extra data python scripts/build_match_index.py --out match_index.parquet
    uv run --extra data python scripts/build_match_index.py --limit-partitions 8 \\
        --out /tmp/match_index_smoke.parquet      # smoke index for tests
    uv run --extra data python scripts/build_match_index.py \\
        --out match_index.parquet --upload-to UniverseTBD/astropt3-match-index

The published artifact is consumed straight from the hub — pyarrow resolves
``hf://`` through huggingface_hub, so ``match_index:
hf://datasets/<repo>/match_index.parquet`` needs no download step.

Partitions are identified by their HEALPix ``(order, pixel)`` cell rather
than by position in a catalog listing, so the artifact stays valid if MMU
adds or drops partitions; the loader resolves cells to paths through hats
and raises if the index references a partition the catalog no longer has.
"""

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from hats.pixel_math.healpix_shim import radec2pix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astropt3.data.streaming import (  # noqa: E402
    CROSSMATCH_RADIUS_ARCSEC,
    IMAGES_CATALOG,
    SPECTRA_CATALOG,
)

# resolve spectra to their partitions at a fine order, then walk up the
# quadtree to whichever order that catalog actually partitions at
HEALPIX_ORDER = 12

# Partitions are identified by their HEALPix cell, NOT by position in the
# catalog's partition list: this artifact is published and outlives any one
# listing, and a positional index would silently shift if MMU ever adds or
# drops a partition.
SCHEMA = pa.schema(
    [
        ("image_order", pa.int8()),
        ("image_pixel", pa.int64()),
        ("image_id", pa.string()),
        ("spectrum_order", pa.int8()),
        ("spectrum_pixel", pa.int64()),
        ("spectrum_id", pa.string()),
        ("dist_arcsec", pa.float32()),
    ]
)


def partition_cells(catalog) -> set:
    """The (order, pixel) cells this catalog actually partitions at."""
    return {(int(p.order), int(p.pixel)) for p in catalog.get_ordered_healpix_pixels()}


def containing_partition(order: int, pixel: int, cells: set) -> tuple:
    """The (order, pixel) of the catalog partition covering this HEALPix cell.

    A crossmatch partition is refined to the FINER of the two sides, so its
    pixel is often a sub-pixel of the image partition that holds the row
    (and an order-29 `_healpix_29` is always below any partition). Walking up
    the quadtree — each coarser order drops two bits — finds the covering
    partition in at most 30 steps.
    """
    order, pixel = int(order), int(pixel)
    while order >= 0:
        if (order, pixel) in cells:
            return (order, pixel)
        order -= 1
        pixel >>= 2
    raise KeyError(f"no partition covers HEALPix cell (order={order}, pixel={pixel})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--images", default=IMAGES_CATALOG)
    parser.add_argument("--spectra", default=SPECTRA_CATALOG)
    parser.add_argument("--radius-arcsec", type=float, default=CROSSMATCH_RADIUS_ARCSEC)
    parser.add_argument(
        "--upload-to",
        default=None,
        metavar="REPO_ID",
        help="also upload to this HF dataset repo (needs `hf auth login` or $HF_TOKEN)",
    )
    parser.add_argument(
        "--upload-path",
        default="match_index.parquet",
        help="path within the HF repo (default: match_index.parquet)",
    )
    parser.add_argument(
        "--limit-partitions",
        type=int,
        default=None,
        help="smoke mode: only the first N crossmatch partitions",
    )
    args = parser.parse_args()

    import lsdb

    images = lsdb.open_catalog(args.images)
    spectra = lsdb.open_catalog(args.spectra)
    image_cells = partition_cells(images)
    spectrum_cells_available = partition_cells(spectra)

    pairs = images.crossmatch(
        spectra,
        n_neighbors=1,
        radius_arcsec=args.radius_arcsec,
        suffixes=("", "_desi"),
        # pinned: lsdb's default flips to "overlapping_columns" in a future
        # release, and the _desi suffix is what identifies the right side
        suffix_method="all_columns",
        how="inner",
    )
    pixels = pairs.get_ordered_healpix_pixels()
    if args.limit_partitions is not None:
        pixels = pixels[: args.limit_partitions]

    rows = {name: [] for name in SCHEMA.names}
    started = time.time()
    for n, pixel in enumerate(pixels):
        frame = pairs.get_partition(pixel.order, pixel.pixel).compute()
        if len(frame) == 0:
            continue
        img_order, img_pixel = containing_partition(pixel.order, pixel.pixel, image_cells)
        # only the LEFT catalog's _healpix_29 survives the join (as the frame
        # index); the right side contributes ra_desi/dec_desi, so the matched
        # spectrum's cell is recomputed from its coordinates
        spectrum_cells = radec2pix(
            HEALPIX_ORDER,
            frame["ra_desi"].to_numpy(dtype="float64"),
            frame["dec_desi"].to_numpy(dtype="float64"),
        )
        for i, (_, row) in enumerate(frame.iterrows()):
            spec_order, spec_pixel = containing_partition(
                HEALPIX_ORDER, spectrum_cells[i], spectrum_cells_available
            )
            rows["image_order"].append(img_order)
            rows["image_pixel"].append(img_pixel)
            rows["image_id"].append(str(row["object_id"]))
            rows["spectrum_order"].append(spec_order)
            rows["spectrum_pixel"].append(spec_pixel)
            rows["spectrum_id"].append(str(row["object_id_desi"]))
            rows["dist_arcsec"].append(float(row.get("_dist_arcsec", 0.0)))
        print(
            f"[{n + 1}/{len(pixels)}] Norder={pixel.order} Npix={pixel.pixel}: "
            f"{len(frame)} matches | total {len(rows['image_id'])} "
            f"({time.time() - started:.0f}s)",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(rows, schema=SCHEMA), args.out)
    print(f"wrote {len(rows['image_id'])} matches -> {args.out}")

    if args.upload_to:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.upload_to, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(args.out),
            path_in_repo=args.upload_path,
            repo_id=args.upload_to,
            repo_type="dataset",
            commit_message=(
                f"match index: {len(rows['image_id'])} pairs, "
                f"{args.radius_arcsec}\" radius"
            ),
        )
        url = f"hf://datasets/{args.upload_to}/{args.upload_path}"
        print(f"uploaded -> {url}")
        print(f"use it with: match_index: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Prepare the pilot corpus: LEFT-crossmatch MMU images with DESI spectra.

Runs on the login node in the ``[data]`` env (needs network; lsdb pulls the
HATS partitions from the HF hub):

    uv run --extra data python scripts/prepare_pilot_data.py

Both sources are HATS collections with 10-arcsec margin caches (verified
2026-07-07), so ``lsdb.open_catalog`` works directly and the crossmatch is
correct across partition borders. ``how="left"`` keeps every image row and
attaches the nearest spectrum within the match radius where one exists
(~0.5-1M matched, ~13M image-only expected).

The result keeps the left catalog's HEALPix partitioning; partitions are
computed one at a time (bounded memory), each row is normalized to
``PILOT_FEATURES``, split train/val by coarse HEALPix tile, and written as
per-partition parquet shards of ``--shard-size`` rows. A partition is
journalled in ``progress.jsonl`` only after all its shards are renamed into
place, and stale files of unjournalled partitions are deleted before redo,
so rerunning after an interruption is exactly correct (no loss, no
duplicates); matched/unmatched counts are logged throughout and written to
``provenance.json``.

Smoke run (a few partitions only):

    uv run --extra data python scripts/prepare_pilot_data.py \\
        --out /tmp/pilot_smoke --limit-partitions 2
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astropt3.config_io import resolve_data_root  # noqa: E402
from astropt3.data.mmu import assign_split, write_shard  # noqa: E402

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "data" / "pilot_images_spectra.yaml"
)

# Image-catalog columns copied into the shards; the right (DESI) side
# contributes spectrum, Z, ZERR, ZWARN and the match distance.
_LEFT_SCALARS = ("ebv", "flux_g", "flux_r", "flux_z", "z_spec")


def _is_null(cell) -> bool:
    if cell is None or cell is pd.NA:
        return True
    return isinstance(cell, (float, np.floating)) and math.isnan(cell)


def _struct(cell) -> dict:
    """Coerce a struct-column cell (dict or nested-pandas frame) to a dict."""
    if isinstance(cell, dict):
        return cell
    if hasattr(cell, "columns"):  # nested-pandas: a per-row sub-DataFrame
        return {c: cell[c].to_numpy() for c in cell.columns}
    raise TypeError(f"cannot interpret struct cell of type {type(cell)}")


class _Row:
    """Suffix-tolerant column access on a crossmatch result row."""

    def __init__(self, row, suffixes=("", "_desi")):
        self._row = row
        self._suffixes = suffixes

    def get(self, name, side=0):
        for key in (name + self._suffixes[side], name):
            if key in self._row.index:
                return self._row[key]
        return None


def row_to_record(row, suffixes=("", "_desi")) -> dict:
    """One crossmatch-result row -> raw record for ``normalize_record``."""
    r = _Row(row, suffixes)
    healpix = r.get("_healpix_29")
    if healpix is None:
        # lsdb frames carry _healpix_29 as the spatial index, not a column;
        # under iterrows() it surfaces as the row name
        healpix = row.name
    record = {
        "object_id": r.get("object_id"),
        "ra": r.get("ra"),
        "dec": r.get("dec"),
        "_healpix_29": healpix,
        "image": _struct(r.get("image")),
    }
    for key in _LEFT_SCALARS:
        value = r.get(key)
        if not _is_null(value):
            record[key] = float(value)

    spectrum = r.get("spectrum", side=1)
    if not _is_null(spectrum):
        record["spectrum"] = _struct(spectrum)
        for key in ("Z", "ZERR", "ZWARN"):
            value = r.get(key, side=1)
            if not _is_null(value):
                record[key] = bool(value) if key == "ZWARN" else float(value)
        dist = r.get("_dist_arcsec", side=1)
        if not _is_null(dist):
            record["match_dist_arcsec"] = float(dist)
    return record


def _load_progress(path: Path) -> dict:
    done = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                done[(entry["order"], entry["pixel"])] = entry
    return done


def _partition_glob(order: int, pixel: int) -> str:
    return f"part-{order:02d}-{pixel:07d}-*"


def clean_partition(out_dir: Path, order: int, pixel: int) -> None:
    """Delete stale shards of a partition that was never journalled as done."""
    for split in ("train", "val"):
        for stale in (out_dir / split).glob(_partition_glob(order, pixel)):
            stale.unlink()


def write_partition(
    records_by_split: dict, out_dir: Path, order: int, pixel: int, shard_size: int
) -> None:
    """Write one partition's records as per-split shards of <= shard_size rows."""
    for split, records in records_by_split.items():
        for k, start in enumerate(range(0, len(records), shard_size)):
            write_shard(
                records[start : start + shard_size],
                out_dir / split / f"part-{order:02d}-{pixel:07d}-{k:03d}.parquet",
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser.add_argument("--out", type=Path, default=resolve_data_root(config))
    parser.add_argument("--images", default=config["sources"]["images"])
    parser.add_argument("--spectra", default=config["sources"]["spectra"])
    parser.add_argument(
        "--radius-arcsec",
        type=float,
        default=config["sources"]["crossmatch_radius_arcsec"],
    )
    parser.add_argument(
        "--val-fraction", type=float, default=config["split"]["val_fraction"]
    )
    parser.add_argument(
        "--split-order", type=int, default=config["split"]["healpix_order"]
    )
    parser.add_argument("--split-salt", type=int, default=config["split"]["salt"])
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument(
        "--limit-partitions",
        type=int,
        default=None,
        help="smoke mode: process only the first N result partitions",
    )
    parser.add_argument(
        "--cone",
        type=float,
        nargs=3,
        metavar=("RA", "DEC", "RADIUS_ARCSEC"),
        default=None,
        help="smoke mode: cone-filter the image catalog before crossmatching",
    )
    args = parser.parse_args()

    import lsdb  # [data] extra; deliberately absent from the CPU test env

    print(f"opening {args.images}")
    images = lsdb.open_catalog(args.images)
    print(f"opening {args.spectra}")
    spectra = lsdb.open_catalog(args.spectra)
    if args.cone is not None:
        images = images.cone_search(*args.cone)

    xmatched = images.crossmatch(
        spectra,
        radius_arcsec=args.radius_arcsec,
        n_neighbors=1,
        how="left",
        suffixes=("", "_desi"),
        suffix_method="all_columns",
    )
    pixels = xmatched.get_ordered_healpix_pixels()
    if args.limit_partitions is not None:
        pixels = pixels[: args.limit_partitions]
    print(f"{len(pixels)} partitions to process")

    args.out.mkdir(parents=True, exist_ok=True)
    progress_path = args.out / "progress.jsonl"
    done = _load_progress(progress_path)
    counts = {"n_objects": 0, "n_matched": 0, "n_val": 0}
    for entry in done.values():
        counts["n_objects"] += entry["n_rows"]
        counts["n_matched"] += entry["n_matched"]
        counts["n_val"] += entry["n_val"]
    if done:
        print(f"resuming: {len(done)} partitions already done ({counts})")

    started = time.time()
    for i, pixel in enumerate(pixels):
        key = (pixel.order, pixel.pixel)
        if key in done:
            continue
        clean_partition(args.out, pixel.order, pixel.pixel)
        df = xmatched.partitions[i].compute()
        records_by_split = {"train": [], "val": []}
        n_matched = 0
        for _, row in df.iterrows():
            record = row_to_record(row)
            split = assign_split(
                record["_healpix_29"],
                val_fraction=args.val_fraction,
                order=args.split_order,
                salt=args.split_salt,
            )
            records_by_split[split].append(record)
            n_matched += "spectrum" in record
        write_partition(
            records_by_split, args.out, pixel.order, pixel.pixel, args.shard_size
        )
        n_val = len(records_by_split["val"])
        counts["n_objects"] += len(df)
        counts["n_matched"] += n_matched
        counts["n_val"] += n_val
        with progress_path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "order": int(pixel.order),
                        "pixel": int(pixel.pixel),
                        "n_rows": len(df),
                        "n_matched": int(n_matched),
                        "n_val": n_val,
                    }
                )
                + "\n"
            )
        rate = counts["n_objects"] / max(time.time() - started, 1e-9)
        print(
            f"[{i + 1}/{len(pixels)}] Norder={pixel.order} Npix={pixel.pixel}: "
            f"{len(df)} objects ({n_matched} matched) | totals: "
            f"{counts['n_objects']} objects, {counts['n_matched']} matched "
            f"({100 * counts['n_matched'] / max(counts['n_objects'], 1):.2f}%), "
            f"{counts['n_val']} val | {rate:.0f} obj/s",
            flush=True,
        )

    provenance = {
        "sources": {
            "images": args.images,
            "spectra": args.spectra,
            "crossmatch_radius_arcsec": args.radius_arcsec,
            "crossmatch_how": "left",
            "n_neighbors": 1,
        },
        "split": {
            "healpix_order": args.split_order,
            "val_fraction": args.val_fraction,
            "salt": args.split_salt,
        },
        "counts": {
            **counts,
            "n_image_only": counts["n_objects"] - counts["n_matched"],
            "n_train": counts["n_objects"] - counts["n_val"],
        },
        "versions": {
            "lsdb": lsdb.__version__,
            "numpy": np.__version__,
        },
        "created": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance["counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Diagnose the crossmatch-only loader's memory: rebuild leak vs per-cell spike.

The 20k-step `astropt3-70m-jetformer-crossmatch-only` run was OOM-killed at
step 13,354 (68.7 GiB cgroup, 8 workers x 2 ranks) after 52 stream rebuilds,
with step time flat from start to finish. Two candidate causes, one arm each:

  --rebuilds N   the rebuild path leaks (offline, fake_mmu fixtures). Reports
                 RSS growth per rebuild with and without the gc.collect() that
                 `nanotron_loader` does. Flat => not the rebuild path.
  --cells N      `_crossmatch_examples` holds a whole cell's filtered spectra
                 as one Arrow table for the length of the image scan. Reads
                 parquet FOOTERS only (no row data) off the real catalogs and
                 reports the uncompressed bytes each cell pins per worker.

ponytail: a diagnostic, not a gate — nothing imports it and no test runs it.

    uv run python scripts/probe_stream_rss.py --rebuilds 40
    uv run python scripts/probe_stream_rss.py --cells 12
"""

import argparse
import gc
import importlib
import itertools
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

MIB = 1024 * 1024


def rss_mib() -> float:
    """Peak-free current RSS (statm is cheap and exact enough here)."""
    try:
        with open("/proc/self/statm") as handle:
            pages = int(handle.read().split()[1])
    except (OSError, ValueError, IndexError) as error:
        raise RuntimeError("cannot read process RSS from /proc") from error
    return pages * os.sysconf("SC_PAGE_SIZE") / MIB


def probe_rebuilds(rebuilds: int, draws: int) -> None:
    """Open/abandon the real datasets machinery over local parquet, N times."""
    fake_mmu = importlib.import_module("fake_mmu")
    fixtures = fake_mmu._fixtures
    fake_open_stream = fake_mmu.fake_open_stream

    fixtures()  # build once, outside the measured loop
    for collect in (False, True):
        stream = None
        gc.collect()
        base = rss_mib()
        for _ in range(rebuilds):
            stream = fake_open_stream()
            for _ in itertools.islice(iter(stream), draws):
                pass
            if collect:
                stream = None
                gc.collect()
        del stream
        gc.collect()
        grown = rss_mib() - base
        arm = "gc.collect()" if collect else "abandon"
        print(
            f"{arm:>13}: {grown:+7.1f} MiB over {rebuilds} rebuilds "
            f"({grown / rebuilds:+.2f} MiB/rebuild)"
        )


def probe_cells(cells: int) -> None:
    """Uncompressed bytes each image cell pins, from parquet footers only."""
    import fsspec
    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from astropt3.data.streaming import (
        IMAGES_CATALOG,
        MATCH_INDEX_ENV,
        SPECTRA_CATALOG,
        _spectrum_owners,
        catalog_files,
        load_match_index,
        resolve_match_index,
    )

    index = resolve_match_index()
    if index is None:
        raise SystemExit(f"set ${MATCH_INDEX_ENV} to the published match index")

    _, image_by_cell = catalog_files(IMAGES_CATALOG)
    _, spectrum_by_cell = catalog_files(SPECTRA_CATALOG)
    matches, spectra_of = load_match_index(index)

    paths_by_cell = {
        cell: [spectrum_by_cell[s] for s in sorted(spectra_of[cell])]
        for cell in sorted(matches)
    }
    owners = _spectrum_owners(paths_by_cell)

    def footer_mib(path: str) -> tuple[float, int, float]:
        with fsspec.open(path, "rb") as handle:
            meta = pq.ParquetFile(handle).metadata
        groups = [meta.row_group(i) for i in range(meta.num_row_groups)]
        total = sum(g.total_byte_size for g in groups) / MIB
        largest = max(g.total_byte_size for g in groups) / MIB
        return total, meta.num_rows, largest

    header = f"{'cell':>16} {'owned':>5} {'pinned MiB':>11} {'spec rows':>10}"
    print(f"{header} {'img rows':>9} {'pairs':>6} {'img rowgrp':>11}")
    worst = 0.0
    for cell in sorted(matches)[:cells]:
        owned = [p for p in paths_by_cell[cell] if owners[p] == cell]
        footers = [footer_mib(p) for p in owned]
        pinned = sum(f[0] for f in footers)
        spec_rows = sum(f[1] for f in footers)
        _, image_rows, image_group = footer_mib(image_by_cell[cell])
        worst = max(worst, pinned)
        print(
            f"{str(cell):>16} {len(owned):>5} {pinned:>11.1f} {spec_rows:>10} "
            f"{image_rows:>9} {len(matches[cell]):>6} {image_group:>11.1f}"
        )
    print(
        f"\nworst cell pins {worst:.0f} MiB of spectra per worker; "
        f"8 workers x 2 ranks = {worst * 16 / 1024:.1f} GiB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuilds", type=int, default=0)
    parser.add_argument("--draws", type=int, default=8, help="records per rebuild")
    parser.add_argument("--cells", type=int, default=0)
    args = parser.parse_args()
    if not args.rebuilds and not args.cells:
        parser.error("pass --rebuilds N and/or --cells N")
    if args.rebuilds:
        probe_rebuilds(args.rebuilds, args.draws)
    if args.cells:
        probe_cells(args.cells)


if __name__ == "__main__":
    main()

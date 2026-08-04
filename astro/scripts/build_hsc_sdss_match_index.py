"""Build a 1-arcsec HSC PDR3 Deep/UltraDeep × SDSS match index.

Run offline on a login node:

    uv run --extra data python scripts/build_hsc_sdss_match_index.py \
        --out hsc_sdss_match_index.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _match_index import build_match_index

HSC_CATALOG = "hf://datasets/UniverseTBD/mmu_hsc_pdr3_dud_22.5"
SDSS_CATALOG = "hf://datasets/LSDB/mmu_sdss_sdss"
RADIUS_ARCSEC = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-partitions", type=int, default=None)
    args = parser.parse_args()
    count = build_match_index(
        image_catalog_url=HSC_CATALOG,
        spectrum_catalog_url=SDSS_CATALOG,
        spectrum_suffix="_sdss",
        radius_arcsec=RADIUS_ARCSEC,
        out=args.out,
        limit_partitions=args.limit_partitions,
    )
    print(f"wrote {count} HSC × SDSS matches -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

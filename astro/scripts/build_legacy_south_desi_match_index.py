"""Build a 1-arcsec Legacy Survey DR10 South × DESI EDR/SV3 match index.

Run offline on a login node:

    uv run --extra data python scripts/build_legacy_south_desi_match_index.py \
        --out legacy_south_desi_match_index.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _match_index import build_match_index

LEGACY_SOUTH_CATALOG = "hf://datasets/hugging-science/mmu_legacysurvey_dr10_south_21"
DESI_CATALOG = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"
RADIUS_ARCSEC = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-partitions", type=int, default=None)
    args = parser.parse_args()
    count = build_match_index(
        image_catalog_url=LEGACY_SOUTH_CATALOG,
        spectrum_catalog_url=DESI_CATALOG,
        spectrum_suffix="_desi",
        radius_arcsec=RADIUS_ARCSEC,
        out=args.out,
        limit_partitions=args.limit_partitions,
    )
    print(f"wrote {count} Legacy South × DESI matches -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

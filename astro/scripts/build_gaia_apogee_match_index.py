"""Build an AION-style reciprocal Gaia DR3 × APOGEE DR17 pilot index.

This uses reciprocal 1-arcsec nearest neighbors. It is a density diagnostic,
not a publishable stellar match index: production needs epoch propagation.

    uv run --extra data python scripts/build_gaia_apogee_match_index.py \
        --limit-partitions 6 --out gaia_apogee_match_index_scout.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _match_index import build_match_index

GAIA_CATALOG = "hf://datasets/UniverseTBD/mmu_gaia_gaia"
APOGEE_CATALOG = "hf://datasets/hugging-science/mmu_apogee_dr17"
RADIUS_ARCSEC = 1.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-partitions", type=int, default=None)
    args = parser.parse_args()
    count = build_match_index(
        image_catalog_url=GAIA_CATALOG,
        spectrum_catalog_url=APOGEE_CATALOG,
        spectrum_suffix="_apogee",
        radius_arcsec=RADIUS_ARCSEC,
        out=args.out,
        limit_partitions=args.limit_partitions,
    )
    print(f"wrote {count} reciprocal Gaia × APOGEE matches -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

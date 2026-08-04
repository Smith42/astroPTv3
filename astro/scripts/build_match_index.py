"""Build the reciprocal LegacySurvey North × DESI match index (ADR 0006).

Runs offline on a login node in the ``[data]`` environment. The result has
only source IDs and HATS HEALPix cells—no copied pixels or spectra.

    uv run --extra data python scripts/build_match_index.py --out match_index.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _match_index import build_match_index
from astropt3.data.streaming import (  # noqa: E402
    CROSSMATCH_RADIUS_ARCSEC,
    IMAGES_CATALOG,
    SPECTRA_CATALOG,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit-partitions", type=int, default=None)
    args = parser.parse_args()
    count = build_match_index(
        image_catalog_url=IMAGES_CATALOG,
        spectrum_catalog_url=SPECTRA_CATALOG,
        spectrum_suffix="_desi",
        radius_arcsec=CROSSMATCH_RADIUS_ARCSEC,
        out=args.out,
        limit_partitions=args.limit_partitions,
    )
    print(f"wrote {count} reciprocal matches -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

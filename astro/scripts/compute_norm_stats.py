"""Calibrate the Platonic Universe asinh stretch from prepared pilot shards.

Streams images from the train split (offline; no network), samples pixels per
band, and writes per-band [g, r, z] 1st/99th flux percentiles into the
``normalization`` block of ``configs/data/pilot_images_spectra.yaml`` — the
values ``ObjectSequencer(image_p1=..., image_p99=...)`` consumes. Prints
before/after-stretch flux histograms per band and the observed spectrum
wavelength range so the calibration can be eyeballed:

    uv run python scripts/compute_norm_stats.py \\
        [--data-dir {root}/train] [--n-images 10000] [--pixels-per-image 8192]
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astropt3.config_io import resolve_data_root  # noqa: E402
from astropt3.data.mmu import N_BANDS, MMUIterableDataset  # noqa: E402
from astropt3.data.transforms import (  # noqa: E402
    ASINH_ALPHA,
    asinh_params_from_percentiles,
    asinh_stretch,
)

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "data" / "pilot_images_spectra.yaml"
)

_SUMMARY_PERCENTILES = (0.1, 1, 50, 99, 99.9)


def _text_histogram(values: np.ndarray, bins: int = 20, width: int = 50) -> str:
    counts, edges = np.histogram(values, bins=bins)
    peak = counts.max()
    lines = []
    for count, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * round(width * count / peak) if peak else ""
        lines.append(f"  [{lo:>10.3f}, {hi:>10.3f}) {bar} {count}")
    return "\n".join(lines)


def _report_band(name: str, raw: np.ndarray, stretched: np.ndarray) -> None:
    raw_p = np.percentile(raw, _SUMMARY_PERCENTILES)
    str_p = np.percentile(stretched, _SUMMARY_PERCENTILES)
    header = " ".join(f"p{p:<5g}" for p in _SUMMARY_PERCENTILES)
    print(f"\nband {name}:")
    print(f"  percentiles      {header}")
    print("  raw flux    " + " ".join(f"{v:9.3f}" for v in raw_p))
    print("  stretched   " + " ".join(f"{v:9.3f}" for v in str_p))
    print(" raw flux histogram (clipped to p0.1-p99.9):")
    print(_text_histogram(np.clip(raw, raw_p[0], raw_p[-1])))
    print(" stretched histogram:")
    print(_text_histogram(stretched))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    config = yaml.safe_load(CONFIG_PATH.read_text())
    parser.add_argument(
        "--data-dir", type=Path, default=resolve_data_root(config) / "train"
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--n-images", type=int, default=10_000)
    parser.add_argument("--pixels-per-image", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset = MMUIterableDataset(args.data_dir, shuffle_buffer_size=256, seed=args.seed)

    samples: list[np.ndarray] = []  # each (3, pixels_per_image)
    lambda_min, lambda_max = np.inf, -np.inf
    n_images = n_spectra = 0
    for record in dataset:
        flux = record["image"]["flux"].reshape(N_BANDS, -1)
        cols = rng.choice(flux.shape[1], size=args.pixels_per_image, replace=False)
        samples.append(flux[:, cols])
        n_images += 1
        if "spectrum" in record:
            lam = record["spectrum"]["lambda"]
            lambda_min = min(lambda_min, float(lam.min()))
            lambda_max = max(lambda_max, float(lam.max()))
            n_spectra += 1
        if n_images >= args.n_images:
            break
    if n_images < args.n_images:
        print(f"warning: only {n_images} images available (asked for {args.n_images})")

    pixels = np.concatenate(samples, axis=1)  # (3, n_images * pixels_per_image)
    p1 = np.percentile(pixels, 1, axis=1).astype(np.float32)
    p99 = np.percentile(pixels, 99, axis=1).astype(np.float32)

    offset, scale = asinh_params_from_percentiles(p1, p99, ASINH_ALPHA)
    stretched = asinh_stretch(
        torch.from_numpy(pixels).unsqueeze(-1), scale, offset
    ).squeeze(-1).numpy()
    for b, band in enumerate(("g", "r", "z")):
        _report_band(band, pixels[b], stretched[b])

    print(f"\n{n_images} images sampled ({n_spectra} with spectra)")
    print(f"per-band p1  = {p1.tolist()}")
    print(f"per-band p99 = {p99.tolist()}")
    if n_spectra:
        print(f"spectrum lambda range: [{lambda_min:.1f}, {lambda_max:.1f}] A")
        if lambda_min < 3000.0 or lambda_max > 10_000.0:
            print("warning: lambda range outside the expected DESI 3600-9824 A")

    updated = yaml.safe_load(args.config.read_text())
    updated["normalization"] = {
        "asinh_alpha": float(ASINH_ALPHA),
        "image_p1": [float(v) for v in p1],
        "image_p99": [float(v) for v in p99],
        "stats_provenance": {
            "data_dir": str(args.data_dir),
            "n_images": n_images,
            "pixels_per_image": args.pixels_per_image,
            "seed": args.seed,
            "created": datetime.now(timezone.utc).isoformat(),
        },
    }
    args.config.write_text(yaml.safe_dump(updated, sort_keys=False))
    print(f"wrote normalization block to {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

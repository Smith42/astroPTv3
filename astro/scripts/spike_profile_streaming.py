"""ADR 0006 profiling spike: is naive pyarrow+hats streaming fast enough?

The ADR's Performance section requires >=2x training-consumption per rank and
says nothing is built until this runs. It answers four questions:

1. Does `hats` alone (no lsdb) enumerate partitions and resolve `hf://` paths?
2. Where does the time go — partition read vs row decode vs tokenize?
3. Is ~256 concurrent workers RAM-feasible?
4. Is the image flux column fixed-shape (the precondition for a columnar
   decode)?

Throwaway. Needs network:

    uv run python scripts/spike_profile_streaming.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from astropt3.config_io import load_model_config  # noqa: E402
from astropt3.data.packing import ObjectSequencer  # noqa: E402
from astropt3.data.streaming import row_to_record, spectra_row_to_record  # noqa: E402

IMAGES = "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north"
SPECTRA = "hf://datasets/UniverseTBD/mmu_desi_edr_sv3"
N_PARTITIONS = 5
N_OBJECTS = 150  # decode/tokenize this many rows per partition (read is whole)


def hf_filesystem():
    from huggingface_hub import HfFileSystem

    return HfFileSystem()


def partition_paths(url: str, n: int):
    """`hats` alone: partition enumeration + HEALPix order + hf:// resolution."""
    import hats
    from hats.io import paths

    collection = hats.read_hats(url)
    catalog = getattr(collection, "main_catalog", collection)
    pixels = catalog.get_healpix_pixels()
    return [paths.pixel_catalog_file(catalog.catalog_base_dir, p) for p in pixels[:n]], len(pixels)


def strip(uri: str) -> str:
    return str(uri).replace("hf://datasets/", "datasets/")


def profile(name: str, url: str, decode, sequencer):
    files, n_total = partition_paths(url, N_PARTITIONS)
    fs = hf_filesystem()
    print(f"\n=== {name}: {n_total} partitions, profiling {len(files)} ===")

    read_s = decode_s = tok_s = 0.0
    rows_read = objs = 0
    peak_bytes = 0
    flux_shapes = set()

    for uri in files:
        t0 = time.perf_counter()
        table = pq.read_table(strip(uri), filesystem=fs)
        read_s += time.perf_counter() - t0
        rows_read += table.num_rows
        peak_bytes = max(peak_bytes, table.nbytes)

        t0 = time.perf_counter()
        df = table.to_pandas()
        frame_s = time.perf_counter() - t0

        take = min(N_OBJECTS, len(df))
        t0 = time.perf_counter()
        records = [decode(df.iloc[i]) for i in range(take)]
        decode_s += (time.perf_counter() - t0) + frame_s * take / max(len(df), 1)

        for rec in records:
            image = rec.get("image")
            if image is not None:
                flux_shapes.add(np.asarray(image["flux"]).shape)

        t0 = time.perf_counter()
        for rec in records:
            sequencer.build(rec)
        tok_s += time.perf_counter() - t0
        objs += take

    read_per_obj = read_s / max(rows_read, 1)
    decode_per_obj = decode_s / max(objs, 1)
    tok_per_obj = tok_s / max(objs, 1)
    total = read_per_obj + decode_per_obj + tok_per_obj

    print(f"  rows read      {rows_read} in {read_s:.1f}s")
    print(f"  partition RAM  {peak_bytes / 1e6:.0f} MB (largest of {len(files)})")
    print(f"  read           {read_per_obj * 1e3:7.2f} ms/obj  ({read_per_obj / total:5.1%})")
    print(f"  decode         {decode_per_obj * 1e3:7.2f} ms/obj  ({decode_per_obj / total:5.1%})")
    print(f"  tokenize       {tok_per_obj * 1e3:7.2f} ms/obj  ({tok_per_obj / total:5.1%})")
    print(f"  => {1 / total:.1f} obj/s per worker")
    if flux_shapes:
        print(f"  image flux shapes: {flux_shapes}  "
              f"({'fixed — columnar decode viable' if len(flux_shapes) == 1 else 'RAGGED'})")
    return 1 / total, peak_bytes


def main() -> int:
    config, _ = load_model_config(
        Path(__file__).resolve().parents[1] / "configs" / "model" / "test-tiny.yaml"
    )
    sequencer = ObjectSequencer(config)

    spec_rate, spec_ram = profile("spectra", SPECTRA, spectra_row_to_record, sequencer)
    img_rate, img_ram = profile("images", IMAGES, row_to_record, sequencer)

    print("\n=== verdict ===")
    # the corpus is 60% images / 15% spectra / 25% pairs; pairs read both sides,
    # so per-object cost sits between the two measured rates
    print(f"images {img_rate:.1f} obj/s/worker, spectra {spec_rate:.1f} obj/s/worker")
    for workers in (8, 32, 256):
        print(f"  {workers:3d} workers: ~{img_rate * workers:8.0f} img obj/s, "
              f"RAM ~{img_ram * workers / 1e9:6.1f} GB resident partitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

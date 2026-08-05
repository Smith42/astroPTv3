#!/usr/bin/env python
"""Deterministic ADR 0013 Legacy North/South overlap scout.

The first gate is intentionally narrower than the final corpus audit: it measures
reciprocal positional-spoke matches per physical catalog byte so one anchor can
be prototyped before the new source transforms exist. The evidence marks the
full target-token/byte gate as pending rather than inventing token counts.

Prepare the frozen stratified cell order, then run the live adaptive scout::

    uv run python scripts/scout_legacy_anchor.py plan --out ../tmp/adr0013-scout-plan.json
    uv run --extra data python scripts/scout_legacy_anchor.py run \
        --plan ../tmp/adr0013-scout-plan.json \
        --out ../tmp/adr0013-scout-evidence.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import re
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import cdshealpix
import httpx
import numpy as np
from astropy.io import fits
from hats.pixel_math.healpix_shim import radec2pix
from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CONFIG = ROOT / "configs/data/adr0013_anchor_scout.json"
PARTITION_RE = re.compile(r"/dataset/Norder=(\d+)/Dir=\d+/Npix=(\d+)\.parquet$")
HEALPIX_ORDER = 12


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"expected an integer, got {value!r}") from error


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"expected a number, got {value!r}") from error


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read scout JSON {path}") from error


def retry(call, label: str, attempts: int = 5):
    for attempt in range(attempts):
        try:
            return call()
        except (httpx.HTTPError, OSError, TimeoutError):
            if attempt + 1 == attempts:
                raise
            wait = 2**attempt
            print(f"{label} failed; retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise AssertionError("unreachable")


def pinned_url(spec: dict[str, Any]) -> str:
    return f"hf://datasets/{spec['repo']}@{spec['revision']}"


def stable_key(seed: int, *parts: object) -> str:
    value = ":".join(map(str, (seed, *parts)))
    return hashlib.sha256(value.encode()).hexdigest()


def partition_sizes(spec: dict[str, Any]) -> dict[tuple[int, int], int]:
    info = HfApi().dataset_info(
        spec["repo"], revision=spec["revision"], files_metadata=True
    )
    if info.sha != spec["revision"]:
        raise RuntimeError(f"revision mismatch for {spec['repo']}: {info.sha}")
    roots = spec["roots"] if "roots" in spec else [spec["root"]]
    sizes: dict[tuple[int, int], int] = {}
    for sibling in info.siblings or []:
        if not any(sibling.rfilename.startswith(root + "/") for root in roots):
            continue
        match = PARTITION_RE.search(sibling.rfilename)
        if match and sibling.size is not None:
            cell = as_int(match.group(1)), as_int(match.group(2))
            sizes[cell] = sizes.get(cell, 0) + as_int(sibling.size)
    if not sizes:
        raise RuntimeError(f"no parquet partitions found for {spec['repo']}")
    return sizes


def load_skymap(spec: dict[str, Any]) -> np.ndarray:
    total: np.ndarray | None = None
    for root in spec["roots"] if "roots" in spec else [spec["root"]]:
        path = hf_hub_download(
            spec["repo"],
            f"{root}/skymap.fits",
            repo_type="dataset",
            revision=spec["revision"],
        )
        with fits.open(path, memmap=True) as hdus:
            table = cast(Any, hdus[1])
            data = table.data
            if "T" in table.columns.names:
                values = np.asarray(data["T"], dtype=np.int64).copy()
            else:
                values = np.zeros(
                    12 * as_int(table.header["NSIDE"]) ** 2, dtype=np.int64
                )
                values[np.asarray(data["PIXEL"], dtype=np.int64)] = data["VALUE"]
        total = values if total is None else total + values
    if total is None:
        raise RuntimeError(f"no skymaps found for {spec['repo']}")
    return total


def rows_in_cell(skymap: np.ndarray, order: int, pixel: int) -> int:
    shift = 2 * (10 - order)
    if shift < 0:
        return as_int(skymap[pixel >> -shift])
    return as_int(skymap[pixel << shift : (pixel + 1) << shift].sum())


def galactic_latitudes(cells: list[tuple[int, int]]) -> np.ndarray:
    pixels = np.asarray([pixel for _, pixel in cells], dtype=np.uint64)
    orders = np.asarray([order for order, _ in cells], dtype=np.uint8)
    lon, lat = cdshealpix.healpix_to_lonlat(pixels, orders)
    ra = np.asarray(lon.to_value("rad"))
    dec = np.asarray(lat.to_value("rad"))
    pole_ra, pole_dec = np.deg2rad([192.85948, 27.12825])
    sin_b = np.sin(dec) * np.sin(pole_dec) + np.cos(dec) * np.cos(pole_dec) * np.cos(
        ra - pole_ra
    )
    return np.abs(np.rad2deg(np.arcsin(np.clip(sin_b, -1, 1))))


def deterministic_order(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["stratum"]].append(row)
    for stratum, values in groups.items():
        values.sort(
            key=lambda row: (
                stable_key(seed, stratum, row.get("footprint_bin", 0)),
                stable_key(seed, stratum, row["order"], row["pixel"]),
            )
        )
        footprints: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            footprints[row.get("footprint_bin", 0)].append(row)
        groups[stratum] = [
            row
            for offset in range(max(map(len, footprints.values())))
            for footprint in sorted(footprints)
            if offset < len(footprints[footprint])
            for row in [footprints[footprint][offset]]
        ]
    ordered = []
    for offset in range(max(map(len, groups.values()))):
        for stratum in sorted(groups):
            if offset < len(groups[stratum]):
                ordered.append(groups[stratum][offset])
    return ordered


def bootstrap_ratio(
    rows: list[dict[str, Any]],
    seed: int,
    replicates: int,
    population: dict[str, int] | None = None,
) -> dict[str, float]:
    matches = np.asarray([row["matches"] for row in rows], dtype=np.float64)
    byte_counts = np.asarray([row["physical_bytes"] for row in rows], dtype=np.float64)
    groups = {
        stratum: np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if row.get("stratum", "all") == stratum
            ]
        )
        for stratum in {row.get("stratum", "all") for row in rows}
    }
    population = population or {
        stratum: len(indices) for stratum, indices in groups.items()
    }
    weights = np.asarray(
        [
            population[row.get("stratum", "all")]
            / len(groups[row.get("stratum", "all")])
            for row in rows
        ]
    )
    point = as_float(np.dot(matches, weights) / np.dot(byte_counts, weights))
    rng = np.random.default_rng(seed)
    sample_matches = np.zeros(replicates)
    sample_bytes = np.zeros(replicates)
    for stratum, indices in groups.items():
        draws = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        weight = population[stratum] / len(indices)
        sample_matches += matches[draws].sum(axis=1) * weight
        sample_bytes += byte_counts[draws].sum(axis=1) * weight
    samples = sample_matches / sample_bytes
    low, high = np.quantile(samples, [0.025, 0.975])
    half_width = as_float((high - low) / 2)
    return {
        "matches_per_byte": point,
        "ci95_low": as_float(low),
        "ci95_high": as_float(high),
        "relative_half_width": half_width / point if point else math.inf,
    }


def source_hash() -> str:
    paths = [
        ROOT / "src/astropt3/modalities.py",
        ROOT / "src/astropt3/tokenization.py",
        ROOT / "src/astropt3/data/band_registry.py",
        ROOT / "src/astropt3/data/scalar_registry.py",
        ROOT / "src/astropt3/data/spectral.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_plan(config: dict[str, Any]) -> dict[str, Any]:
    anchors = config["anchors"]
    spokes = config["spokes"]
    anchor_sizes = {name: partition_sizes(spec) for name, spec in anchors.items()}
    source_sizes = {
        name: partition_sizes(spec)
        for name, spec in spokes.items()
        if spec["kind"] == "positional"
    }
    anchor_maps = {name: load_skymap(spec) for name, spec in anchors.items()}
    anchor_masks = {name: values > 0 for name, values in anchor_maps.items()}

    source_maps: dict[str, np.ndarray] = {}
    footprint_summary: dict[str, Any] = {}
    for source, spec in spokes.items():
        values = load_skymap(spec)
        source_maps[source] = values
        north = anchor_masks["north"]
        south = anchor_masks["south"]
        footprint_summary[source] = {
            "catalog_rows": as_int(values.sum()),
            "north_only_rows": as_int(values[north & ~south].sum()),
            "south_only_rows": as_int(values[south & ~north].sum()),
            "both_rows": as_int(values[north & south].sum()),
            "outside_rows": as_int(values[~north & ~south].sum()),
        }

    planned: dict[str, Any] = {}
    for anchor, sizes in anchor_sizes.items():
        cells = list(sizes)
        source_counts = {
            source: np.asarray(
                [rows_in_cell(values, *cell) for cell in cells], dtype=np.int64
            )
            for source, values in source_maps.items()
        }
        positional_sources = [
            source for source, spec in spokes.items() if spec["kind"] == "positional"
        ]
        totals = np.sum(
            [source_counts[source] for source in positional_sources], axis=0
        )
        keep = np.flatnonzero(totals > 0)
        kept_cells = [cells[index] for index in keep]
        bytes_ = np.asarray([sizes[cell] for cell in kept_cells], dtype=np.int64)
        latitude = galactic_latitudes(kept_cells)
        area_deg2 = np.asarray(
            [41252.96124941927 / (12 * 4**order) for order, _ in kept_cells]
        )
        density = totals[keep] / area_deg2
        breadth = np.sum(
            [source_counts[source][keep] > 0 for source in positional_sources], axis=0
        )
        cuts = {
            "attachment_density_rows_per_deg2": as_float(np.median(density)),
            "absolute_galactic_latitude_deg": as_float(np.median(latitude)),
            "partition_bytes": as_int(np.median(bytes_)),
        }
        rows = []
        for offset, (order, pixel) in enumerate(kept_cells):
            counts = {
                source: as_int(source_counts[source][keep[offset]])
                for source in source_counts
            }
            bits = (
                as_int(density[offset] > cuts["attachment_density_rows_per_deg2"]),
                as_int(latitude[offset] > cuts["absolute_galactic_latitude_deg"]),
                as_int(bytes_[offset] > cuts["partition_bytes"]),
            )
            rows.append(
                {
                    "order": order,
                    "pixel": pixel,
                    "partition_bytes": as_int(bytes_[offset]),
                    "anchor_rows": rows_in_cell(anchor_maps[anchor], order, pixel),
                    "source_footprint_rows": counts,
                    "attachment_density_rows_per_deg2": as_float(density[offset]),
                    "absolute_galactic_latitude_deg": as_float(latitude[offset]),
                    "footprint_bin": as_int(breadth[offset] > 1),
                    "stratum": "d{}-b{}-z{}".format(*bits),
                }
            )
        order = deterministic_order(rows, config["sample_seed"])
        limit = config["max_cells"]
        planned[anchor] = {
            "population_cells": len(rows),
            "stratum_population": dict(
                sorted(
                    (stratum, sum(row["stratum"] == stratum for row in rows))
                    for stratum in {row["stratum"] for row in rows}
                )
            ),
            "quantile_cuts": cuts,
            "cell_order": order[:limit],
        }

    return {
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "registry_transform_hash": source_hash(),
        "metric_scope": (
            "prototype reciprocal positional-spoke matches per physical catalog byte; "
            "final non-padding target-token/actual-byte gate remains pending source adapters"
        ),
        "footprint_summary": footprint_summary,
        "anchors": planned,
        "positional_partition_bytes": {
            source: {f"{order}/{pixel}": size for (order, pixel), size in sizes.items()}
            for source, sizes in source_sizes.items()
        },
    }


def reciprocal_cell_matches(
    anchor_catalog,
    spoke_catalog,
    source: str,
    order: int,
    pixel: int,
    radius_arcsec: float,
) -> tuple[list[tuple[str, str, float, float]], int]:
    try:
        left = anchor_catalog.pixel_search((order, pixel), fine=True)
        right = spoke_catalog.pixel_search((order, pixel), fine=False)
    except ValueError:
        return [], 0
    kwargs = {
        "n_neighbors": 1,
        "radius_arcsec": radius_arcsec,
        "suffix_method": "all_columns",
        "how": "inner",
        "require_right_margin": True,
    }

    def compute():
        forward = left.crossmatch(
            right, suffixes=("", f"_{source}"), **kwargs
        ).compute()
        reverse = right.crossmatch(left, suffixes=("", "_anchor"), **kwargs).compute()
        return forward, reverse

    try:
        forward, reverse = retry(compute, f"{source} {order}/{pixel} crossmatch")
    except RuntimeError as error:
        if str(error) == "Catalogs do not overlap":
            return [], 0
        raise
    forward_rows = {
        (str(row.object_id), str(getattr(row, f"object_id_{source}"))): row
        for row in forward.itertuples()
    }
    reverse_pairs = {
        (str(row.object_id_anchor), str(row.object_id)) for row in reverse.itertuples()
    }
    matches = []
    for pair in sorted(forward_rows.keys() & reverse_pairs):
        row = forward_rows[pair]
        matches.append(
            (
                pair[0],
                pair[1],
                as_float(getattr(row, f"ra_{source}")),
                as_float(getattr(row, f"dec_{source}")),
            )
        )
    return matches, len(forward_rows) - len(matches)


def containing_partition(order: int, pixel: int, cells: set[tuple[int, int]]):
    while order >= 0:
        if (order, pixel) in cells:
            return order, pixel
        order -= 1
        pixel >>= 2
    raise KeyError("no catalog partition contains the HEALPix cell")


def run_anchor(
    name: str,
    plan: dict[str, Any],
    config: dict[str, Any],
    max_cells: int | None,
) -> dict[str, Any]:
    lsdb = cast(Any, importlib.import_module("lsdb"))
    open_catalog = lsdb.open_catalog

    anchor_spec = config["anchors"][name]
    columns = ["object_id", "ra", "dec"]
    anchor_catalog = retry(
        lambda: open_catalog(pinned_url(anchor_spec), columns=columns),
        f"open {name}",
    )
    positional = {
        source: spec
        for source, spec in config["spokes"].items()
        if spec["kind"] == "positional"
    }
    catalogs = {
        source: retry(
            lambda spec=spec: open_catalog(pinned_url(spec), columns=columns),
            f"open {source}",
        )
        for source, spec in positional.items()
    }
    partition_bytes = {
        source: {
            (as_int(cell.split("/")[0]), as_int(cell.split("/")[1])): size
            for cell, size in plan["positional_partition_bytes"][source].items()
        }
        for source in positional
    }
    cap = min(max_cells or config["max_cells"], config["max_cells"])
    rows = []
    status = "inconclusive"
    interval = None
    for cell in plan["anchors"][name]["cell_order"][:cap]:
        source_metrics = {}
        physical_bytes = cell["partition_bytes"]
        total_matches = 0
        for source in positional:
            matches, one_way = reciprocal_cell_matches(
                anchor_catalog,
                catalogs[source],
                source,
                cell["order"],
                cell["pixel"],
                config["crossmatch_radius_arcsec"],
            )
            cells = set(partition_bytes[source])
            touched = set()
            if matches:
                pixels = cast(Any, radec2pix)(
                    HEALPIX_ORDER,
                    np.asarray([match[2] for match in matches]),
                    np.asarray([match[3] for match in matches]),
                )
                touched = {
                    containing_partition(HEALPIX_ORDER, as_int(pixel), cells)
                    for pixel in pixels
                }
            source_bytes = sum(partition_bytes[source][cell] for cell in touched)
            physical_bytes += source_bytes
            total_matches += len(matches)
            source_metrics[source] = {
                "reciprocal_matches": len(matches),
                "one_way_rejected": one_way,
                "partner_partitions": [list(cell) for cell in sorted(touched)],
                "partner_partition_bytes": source_bytes,
                "matched_id_hash": hashlib.sha256(
                    "\n".join(match[1] for match in matches).encode()
                ).hexdigest(),
            }
        rows.append(
            {
                **cell,
                "matches": total_matches,
                "physical_bytes": physical_bytes,
                "spokes": source_metrics,
                "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / 1024**2,
            }
        )
        if len(rows) < config["min_cells"] or len(rows) % config["batch_size"]:
            continue
        interval = bootstrap_ratio(
            rows,
            config["bootstrap_seed"],
            config["bootstrap_replicates"],
            plan["anchors"][name]["stratum_population"],
        )
        print(
            f"{name}: {len(rows)} cells, {interval['matches_per_byte'] * 2**30:.2f} "
            f"matches/GiB, relative half-width={interval['relative_half_width']:.3f}",
            flush=True,
        )
        if interval["relative_half_width"] <= config["relative_half_width"]:
            status = "converged"
            break
    if interval is None or len(rows) % config["batch_size"]:
        interval = bootstrap_ratio(
            rows,
            config["bootstrap_seed"],
            config["bootstrap_replicates"],
            plan["anchors"][name]["stratum_population"],
        )
    if len(rows) == config["max_cells"] and status != "converged":
        status = "inconclusive"
    return {
        "status": status,
        "cells_sampled": len(rows),
        "interval": interval,
        "cells": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--out", type=Path, required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--plan", type=Path, required=True)
    run_parser.add_argument("--out", type=Path, required=True)
    run_parser.add_argument(
        "--anchor", choices=["north", "south", "both"], default="both"
    )
    run_parser.add_argument("--max-cells", type=int)
    args = parser.parse_args()

    config = load_json(args.config)
    if args.command == "plan":
        output = build_plan(config)
    else:
        plan = load_json(args.plan)
        if plan["config"] != config:
            raise RuntimeError("plan config differs from the requested scout config")
        names = config["anchors"] if args.anchor == "both" else [args.anchor]
        plan_sha = hashlib.sha256(args.plan.read_bytes()).hexdigest()
        if args.out.exists():
            output = load_json(args.out)
            if output.get("plan_sha256") != plan_sha:
                raise RuntimeError("existing evidence belongs to a different plan")
        else:
            output = {
                "schema_version": config["schema_version"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": plan_sha,
                "metric_scope": plan["metric_scope"],
                "lineage_gate": config["lineage_gate"],
                "anchors": {},
            }
        for name in names:
            if name in output["anchors"]:
                print(f"keeping completed {name} evidence", flush=True)
                continue
            output["anchors"][name] = run_anchor(name, plan, config, args.max_cells)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

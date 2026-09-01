"""Sample from a converted AstroPT3 checkpoint and render the results.

Template records come from one live draw of the LSDB LegacySurvey North
catalog (``hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north``, ADR 0015)
— image-only/uncrossmatched, so ``image-to-spectra``/``spectra-to-images``
modes (jetformer-only, and needing a spectrum in the record) have nothing to
condition on until a crossmatched source returns. ``reconstruct`` works for
any checkpoint; ``unconditional`` works for jetformer checkpoints.
``astropt3.eval.samples.sample_checkpoint`` (model-side only, ADR 0015 §6)
does the sampling/rendering; this script only supplies live records and
optional wandb logging.

Usage:
    uv run python scripts/generate.py --checkpoint <hf_dir> \
        [--mode reconstruct] [--n 4] [--rows 0,1,2] [--stream-seed 0] \
        [--seed 0] [--out generated] [--wandb] [--wandb-run-id <id>]

Outputs land in ``--out`` as PNGs (a grid for images, flux-vs-wavelength for
spectra). ``--wandb`` logs the same figures to the astropt3 wandb project as
a fresh generation run; ``--wandb-run-id <id>`` appends them to an existing
run instead (e.g. the training run that produced the checkpoint).
"""

import argparse
import importlib
from pathlib import Path
from typing import Any, cast


def _draw_live_records(config, rows: list[int], stream_seed: int) -> list[dict]:
    """Decode ``rows`` positions out of one LSDB InfiniteStream chunk."""
    from lsdb.loaders.hats.read_hats import open_catalog
    from lsdb.streams.catalog_streams import InfiniteStream

    from astropt3.data.nanotron_loader import (
        LEGACY_CATALOG,
        _catalog_columns,
        decode_legacy_row,
    )

    catalog = open_catalog(LEGACY_CATALOG, columns=_catalog_columns(config))
    stream = InfiniteStream(catalog, client=None, partitions_per_chunk=1, seed=stream_seed)
    frame = next(iter(stream))
    if max(rows) >= len(frame):
        raise ValueError(f"drew {len(frame)} rows, but --rows asked for index {max(rows)}")
    return [decode_legacy_row(dict(frame.iloc[i].items())) for i in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="HF checkpoint dir")
    parser.add_argument(
        "--mode",
        choices=["unconditional", "image-to-spectra", "spectra-to-images", "reconstruct"],
        default=None,
        help="default: every mode the checkpoint's tokeniser supports",
    )
    parser.add_argument("--n", type=int, default=4, help="samples to draw per mode")
    parser.add_argument("--temperature", type=float, default=1.0, help="scales GMM sigma")
    parser.add_argument("--seed", type=int, default=0, help="sampling RNG seed")
    parser.add_argument(
        "--stream-seed", type=int, default=0, help="LSDB InfiniteStream draw seed"
    )
    parser.add_argument(
        "--rows",
        default="0",
        help="comma-separated row positions in the drawn chunk to render",
    )
    parser.add_argument("--out", default="generated", help="output directory")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="log the figures to wandb (project astropt3, job_type generation)",
    )
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="append to an existing wandb run (e.g. the training run) instead of a new one",
    )
    args = parser.parse_args()

    import astropt3  # noqa: F401  -- registers the Auto classes
    from transformers import AutoConfig

    from astropt3.eval.samples import sample_checkpoint

    try:
        rows = [int(i) for i in args.rows.split(",")]
    except ValueError:
        parser.error("--rows must be a comma-separated list of integers")

    config = AutoConfig.from_pretrained(args.checkpoint)
    records = _draw_live_records(config, rows, args.stream_seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = sample_checkpoint(
        args.checkpoint,
        records,
        modes=[args.mode] if args.mode else None,
        n=args.n,
        temperature=args.temperature,
        seed=args.seed,
        out_dir=out_dir,
        device=args.device,
    )

    wandb_run = None
    if args.wandb:
        wandb_module = cast(Any, importlib.import_module("wandb"))
        wandb_run = wandb_module.init(
            project="astropt3",
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            name=None if args.wandb_run_id else "generate",
            job_type="generation",
            config={k: v for k, v in vars(args).items() if k not in ("wandb", "wandb_run_id")},
        )
        wandb_run.log({f"generation/{key}": wandb_module.Image(png) for key, png in pngs.items()})
        wandb_run.finish()

    for key, png in pngs.items():
        print(f"wrote {key}: {png}")


if __name__ == "__main__":
    main()

"""Sample from a (jetformer) AstroPT3 checkpoint and render the results.

Modes:
- ``unconditional``:    <|bos|> -> full image span -> spectra span (if the
                        template object has one), all sampled.
- ``image-to-spectra``: teacher-force the template record's image tokens and
                        sample only the spectra span.
- ``reconstruct``:      one-step teacher-forced predictions for every span
                        (works for affine checkpoints too).

The template record fixes the token skeleton and the positions (image patch
indices, spectra wavelengths). ``--data-root synthetic`` (default) uses the
deterministic synthetic records; point it at a prepared val shard dir to use
a real record.

Usage:
    uv run python scripts/generate.py --checkpoint <hf_dir> \
        --mode image-to-spectra --n 4 --temperature 0.9 \
        [--data-root <val_dir>|synthetic] [--record-index 0] [--out outdir]

Outputs land in ``--out`` as ``.npy`` (raw sampled values, data space) plus
PNGs: a grid for images, flux-vs-wavelength for spectra. Non-unconditional
modes lead with a ground-truth panel/trace from the template record. Image
values get the physical inverse normalization (band-registry keyed by the
template record's bands; no calibration file needed) — exact for jetformer
checkpoints (whose sequencer skips per-patch standardization precisely so
the token map inverts back to flux); for affine checkpoints it is
qualitative only, since standardization discards each patch's mean/std.
``--wandb`` logs the figures to the astropt3
wandb project as a fresh generation run; ``--wandb-run-id <id>`` appends
them to an existing run instead (e.g. the training run). Pass several
comma-separated ``--record-index`` values to collect a batch of
reconstructions into one run.

The sampling/rendering implementation lives in ``astropt3.eval.samples``,
shared with the per-checkpoint sweep (``run_probe_sweep.py`` — ADR 0003).
"""

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="HF checkpoint dir")
    parser.add_argument(
        "--mode",
        choices=["unconditional", "image-to-spectra", "reconstruct"],
        default="unconditional",
    )
    parser.add_argument("--n", type=int, default=4, help="samples to draw")
    parser.add_argument("--temperature", type=float, default=1.0, help="scales GMM sigma")
    parser.add_argument("--argmax", action="store_true", help="mixture-mean point sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default="synthetic", help="val shard dir or 'synthetic'")
    parser.add_argument(
        "--record-index",
        default="0",
        help="template record index; comma-separated list logs several into one run",
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
    from transformers import AutoModel

    from astropt3.data.packing import ObjectSequencer
    from astropt3.eval.samples import (
        load_template_record,
        render_sampled_tokens,
        sample_template,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.checkpoint).to(device).eval()

    # sampling modes want the full skeleton (image + spectra spans)
    need_spectrum = args.mode != "reconstruct"
    record_indices = [int(i) for i in str(args.record_index).split(",")]
    sequencer = ObjectSequencer(model.config)

    # one wandb run for the whole invocation: a fresh generation run by
    # default, or the run named by --wandb-run-id (e.g. the training run)
    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project="astropt3",
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
            name=None if args.wandb_run_id else f"generate-{args.mode}",
            job_type="generation",
            config={k: v for k, v in vars(args).items() if k not in ("wandb", "wandb_run_id")},
        )

    # ground truth for teacher-forced/conditioned spans: unconditional samples
    # have none; reconstruct compares both spans; image-to-spectra compares
    # the generated spectra against the record's real spectrum
    show_truth = args.mode != "unconditional"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for record_index in record_indices:
        record = load_template_record(args.data_root, record_index, need_spectrum)
        template = sequencer.build(record)
        print(f"template object {template.object_id!r}: spans {sorted(template.masks)}")

        generator = None
        if args.mode != "reconstruct":
            generator = torch.Generator(device=device).manual_seed(args.seed)
        sampled = sample_template(
            model,
            template,
            args.mode,
            n=args.n,
            temperature=args.temperature,
            argmax=args.argmax,
            generator=generator,
        )

        # the object id keeps figures from different records distinct on disk
        # and as wandb media keys, so a multi-record batch shares one run
        tag = f"{args.mode}_{template.object_id}_seed{args.seed}"
        for name, tokens in sampled.items():
            np.save(out_dir / f"{name}_{tag}.npy", tokens.cpu().float().numpy())
        pngs = render_sampled_tokens(
            model, record, template, sampled, out_dir=out_dir, tag=tag, show_truth=show_truth
        )
        for name, png in pngs.items():
            if wandb_run is not None:
                wandb_run.log({f"generation/{name}_{tag}": wandb.Image(str(png))})
            print(f"wrote {name}: {tuple(sampled[name].shape)} -> {out_dir}/{name}_{tag}.{{npy,png}}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

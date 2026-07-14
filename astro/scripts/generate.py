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
        [--data-root <val_dir>|synthetic] [--record-index 0] \
        [--norm-stats configs/data/pilot_images_spectra.yaml] [--out outdir]

Outputs land in ``--out`` as ``.npy`` (raw sampled values, data space) plus
PNGs: a grid for images, flux-vs-wavelength for spectra. Non-unconditional
modes lead with a ground-truth panel/trace from the template record. With
``--norm-stats`` image values additionally get the inverse asinh stretch —
exact for jetformer checkpoints (whose sequencer skips per-patch
standardization precisely so the token map inverts back to flux); for
affine checkpoints it is qualitative only, since standardization discards
each patch's mean/std. ``--wandb`` logs the figures to the astropt3
wandb project as a fresh generation run; ``--wandb-run-id <id>`` appends
them to an existing run instead (e.g. the training run). Pass several
comma-separated ``--record-index`` values to collect a batch of
reconstructions into one run.
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import torch


def load_template_record(data_root: str, record_index: int, need_spectrum: bool) -> dict:
    if data_root == "synthetic":
        from astropt3.data.synthetic import make_record

        record = make_record(record_index, image_only_fraction=0.0 if need_spectrum else 0.3)
        return record
    from astropt3.data.mmu import MMUIterableDataset

    dataset = MMUIterableDataset(data_root, rank=0, world_size=1, shuffle_buffer_size=0)
    wanted = (
        r for r in dataset if not need_spectrum or r.get("spectrum") is not None
    )
    record = next(itertools.islice(wanted, record_index, None), None)
    if record is None:
        raise ValueError(f"fewer than {record_index + 1} usable records in {data_root}")
    return record


def save_image_png(values: np.ndarray, path: Path, title: str, truth: np.ndarray | None = None):
    """[n, C, H, W] -> one PNG grid (per-image normalized RGB).

    With ``truth`` [C, H, W] the ground-truth panel leads the grid.
    """
    import matplotlib.pyplot as plt

    panels = ([("truth", truth)] if truth is not None else []) + [
        (f"sample {i}", img) for i, img in enumerate(values)
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2), squeeze=False)
    for ax, (label, img) in zip(axes[0], panels):
        rgb = np.transpose(img, (1, 2, 0))
        lo, hi = np.percentile(rgb, [1, 99])
        ax.imshow(np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1))
        ax.set_title(label, fontsize="small")
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_spectra_png(flux: np.ndarray, lam: np.ndarray, path: Path, title: str, truth: np.ndarray | None = None):
    """[n, W] flux + [W] wavelength -> overlaid flux-vs-wavelength PNG.

    With ``truth`` [W] the ground-truth spectrum is drawn in black behind
    the samples.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    if truth is not None:
        ax.plot(lam, truth, lw=1.2, color="black", alpha=0.9, label="truth")
    for i, f in enumerate(flux):
        ax.plot(lam, f, lw=0.7, alpha=0.8, label=f"sample {i}")
    ax.set_xlabel("wavelength [$\\AA$]")
    ax.set_ylabel("flux (model patch space)")
    ax.set_title(title)
    if len(flux) <= 8:
        ax.legend(fontsize="small")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    parser.add_argument("--norm-stats", default=None, help="data yaml with asinh percentiles")
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

    from astropt3.config_io import load_data_config, sequencer_kwargs_from_data_config
    from astropt3.data.packing import ObjectSequencer
    from astropt3.data.transforms import asinh_params_from_percentiles
    from astropt3.generation import generate, reconstruct
    from astropt3.tokenization import unpatchify_image, unpatchify_spectrum

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.checkpoint).to(device).eval()
    registry = model.config.modality_registry()

    sequencer_kwargs, asinh_params = {}, None
    if args.norm_stats:
        data_config = load_data_config(args.norm_stats)
        sequencer_kwargs = sequencer_kwargs_from_data_config(data_config)
        if sequencer_kwargs:
            asinh_params = asinh_params_from_percentiles(
                sequencer_kwargs["image_p1"],
                sequencer_kwargs["image_p99"],
                sequencer_kwargs["alpha"],
            )

    # sampling modes want the full skeleton (image + spectra spans)
    need_spectrum = args.mode != "reconstruct"
    record_indices = [int(i) for i in str(args.record_index).split(",")]
    sequencer = ObjectSequencer(model.config, **sequencer_kwargs)

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

    def maybe_sinh(imgs):
        if asinh_params is not None:
            offset, scale = asinh_params
            return torch.sinh(imgs) * scale.view(-1, 1, 1) + offset.view(-1, 1, 1)
        return imgs

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

        if args.mode == "reconstruct":
            sampled = {m: v.unsqueeze(0) for m, v in reconstruct(model, template).items()}
        else:
            if args.mode == "image-to-spectra":
                gen_modalities = {"spectra"}
            else:
                gen_modalities = set(template.masks)
            generator = torch.Generator(device=device).manual_seed(args.seed)
            sampled = generate(
                model,
                template,
                gen_modalities,
                n=args.n,
                temperature=args.temperature,
                argmax=args.argmax,
                generator=generator,
            )

        # the object id keeps figures from different records distinct on disk
        # and as wandb media keys, so a multi-record batch shares one run
        tag = f"{args.mode}_{template.object_id}_seed{args.seed}"
        for name, tokens in sampled.items():
            tokens = tokens.cpu()
            np.save(out_dir / f"{name}_{tag}.npy", tokens.numpy())
            mod = registry.get_config(name)
            png = out_dir / f"{name}_{tag}.png"
            if name == "images":
                side = int(round((tokens.shape[1]) ** 0.5)) * mod.patch_size
                channels = mod.input_size // (mod.patch_size**2)
                imgs = maybe_sinh(
                    torch.stack([unpatchify_image(t, mod.patch_size, channels, side) for t in tokens])
                )
                truth = None
                if show_truth:
                    truth = maybe_sinh(
                        unpatchify_image(template.values[name], mod.patch_size, channels, side).unsqueeze(0)
                    )[0].numpy()
                save_image_png(imgs.numpy(), png, f"{name} {tag}", truth=truth)
            elif name == "spectra":
                lam = np.asarray(record["spectrum"]["lambda"])
                flux = torch.stack(
                    [unpatchify_spectrum(t, len(lam)) for t in tokens]
                )
                truth = (
                    unpatchify_spectrum(template.values[name], len(lam)).numpy() if show_truth else None
                )
                save_spectra_png(flux.numpy(), lam, png, f"{name} {tag}", truth=truth)
            if wandb_run is not None:
                wandb_run.log({f"generation/{name}_{tag}": wandb.Image(str(png))})
            print(f"wrote {name}: {tuple(tokens.shape)} -> {out_dir}/{name}_{tag}.{{npy,png}}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()

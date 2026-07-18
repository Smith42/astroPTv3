"""Fixed-template sample rendering for converted HF checkpoints.

Shared by ``scripts/generate.py`` (one-off, any checkpoint) and
``scripts/run_probe_sweep.py`` (per-checkpoint evolution panels — see ADR
0003). The template record fixes the token skeleton and positions; keeping
the record(s) and the sampling seed fixed across a run's checkpoints means
the rendered panels differ only through the model weights.

Modes:
- ``unconditional``:      sample every span the template has (jetformer only).
- ``image-to-spectra``:   teacher-force the image tokens, sample the spectra
                          span (jetformer only).
- ``spectra-to-images``:  teacher-force the spectra tokens, sample the image
                          span; the template is built spectra-first so the
                          image span attends back to the spectrum. Only
                          meaningful for checkpoints trained with
                          ``shuffle_modality_order`` (ADR 0005 amendment) —
                          fixed-order checkpoints never saw spectra-first
                          sequences.
- ``reconstruct``:        one-step teacher-forced predictions for every span
                          (works for affine checkpoints too).

Both modalities are rendered through their physical inverse normalization
(images: band-registry keyed by the record's bands, the checkpoint's own
``image_norm_divisor`` knee; spectra: the DESI f_ν map with the checkpoint's
``spectra_norm_divisor`` knee, ADR 0007) — exact for jetformer checkpoints,
qualitative for affine ones (their sequencer's per-patch standardization
discards each patch's mean/std).
"""

import itertools
from pathlib import Path

import numpy as np
import torch

from ..data.band_registry import physical_inverse
from ..data.spectral import spectral_inverse
from ..data.packing import ObjectSequencer
from ..generation import generate, reconstruct
from ..tokenization import antispiralise, unpatchify_image, unpatchify_spectrum

MODES = ("unconditional", "image-to-spectra", "spectra-to-images", "reconstruct")


def build_template(sequencer, record: dict, mode: str):
    """The mode's template: spectra-first for ``spectra-to-images``.

    Explicit order (not the sequencer's parity rule) so the conditioning
    span always precedes the generated one under a causal mask.
    """
    if mode == "spectra-to-images":
        return sequencer.build(record, modality_order=["spectra", "images"])
    return sequencer.build(record)


def load_template_record(
    data_root: str, record_index: int, prefer_spectrum: bool, spectrum_only: bool = False
) -> dict:
    """The ``record_index``-th usable template record.

    ``prefer_spectrum`` is a preference, not a requirement: a corpus whose
    crossmatch kept the redshift labels but not the spectrum arrays (or one
    with no spectroscopic overlap at all, like ``shakeout_mix2``) carries
    none, and an image-only template still renders every mode except
    ``image-to-spectra``, which ``sample_checkpoint`` skips.

    ``spectrum_only=True`` selects the ADR-0005 spectrum-only rows (no
    image) so a sweep can track pure-spectrum generation panels; unlike
    ``prefer_spectrum`` this is a hard requirement — there is no image to
    fall back to.
    """
    if data_root == "synthetic":
        from ..data.synthetic import make_record

        if spectrum_only:
            return make_record(record_index, image_only_fraction=0.0, spectrum_only_fraction=1.0)
        return make_record(record_index, image_only_fraction=0.0 if prefer_spectrum else 0.3)
    from ..data.mmu import MMUIterableDataset

    # spectrum-only rows (ADR 0005) live in their own spectra/ shards, sorted
    # AFTER every crossmatched shard in the stream — read the subdir directly
    # rather than scanning the whole split to reach them; the record order
    # within it is unchanged, so indices stay stable
    spectra_dir = Path(data_root) / MMUIterableDataset.SPECTRA_SUBDIR
    root = spectra_dir if spectrum_only and spectra_dir.is_dir() else data_root
    dataset = MMUIterableDataset(root, rank=0, world_size=1, shuffle_buffer_size=0)
    if spectrum_only:
        wanted = (
            r
            for r in dataset
            if r.get("spectrum") is not None and r.get("image") is None
        )
        record = next(itertools.islice(wanted, record_index, None), None)
        if record is None:
            raise ValueError(
                f"fewer than {record_index + 1} spectrum-only records in {data_root}"
            )
        return record
    if not prefer_spectrum:
        record = next(itertools.islice(dataset, record_index, None), None)
        if record is None:
            raise ValueError(f"fewer than {record_index + 1} records in {data_root}")
        return record

    with_spectrum, image_only = [], []
    for record in dataset:
        if record.get("spectrum") is not None:
            with_spectrum.append(record)
            if len(with_spectrum) > record_index:
                return with_spectrum[record_index]
        elif len(image_only) <= record_index:
            image_only.append(record)
    if len(image_only) <= record_index:
        raise ValueError(f"fewer than {record_index + 1} records in {data_root}")
    print(
        f"[samples] no spectrum-bearing records in {data_root}; "
        f"falling back to image-only template {record_index}",
        flush=True,
    )
    return image_only[record_index]


def save_image_png(
    values: np.ndarray,
    path: Path,
    title: str,
    truth: np.ndarray | None = None,
    truth_label: str = "truth",
):
    """[n, C, H, W] -> one PNG grid (per-image normalized RGB).

    With ``truth`` [C, H, W] the ground-truth panel leads the grid.
    """
    import matplotlib.pyplot as plt

    panels = ([(truth_label, truth)] if truth is not None else []) + [
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


def save_spectra_png(
    flux: np.ndarray,
    lam: np.ndarray,
    path: Path,
    title: str,
    truth: np.ndarray | None = None,
    truth_label: str = "truth",
):
    """[n, W] flux + [W] wavelength -> one subplot per spectrum, stacked.

    With ``truth`` [W] the ground-truth spectrum leads the stack (mirroring
    the image grid); shared axes keep the panels comparable.
    """
    import matplotlib.pyplot as plt

    panels = ([(truth_label, truth)] if truth is not None else []) + [
        (f"sample {i}", f) for i, f in enumerate(flux)
    ]
    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(10, 2 * len(panels)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for ax, (label, f) in zip(axes[:, 0], panels):
        color = "black" if truth is not None and f is truth else None
        ax.plot(lam, f, lw=0.7, color=color)
        ax.set_title(label, fontsize="small")
        ax.set_ylabel("f$_\\lambda$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ $\\AA^{-1}$]")
    axes[-1, 0].set_xlabel("wavelength [$\\AA$]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def default_modes(config) -> list[str]:
    """The sampling modes a checkpoint supports (``generate`` is jetformer-only)."""
    if config.tokeniser != "jetformer":
        return ["reconstruct"]
    modes = ["unconditional"]
    if "spectra" in config.modality_registry().names():
        modes.append("image-to-spectra")
        # only checkpoints trained on both span orders can condition
        # images on spectra (ADR 0005 amendment)
        if getattr(config, "shuffle_modality_order", False):
            modes.append("spectra-to-images")
    return modes


def sample_template(
    model,
    template,
    mode: str,
    *,
    n: int = 4,
    temperature: float = 1.0,
    argmax: bool = False,
    generator: torch.Generator | None = None,
) -> dict:
    """Run one sampling mode against a template: ``{name: [n, T, D]}``."""
    if mode == "reconstruct":
        return {m: v.unsqueeze(0) for m, v in reconstruct(model, template).items()}
    if mode == "image-to-spectra":
        if "spectra" not in template.masks:
            raise ValueError("image-to-spectra needs a template record carrying a spectrum")
        gen_modalities = {"spectra"}
    elif mode == "spectra-to-images":
        if not {"images", "spectra"} <= set(template.masks):
            raise ValueError("spectra-to-images needs a template carrying both modalities")
        # conditioning only flows left to right under the causal mask
        if int(template.masks["spectra"].nonzero()[0]) > int(
            template.masks["images"].nonzero()[0]
        ):
            raise ValueError(
                "spectra-to-images needs a spectra-first template (build_template)"
            )
        gen_modalities = {"images"}
    elif mode == "unconditional":
        gen_modalities = set(template.masks)
    else:
        raise ValueError(f"unknown mode {mode!r} (expected one of {MODES})")
    return generate(
        model,
        template,
        gen_modalities,
        n=n,
        temperature=temperature,
        argmax=argmax,
        generator=generator,
    )


def render_sampled_tokens(
    model,
    record: dict,
    template,
    sampled: dict,
    *,
    out_dir: Path,
    tag: str,
    show_truth: bool,
    truth_label: str = "truth",
) -> dict:
    """Invert sampled tokens and write one PNG per modality: ``{name: path}``."""
    registry = model.config.modality_registry()
    # keys the physical inverse normalization back to survey flux
    bands = [str(b) for b in (record.get("image") or {}).get("band", [])]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pngs = {}
    for name, tokens in sampled.items():
        tokens = tokens.cpu().float()  # bf16 -> f32 so matplotlib/numpy can ingest
        mod = registry.get_config(name)
        png = out_dir / f"{name}_{tag}.png"
        if name == "images":
            side = int(round((tokens.shape[1]) ** 0.5)) * mod.patch_size
            channels = mod.input_size // (mod.patch_size**2)
            # the checkpoint's own arcsinh knee, so the inverse matches
            # the normalization its training data went through
            divisor = model.config.image_norm_divisor
            # a spiral checkpoint's tokens (sampled AND template) are in
            # spiral order; unpatchify expects raster, so undo the exact
            # order the checkpoint trained in (ADR 0004)
            spiral = getattr(model.config, "spiral", True)

            def to_pixels(t):
                return unpatchify_image(
                    antispiralise(t) if spiral else t, mod.patch_size, channels, side
                )

            imgs = physical_inverse(
                torch.stack([to_pixels(t) for t in tokens]), bands, divisor=divisor
            )
            truth = None
            if show_truth:
                truth = physical_inverse(
                    to_pixels(template.values[name].float()), bands, divisor=divisor
                ).numpy()
            save_image_png(imgs.numpy(), png, f"{name} {tag}", truth=truth, truth_label=truth_label)
        elif name == "spectra":
            lam = np.asarray(record["spectrum"]["lambda"])
            lam_t = torch.as_tensor(lam, dtype=torch.float32)
            # the checkpoint's own arcsinh knee, mirroring the image path
            divisor = model.config.spectra_norm_divisor
            flux = spectral_inverse(
                torch.stack([unpatchify_spectrum(t, len(lam)) for t in tokens]),
                lam_t,
                divisor=divisor,
            )
            truth = (
                spectral_inverse(
                    unpatchify_spectrum(template.values[name].float(), len(lam)),
                    lam_t,
                    divisor=divisor,
                ).numpy()
                if show_truth
                else None
            )
            save_spectra_png(flux.numpy(), lam, png, f"{name} {tag}", truth=truth, truth_label=truth_label)
        pngs[name] = png
    return pngs


def sample_checkpoint(
    checkpoint,
    records: list[dict],
    *,
    modes: list[str] | None = None,
    n: int = 4,
    temperature: float = 1.0,
    seed: int = 0,
    out_dir: Path,
    device=None,
    step: int | None = None,
) -> dict:
    """Sample + render every (record, mode) pair from a converted checkpoint.

    ``records`` are pre-loaded template records (load once per sweep, not per
    step). A fresh seeded generator per (record, mode) keeps the sampling
    noise identical at every checkpoint, so a run's panels differ only
    through the model. ``step``, when given, is folded into the PNG tag so
    filenames are self-identifying across steps. Returns
    ``{"{mode}/{name}/{object_id}": str(png)}``.
    """
    import astropt3  # noqa: F401  -- registers the Auto classes

    from transformers import AutoModel

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    model = AutoModel.from_pretrained(checkpoint).to(device=device, dtype=dtype).eval()
    if modes is None:
        modes = default_modes(model.config)
    sequencer = ObjectSequencer(model.config)

    pngs = {}
    for record in records:
        template = sequencer.build(record)
        for mode in modes:
            if mode in ("image-to-spectra", "spectra-to-images") and not (
                "spectra" in template.masks and "images" in template.masks
            ):
                continue
            # spectra-first skeleton for spectra-to-images; other modes keep
            # the default template
            mode_template = (
                build_template(sequencer, record, mode)
                if mode == "spectra-to-images"
                else template
            )
            generator = torch.Generator(device=device).manual_seed(seed)
            sampled = sample_template(
                model,
                mode_template,
                mode,
                n=n,
                temperature=temperature,
                generator=generator,
            )
            # unconditional samples aren't tied to the template's object, but
            # its record still makes a useful visual reference
            rendered = render_sampled_tokens(
                model,
                record,
                template,
                sampled,
                out_dir=out_dir,
                tag=(f"step{step}_" if step is not None else "")
            + f"{mode}_{template.object_id}_seed{seed}",
                show_truth=True,
                truth_label="truth (reference)" if mode == "unconditional" else "truth",
            )
            for name, png in rendered.items():
                pngs[f"{mode}/{name}/{template.object_id}"] = str(png)
    return pngs

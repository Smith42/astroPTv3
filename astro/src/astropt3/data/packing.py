"""Object -> token sequence assembly and greedy packing into fixed-length batches.

Per object (modalities in alphabetical registry order, only those present):

    <|bos|> <|begin_images|> p0 ... p360 <|end_images|>
            <|begin_spectra|> s0 ... s30 <|end_spectra|>

The collator packs whole objects greedily into rows of ``seq_len`` tokens;
objects are never split. ``position_ids`` restart at 0 on each object, which
is both the RoPE position and the packed-document boundary signal
(transformers' ``create_causal_mask`` builds the block-diagonal doc mask from
these restarts when ``attention_mask`` is None). Tail padding uses
``<|pad|>`` with position_id 0, so each pad token forms its own one-token
document and cannot attend to (or be attended by) real tokens.

Flattened ``modality_values``/``modality_positions`` are concatenated in
row-major (batch, time) order — the same order a boolean mask lookup
``tensor[mask]`` produces — so the model can align them without indices.
"""

from dataclasses import dataclass

import torch

from ..configuration_astropt3 import AstroPT3Config
from ..tokenization import (
    BOS_ID,
    PAD_ID,
    modality_token_ids,
    normalize_wavelength,
    patchify_image,
    patchify_spectrum,
    spiralise,
)
from .transforms import (
    ASINH_ALPHA,
    asinh_params_from_percentiles,
    asinh_stretch,
    per_patch_standardize,
)


@dataclass
class ObjectSeq:
    """One object's token sequence and its continuous payloads."""

    input_ids: torch.LongTensor  # [L]
    masks: dict  # name -> bool [L]
    values: dict  # name -> [n_m, input_size]
    positions: dict  # name -> long [n_m] or float [n_m, pos_input_size]
    object_id: str = ""

    def __len__(self) -> int:
        return len(self.input_ids)


class ObjectSequencer:
    """Turn an MMU-schema record into an :class:`ObjectSeq`."""

    def __init__(
        self,
        config: AstroPT3Config,
        image_p1=None,
        image_p99=None,
        alpha: float = ASINH_ALPHA,
        spiral: bool = False,
    ):
        self.registry = config.modality_registry()
        # Platonic Universe asinh stretch: per-band offset/scale from the 1st/99th
        # flux percentiles (scripts/compute_norm_stats.py, Phase 2). Without stats
        # (synthetic/smoke) fall back to plain asinh(flux).
        if image_p99 is None:
            self.asinh_offset, self.asinh_scale = 0.0, 1.0
        else:
            self.asinh_offset, self.asinh_scale = asinh_params_from_percentiles(
                image_p1, image_p99, alpha
            )
        self.spiral = spiral

    def _images_tokens(self, record: dict):
        mod = self.registry.get_config("images")
        flux = torch.as_tensor(record["image"]["flux"], dtype=torch.float32)
        flux = asinh_stretch(flux, self.asinh_scale, self.asinh_offset)
        patches = patchify_image(flux, mod.patch_size)
        patches = per_patch_standardize(patches)
        if self.spiral:
            patches = spiralise(patches)
        positions = torch.arange(len(patches), dtype=torch.long)
        return patches, positions

    def _spectra_tokens(self, record: dict):
        mod = self.registry.get_config("spectra")
        spec = record["spectrum"]
        flux = torch.as_tensor(spec["flux"], dtype=torch.float32)
        lam = torch.as_tensor(spec["lambda"], dtype=torch.float32)
        mask = torch.as_tensor(spec["mask"], dtype=torch.bool)
        flux = torch.where(mask, torch.zeros_like(flux), flux)
        patches, lam_mean = patchify_spectrum(flux, lam, mod.patch_size)
        patches = per_patch_standardize(patches)
        positions = normalize_wavelength(lam_mean).unsqueeze(-1)
        return patches, positions

    def build(self, record: dict) -> ObjectSeq:
        parts = {}
        for name in self.registry.names():
            if name == "images" and record.get("image") is not None:
                parts[name] = self._images_tokens(record)
            elif name == "spectra" and record.get("spectrum") is not None:
                parts[name] = self._spectra_tokens(record)
        if not parts:
            raise ValueError(f"record {record.get('object_id')!r} has no known modality")

        ids = [BOS_ID]
        spans = {}
        for name, (values, _) in parts.items():
            begin_id, placeholder_id, end_id = modality_token_ids(name)
            ids.append(begin_id)
            spans[name] = (len(ids), len(ids) + len(values))
            ids.extend([placeholder_id] * len(values))
            ids.append(end_id)

        input_ids = torch.tensor(ids, dtype=torch.long)
        masks, values, positions = {}, {}, {}
        for name, (vals, pos) in parts.items():
            start, stop = spans[name]
            m = torch.zeros(len(ids), dtype=torch.bool)
            m[start:stop] = True
            masks[name] = m
            values[name] = vals
            positions[name] = pos
        return ObjectSeq(
            input_ids=input_ids,
            masks=masks,
            values=values,
            positions=positions,
            object_id=str(record.get("object_id", "")),
        )


class PackedCollator:
    """Greedily pack whole :class:`ObjectSeq`s into fixed-length rows."""

    def __init__(self, config: AstroPT3Config, seq_len: int = 4096):
        self.seq_len = seq_len
        self.modality_names = config.modality_registry().names()

    def __call__(self, objects: list[ObjectSeq]) -> dict:
        rows: list[list[ObjectSeq]] = [[]]
        used = 0
        for obj in objects:
            if len(obj) > self.seq_len:
                raise ValueError(
                    f"object of length {len(obj)} exceeds seq_len {self.seq_len}"
                )
            if used + len(obj) > self.seq_len:
                rows.append([])
                used = 0
            rows[-1].append(obj)
            used += len(obj)
        if not rows[-1]:
            rows.pop()

        B, T = len(rows), self.seq_len
        input_ids = torch.full((B, T), PAD_ID, dtype=torch.long)
        position_ids = torch.zeros((B, T), dtype=torch.long)
        masks = {m: torch.zeros((B, T), dtype=torch.bool) for m in self.modality_names}
        values = {m: [] for m in self.modality_names}
        positions = {m: [] for m in self.modality_names}

        for b, row in enumerate(rows):
            t = 0
            for obj in row:
                L = len(obj)
                input_ids[b, t : t + L] = obj.input_ids
                position_ids[b, t : t + L] = torch.arange(L)
                for m in obj.masks:
                    masks[m][b, t : t + L] = obj.masks[m]
                    values[m].append(obj.values[m])
                    positions[m].append(obj.positions[m])
                t += L

        batch = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "modality_masks": {},
            "modality_values": {},
            "modality_positions": {},
        }
        for m in self.modality_names:
            if values[m]:
                batch["modality_masks"][m] = masks[m]
                batch["modality_values"][m] = torch.cat(values[m], dim=0)
                batch["modality_positions"][m] = torch.cat(positions[m], dim=0)
        return batch

"""Modality registry and the modules that move data in and out of embedding space.

Ported from astroPT (src/astropt/model.py) with the affine tokeniser as the
default. Each modality contributes three modules to the model:

- ``Encoder``:   data space  -> embedding space (one token per patch)
- ``Decoder``:   embedding space -> data space (the regression head)
- ``PositionEmbedder``: per-modality positional information, added to the
  input embeddings (SmolLM3's RoPE/NoPE over the flat sequence is unchanged).
"""

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModalityConfig:
    """Configuration for a single modality.

    Attributes:
        name: modality name; registry order (alphabetical) fixes sequence order.
        input_size: flattened patch vector length (e.g. 8*8*3 = 192 for images).
        patch_size: patch side length (images) or window length (1-D data).
        pos_type: "index" for learned integer-position embeddings,
            "continuous" for a projected float position (e.g. wavelength).
        pos_input_size: dimensionality of the continuous position vector.
        max_positions: number of learned positions when pos_type == "index".
        loss_weight: weight of this modality's Huber loss term.
        scalar: ADR 0008 scalar modality — a one-token span holding a
            physical quantity (Z, ebv, photometry). Scalars bypass the
            jetformer flow (their dims are odd and a flow buys nothing on a
            scalar) and are predicted by a ``GMMHead`` under BOTH tokenisers,
            with ``gmm_nll`` on the raw normalized value as the loss.
    """

    name: str
    input_size: int
    patch_size: int
    pos_type: str = "index"
    pos_input_size: int = 1
    max_positions: int = 1024
    loss_weight: float = 1.0
    scalar: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "pos_type": self.pos_type,
            "pos_input_size": self.pos_input_size,
            "max_positions": self.max_positions,
            "loss_weight": self.loss_weight,
            "scalar": self.scalar,
        }


class ModalityRegistry:
    """Central registry for model modalities.

    ``names()`` returns modalities sorted alphabetically; this order defines
    the per-object sequence order (as in astroPT).
    """

    def __init__(self, modalities):
        mods = [
            m if isinstance(m, ModalityConfig) else ModalityConfig(**m)
            for m in modalities
        ]
        self.modalities = {m.name: m for m in mods}

    def get_config(self, name: str) -> ModalityConfig:
        return self.modalities[name]

    def names(self) -> list[str]:
        return sorted(self.modalities.keys())

    def to_list(self) -> list[dict]:
        return [self.modalities[n].to_dict() for n in self.names()]


class Encoder(nn.Module):
    """Data space -> embedding space.

    "affine" (default) and "jetformer" both use a single linear projection;
    the flow that precedes jetformer lives on the model
    (``AstroPT3Model.flows``).
    """

    def __init__(self, hidden_size: int, in_size: int, tokeniser: str = "affine", bias: bool = False):
        super().__init__()
        if tokeniser not in ("affine", "jetformer"):
            raise ValueError(f"unknown tokeniser {tokeniser!r} (expected 'affine' or 'jetformer')")
        self.tokeniser = tokeniser
        self.c_fc = nn.Linear(in_size, hidden_size, bias=bias)

    def forward(self, x):
        return self.c_fc(x)


class Decoder(nn.Module):
    """Embedding space -> data space (the per-modality regression head)."""

    def __init__(self, hidden_size: int, out_size: int, tokeniser: str = "affine", bias: bool = False):
        super().__init__()
        if tokeniser != "affine":
            raise ValueError(f"unknown tokeniser {tokeniser!r} (Decoder supports only 'affine')")
        self.tokeniser = tokeniser
        self.c_fc = nn.Linear(hidden_size, out_size, bias=bias)

    def forward(self, x):
        return self.c_fc(x)


class PositionEmbedder(nn.Module):
    """Per-modality positional embedding, added to the input embeddings.

    pos_type "index": learned nn.Embedding over integer patch indices.
    pos_type "continuous": affine projection of a float position vector
    (e.g. normalized wavelength), so positions off the training grid embed
    sensibly.
    """

    def __init__(self, hidden_size: int, modality: ModalityConfig, bias: bool = False):
        super().__init__()
        self.pos_type = modality.pos_type
        if modality.pos_type == "index":
            self.embed = nn.Embedding(modality.max_positions, hidden_size)
        elif modality.pos_type == "continuous":
            self.embed = nn.Linear(modality.pos_input_size, hidden_size, bias=bias)
        else:
            raise ValueError(f"unknown pos_type {modality.pos_type!r}")

    def forward(self, pos):
        if self.pos_type == "index":
            return self.embed(pos)
        return self.embed(pos.to(self.embed.weight.dtype))


# --- "jetformer" tokeniser: per-token normalizing flow + GMM head -----------
#
# Ported from astroPT v2's sogol_branch (src/astropt/jetformer.py), which
# follows JetFormer (Tschannen et al. 2024) / GIVT (Tschannen et al. 2023).
# v2 flowed on raw image space (TinyFlow2D); here the flow runs on patch
# tokens instead, because the packed collator only ever hands the model flat
# (N, D) token values — this is also what makes the flow modality-generic
# (images and spectra alike). Per-modality loss is
# ``mean_tokens(NLL_GMM(z) - logdet)``: exact likelihood in (standardized)
# patch space, so it can be negative. The v2 noise curriculum lives on the
# model (``AstroPT3Model.set_jet_noise_frac``), not here.


class CouplingMLP(nn.Module):
    """RealNVP-style affine coupling over the feature dim of one token.

    flip=False transforms the second half conditioned on the first;
    flip=True the reverse. Requires an even feature dim.
    """

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.split = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.split, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * (dim - self.split)),
        )

    def forward(self, x, reverse: bool = False, flip: bool = False):
        x1 = x[..., : self.split]
        x2 = x[..., self.split :]
        ident, moved = (x2, x1) if flip else (x1, x2)
        s, t = self.net(ident).chunk(2, dim=-1)
        s = torch.tanh(s) * 1.5  # bound the scale for numerical stability
        if not reverse:
            moved = moved * torch.exp(s) + t
            logdet = s.sum(dim=-1)
        else:
            moved = (moved - t) * torch.exp(-s)
            logdet = -s.sum(dim=-1)
        halves = [moved, ident] if flip else [ident, moved]
        return torch.cat(halves, dim=-1), logdet


class TinyFlow1D(nn.Module):
    """Stack of affine couplings over (..., D) patch tokens.

    Returns the transformed tokens and a per-token log-determinant with
    shape ``x.shape[:-1]``. Invertible via ``reverse=True``.
    """

    def __init__(self, dim: int, steps: int = 4, hidden_dim: int = 128):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"TinyFlow1D requires an even token dim, got {dim}")
        self.blocks = nn.ModuleList(CouplingMLP(dim, hidden_dim) for _ in range(steps))

    def forward(self, x, reverse: bool = False):
        logdet = x.new_zeros(x.shape[:-1])
        indexed = list(enumerate(self.blocks))
        z = x
        for i, block in reversed(indexed) if reverse else indexed:
            z, ld = block(z, reverse=reverse, flip=(i % 2 == 1))
            logdet = logdet + ld
        return z, logdet


class GMMHead(nn.Module):
    """Embedding space -> diagonal Gaussian-mixture parameters per token.

    One linear layer emitting K mixture logits + K*(mu, log_sigma) pairs,
    as in GIVT/JetFormer.
    """

    def __init__(self, hidden_size: int, out_size: int, k: int, bias: bool = False):
        super().__init__()
        self.k = k
        self.d = out_size
        self.proj = nn.Linear(hidden_size, k * (1 + 2 * out_size), bias=bias)

    def forward(self, h):
        out = self.proj(h).view(*h.shape[:-1], self.k, 1 + 2 * self.d)
        logits_pi = out[..., 0]
        mu = out[..., 1 : 1 + self.d]
        log_sigma = out[..., 1 + self.d :].clamp(-7.0, 2.0)
        return logits_pi, mu, log_sigma


def gmm_nll(y, logits_pi, mu, log_sigma):
    """Per-token negative log-likelihood of y under the predicted GMM.

    y: (..., D); logits_pi: (..., K); mu/log_sigma: (..., K, D) -> (...,).
    """
    diff = y.unsqueeze(-2) - mu
    logp = (
        -0.5 * (diff.pow(2) * torch.exp(-2 * log_sigma)).sum(dim=-1)
        - log_sigma.sum(dim=-1)
        - 0.5 * mu.size(-1) * math.log(2 * math.pi)
    )
    return -torch.logsumexp(F.log_softmax(logits_pi, dim=-1) + logp, dim=-1)

"""Modality registry and the modules that move data in and out of embedding space.

Ported from astroPT (src/astropt/model.py) with the affine tokeniser as the
default. Each modality contributes three modules to the model:

- ``Encoder``:   data space  -> embedding space (one token per patch)
- ``Decoder``:   embedding space -> data space (the regression head)
- ``PositionEmbedder``: per-modality positional information, added to the
  input embeddings (SmolLM3's RoPE/NoPE over the flat sequence is unchanged).
"""

from dataclasses import dataclass, field

import torch.nn as nn


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
    """

    name: str
    input_size: int
    patch_size: int
    pos_type: str = "index"
    pos_input_size: int = 1
    max_positions: int = 1024
    loss_weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "pos_type": self.pos_type,
            "pos_input_size": self.pos_input_size,
            "max_positions": self.max_positions,
            "loss_weight": self.loss_weight,
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

    "affine" (default): a single linear projection.
    "aim": 2-layer MLP with tanh-GELU, kept for astroPT back-compatibility.
    """

    def __init__(self, hidden_size: int, in_size: int, tokeniser: str = "affine", bias: bool = False):
        super().__init__()
        self.tokeniser = tokeniser
        if tokeniser == "affine":
            self.c_fc = nn.Linear(in_size, hidden_size, bias=bias)
        elif tokeniser == "aim":
            self.c_fc = nn.Linear(in_size, 4 * hidden_size, bias=bias)
            self.gelu = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(4 * hidden_size, hidden_size, bias=bias)
        else:
            raise ValueError(f"unknown tokeniser {tokeniser!r} (expected 'affine' or 'aim')")

    def forward(self, x):
        if self.tokeniser == "affine":
            return self.c_fc(x)
        return self.c_proj(self.gelu(self.c_fc(x)))


class Decoder(nn.Module):
    """Embedding space -> data space (the per-modality regression head)."""

    def __init__(self, hidden_size: int, out_size: int, tokeniser: str = "affine", bias: bool = False):
        super().__init__()
        self.tokeniser = tokeniser
        if tokeniser == "affine":
            self.c_fc = nn.Linear(hidden_size, out_size, bias=bias)
        elif tokeniser == "aim":
            self.c_fc = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)
            self.gelu = nn.GELU(approximate="tanh")
            self.c_proj = nn.Linear(4 * hidden_size, out_size, bias=bias)
        else:
            raise ValueError(f"unknown tokeniser {tokeniser!r} (expected 'affine' or 'aim')")

    def forward(self, x):
        if self.tokeniser == "affine":
            return self.c_fc(x)
        return self.c_proj(self.gelu(self.c_fc(x)))


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

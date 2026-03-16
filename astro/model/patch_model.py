"""
AstroPatchModel
===============
Autoregressive next-patch prediction model built on the SmolLM transformer
backbone.  SigLIP is bypassed entirely — raw image patches are projected
directly into the LM's embedding space, predictions are regressed back to
patch space, and Huber loss is computed against shifted targets.

Architecture
------------
    patches  (B, 1024, 768)  — 512×512 image → 1024 patches of 16×16×3
         │
    patch_projector  [Linear(patch_dim → hidden_dim)]
         │
    prepend <galaxy_start> embed, drop last projected patch
         ↓
    [<galaxy_start>, proj(p0), proj(p1), ..., proj(p1022)]  (B, 1024, hidden_dim)
         │
    SmolLM transformer  [LlamaModel, causal, via inputs_embeds]
         │
    regression_head  [Linear(hidden_dim → patch_dim)]
         │
    preds  (B, 1024, patch_dim)
         │
    Huber loss vs patches  (B, 1024, 768)  — predicts p0…p1023

Boundary tokens
---------------
    Index 0: <galaxy_start>    (prepended before first galaxy patch)
    Index 1: <galaxy_end>      (appended after last galaxy patch — future use)
    Index 2: <spectrum_start>  (future multimodal use)
    Index 3: <spectrum_end>    (future multimodal use)

Stage 1 (connector warmup): freeze transformer, train projector + head only.
Stage 2 (full pretraining): unfreeze transformer.

Usage
-----
    from transformers import AutoModelForCausalLM
    from astro.model.patch_model import AstroPatchModel

    backbone = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    model = AstroPatchModel(backbone)
    out = model(patches_in=x, patches_target=y)
    out.loss.backward()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM


PATCH_DIM = 16 * 16 * 3   # 768  — matches GalaxyPatchDataset default

# Boundary token indices (shared nn.Embedding of size 4)
GALAXY_START   = 0
GALAXY_END     = 1
SPECTRUM_START = 2
SPECTRUM_END   = 3


@dataclass
class PatchModelOutput:
    """Return type for AstroPatchModel.forward — compatible with HF Trainer."""
    loss:        Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None


class AstroPatchModel(nn.Module):
    """
    SmolLM backbone repurposed for autoregressive patch regression.

    Parameters
    ----------
    backbone : nn.Module
        A loaded `LlamaForCausalLM` (or compatible) model.  Only its
        internal transformer (`backbone.model`) is used; the token
        embedding and LM head are replaced.
    patch_dim : int
        Flat patch dimension (16×16×3 = 768 by default).
    huber_delta : float
        Delta for the Huber (smooth L1) loss.
    """

    def __init__(
        self,
        backbone: nn.Module,
        patch_dim: int = PATCH_DIM,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.patch_dim   = patch_dim
        self.huber_delta = huber_delta

        # Extract the raw transformer (LlamaModel), leaving out the
        # token embedding and LM head.
        self.transformer = backbone.model   # LlamaModel

        hidden_dim = self.transformer.config.hidden_size

        # Input projection: raw patches → transformer embedding space (single linear, matches original AstroPT)
        self.patch_projector = nn.Linear(patch_dim, hidden_dim)

        # Modality boundary tokens: galaxy_start/end, spectrum_start/end
        self.boundary_tokens = nn.Embedding(4, hidden_dim)

        # Output regression head: hidden states → patch space
        self.regression_head = nn.Linear(hidden_dim, patch_dim, bias=True)

        # Expose config for HF Trainer compatibility
        self.config = backbone.config

    # ------------------------------------------------------------------
    # Freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def freeze_transformer(self):
        for p in self.transformer.parameters():
            p.requires_grad = False

    def unfreeze_transformer(self):
        for p in self.transformer.parameters():
            p.requires_grad = True

    def trainable_parameter_count(self) -> tuple[int, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        return trainable, total

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        patches: torch.Tensor,              # (B, N, patch_dim)  full patch sequence
        **kwargs,                           # absorb extra Trainer kwargs
    ) -> PatchModelOutput:
        B = patches.shape[0]

        # Project all patches into transformer embedding space
        proj = self.patch_projector(patches)   # (B, N, hidden_dim)

        # Prepend <galaxy_start> boundary token, drop the last projected patch.
        # Input:  [<galaxy_start>, proj(p0), ..., proj(p_{N-2})]  length N
        # Target: [p0,             p1,       ..., p_{N-1}       ]  length N
        galaxy_start = self.boundary_tokens(
            torch.full((B, 1), GALAXY_START, dtype=torch.long, device=patches.device)
        )  # (B, 1, hidden_dim)
        embeds = torch.cat([galaxy_start, proj[:, :-1, :]], dim=1)  # (B, N, hidden_dim)

        # Run the causal transformer — pass inputs_embeds to bypass token embedding
        transformer_out = self.transformer(
            inputs_embeds=embeds,
            use_cache=False,
        )
        hidden = transformer_out.last_hidden_state  # (B, N, hidden_dim)

        # Regress back to patch space
        preds = self.regression_head(hidden)        # (B, N, patch_dim)

        # Huber loss: each position predicts the corresponding target patch
        loss = F.huber_loss(preds, patches, delta=self.huber_delta)

        return PatchModelOutput(loss=loss, predictions=preds)

    # ------------------------------------------------------------------
    # HF Trainer expects model(return_dict=True) to work and loss to be
    # accessible.  The dataclass satisfies this.
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str, **kwargs):
        """Save full model state dict + config for resumption."""
        import os, json
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "patch_model.pt"))
        # Save backbone config for re-loading
        self.config.save_pretrained(path)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        backbone_name: Optional[str] = None,
        **kwargs,
    ) -> "AstroPatchModel":
        """Load a saved AstroPatchModel checkpoint."""
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(path)
        backbone_id = backbone_name or cfg._name_or_path
        backbone = AutoModelForCausalLM.from_pretrained(backbone_id, config=cfg)
        model = cls(backbone, **kwargs)
        state = torch.load(
            f"{path}/patch_model.pt", map_location="cpu", weights_only=True
        )
        model.load_state_dict(state)
        return model


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def build_patch_model(
    model_name: str = "HuggingFaceTB/SmolLM2-135M",
    patch_dim: int = PATCH_DIM,
    huber_delta: float = 1.0,
    torch_dtype=torch.bfloat16,
    freeze_transformer: bool = True,
) -> AstroPatchModel:
    """
    Load a SmolLM backbone and wrap it for patch autoregression.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID for the SmolLM backbone.
    freeze_transformer : bool
        If True (Stage 1), only patch_projector + regression_head are trained.
        If False (Stage 2), all parameters are trained.
    """
    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_cache=False,
    )
    model = AstroPatchModel(backbone, patch_dim=patch_dim, huber_delta=huber_delta)

    if freeze_transformer:
        model.freeze_transformer()

    trainable, total = model.trainable_parameter_count()
    print(
        f"[AstroPatchModel] {model_name}  |  "
        f"trainable: {trainable:,} / {total:,}  "
        f"({100*trainable/total:.1f}%)"
    )
    return model

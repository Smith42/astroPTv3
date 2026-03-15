"""
AstroPatchModel
===============
Autoregressive next-patch prediction model built on the SmolLM transformer
backbone.  SigLIP is bypassed entirely — raw image patches are projected
directly into the LM's embedding space, predictions are regressed back to
patch space, and Huber loss is computed against shifted targets.

Architecture
------------
    patches_in  (B, N-1, patch_dim)
         │
    patch_projector  [Linear(patch_dim → hidden_dim)]
         │
    SmolLM transformer  [LlamaModel, causal, via inputs_embeds]
         │
    regression_head  [Linear(hidden_dim → patch_dim)]
         │
    preds  (B, N-1, patch_dim)
         │
    Huber loss vs patches_target  (B, N-1, patch_dim)

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

        # Input projection: raw patches → transformer embedding space
        self.patch_projector = nn.Sequential(
            nn.Linear(patch_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

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
        patches_in:     torch.Tensor,                    # (B, N-1, patch_dim)
        patches_target: Optional[torch.Tensor] = None,  # (B, N-1, patch_dim)
        **kwargs,                                        # absorb extra Trainer kwargs
    ) -> PatchModelOutput:

        # Project patches into transformer embedding space
        embeds = self.patch_projector(patches_in)   # (B, N-1, hidden_dim)

        # Run the causal transformer — pass inputs_embeds to bypass token embedding
        transformer_out = self.transformer(
            inputs_embeds=embeds,
            use_cache=False,
        )
        hidden = transformer_out.last_hidden_state  # (B, N-1, hidden_dim)

        # Regress back to patch space
        preds = self.regression_head(hidden)        # (B, N-1, patch_dim)

        loss = None
        if patches_target is not None:
            loss = F.huber_loss(preds, patches_target, delta=self.huber_delta)

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

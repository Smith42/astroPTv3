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
    prepend <|begin_images|> embed (from backbone embed_tokens), drop last projected patch
         ↓
    [<|begin_images|>, proj(p0), ..., proj(p1022)]  (B, 1024, hidden_dim)
         │
    SmolLM transformer  [LlamaModel, causal, via inputs_embeds]
         │
    regression_head  [Linear(hidden_dim → patch_dim)]
         │
    preds  (B, 1024, patch_dim)
         │
    Huber loss vs patches  (B, 1024, 768)  — predicts p0…p1023

Special tokens  (matching original AstroPT naming)
--------------
    <|begin_images|>    prepended before first galaxy patch
    <|end_images|>      appended after last galaxy patch (future multimodal use)
    <|images|>          per-patch placeholder (future interleaved-text use)
    <|begin_spectra|>   future multimodal use
    <|end_spectra|>     future multimodal use
    <|spectra|>         future multimodal use

Stage 1 (connector warmup): freeze transformer, train projector + head only.
Stage 2 (full pretraining): unfreeze transformer.

Usage
-----
    from transformers import AutoTokenizer
    from astro.model.patch_model import build_patch_model

    model, tokenizer = build_patch_model("HuggingFaceTB/SmolLM2-135M")
    out = model(patches=x)
    out.loss.backward()
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


PATCH_DIM = 16 * 16 * 3   # 768  — matches GalaxyPatchDataset default

# Special token strings — match original AstroPT naming convention exactly.
# Format: <|begin_{modality}|>, <|{modality}|>, <|end_{modality}|>
ASTROPT_SPECIAL_TOKENS: list[str] = [
    "<|begin_images|>",
    "<|images|>",
    "<|end_images|>",
    "<|begin_spectra|>",
    "<|spectra|>",
    "<|end_spectra|>",
    "<|begin_metadata|>",
    "<|metadata|>",
    "<|end_metadata|>",
]


def add_astropt_tokens(
    tokenizer,
    backbone: nn.Module,
) -> dict[str, int]:
    """
    Add AstroPT special tokens to *tokenizer* and resize *backbone*'s token
    embedding table to match.  Idempotent — already-present tokens are skipped.

    Returns
    -------
    dict mapping each token string to its integer token ID.
    """
    tokens_to_add = [t for t in ASTROPT_SPECIAL_TOKENS if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        backbone.resize_token_embeddings(len(tokenizer))
    return {t: tokenizer.convert_tokens_to_ids(t) for t in ASTROPT_SPECIAL_TOKENS}


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
        A loaded `LlamaForCausalLM` (or compatible) model whose token embedding
        table has already been resized to include AstroPT special tokens via
        `add_astropt_tokens`.
    special_token_ids : dict[str, int]
        Mapping from AstroPT token strings to their integer IDs in the
        (possibly resized) backbone vocabulary.
    patch_dim : int
        Flat patch dimension (16×16×3 = 768 by default).
    huber_delta : float
        Delta for the Huber (smooth L1) loss.
    """

    def __init__(
        self,
        backbone: nn.Module,
        special_token_ids: dict[str, int],
        patch_dim: int = PATCH_DIM,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.patch_dim        = patch_dim
        self.huber_delta      = huber_delta
        self.special_token_ids = special_token_ids

        # Extract the raw transformer (LlamaModel).
        # backbone.model.embed_tokens holds the (resized) token embedding table;
        # we use it directly to look up boundary token embeddings — the same
        # pattern as original AstroPT's llm.get_input_embeddings().
        self.transformer = backbone.model   # LlamaModel

        hidden_dim = self.transformer.config.hidden_size

        # Input projection: raw patches → transformer embedding space.
        # Single linear layer — matches original AstroPT.
        self.patch_projector = nn.Linear(patch_dim, hidden_dim)

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
        patches: torch.Tensor,   # (B, N, patch_dim)  full patch sequence
        **kwargs,                 # absorb extra Trainer kwargs
    ) -> PatchModelOutput:
        B, N, _ = patches.shape

        # Project all patches into transformer embedding space
        proj = self.patch_projector(patches)   # (B, N, hidden_dim)

        # Look up <|begin_images|> embedding from backbone's token table.
        # This is the same approach as original AstroPT: boundary tokens live in
        # the LLM's own embedding space so they share the same representation.
        begin_id = self.special_token_ids["<|begin_images|>"]
        begin_embed = self.transformer.embed_tokens(
            torch.full((B, 1), begin_id, dtype=torch.long, device=patches.device)
        )  # (B, 1, hidden_dim)

        # Autoregressive shift:
        #   input:  [<|begin_images|>, proj(p0), ..., proj(p_{N-2})]  length N
        #   target: [p0,               p1,       ..., p_{N-1}       ]  length N
        embeds = torch.cat([begin_embed, proj[:, :-1, :]], dim=1)  # (B, N, hidden_dim)

        # Run the causal transformer via inputs_embeds (bypasses token embedding)
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
    # Save / load
    # ------------------------------------------------------------------

    def save_pretrained(self, path: str, tokenizer=None, **kwargs):
        """Save full model state dict + config + special token IDs (+ tokenizer if provided)."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "patch_model.pt"))
        self.config.save_pretrained(path)
        with open(os.path.join(path, "special_token_ids.json"), "w") as f:
            json.dump(self.special_token_ids, f)
        if tokenizer is not None:
            tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        backbone_name: Optional[str] = None,
        **kwargs,
    ) -> "AstroPatchModel":
        """Load a saved AstroPatchModel checkpoint."""
        cfg = AutoConfig.from_pretrained(path)
        backbone_id = backbone_name or cfg._name_or_path

        with open(os.path.join(path, "special_token_ids.json")) as f:
            special_token_ids = json.load(f)

        # Load tokenizer from checkpoint (has the special tokens already added)
        # and use it to resize the backbone embedding table correctly.
        tokenizer = AutoTokenizer.from_pretrained(path)
        backbone = AutoModelForCausalLM.from_pretrained(backbone_id, config=cfg)
        backbone.resize_token_embeddings(len(tokenizer))

        model = cls(backbone, special_token_ids=special_token_ids, **kwargs)
        state = torch.load(
            os.path.join(path, "patch_model.pt"), map_location="cpu", weights_only=True
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
) -> tuple["AstroPatchModel", object]:
    """
    Load a SmolLM backbone, add AstroPT special tokens, and wrap it for
    autoregressive patch regression.

    Returns
    -------
    (model, tokenizer)
        The tokenizer has the AstroPT special tokens added and should be saved
        alongside any checkpoint.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    backbone = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        use_cache=False,
    )

    special_token_ids = add_astropt_tokens(tokenizer, backbone)
    print(
        f"[AstroPatchModel] Added {len(ASTROPT_SPECIAL_TOKENS)} AstroPT special tokens; "
        f"vocab size: {len(tokenizer)}"
    )

    model = AstroPatchModel(
        backbone,
        special_token_ids=special_token_ids,
        patch_dim=patch_dim,
        huber_delta=huber_delta,
    )

    if freeze_transformer:
        model.freeze_transformer()

    trainable, total = model.trainable_parameter_count()
    print(
        f"[AstroPatchModel] {model_name}  |  "
        f"trainable: {trainable:,} / {total:,}  "
        f"({100*trainable/total:.1f}%)"
    )
    return model, tokenizer

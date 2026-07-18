"""AstroPT3: SmolLM3 decoder body with continuous-modality inputs and
per-modality regression heads.

Sequence contract (built by data/packing.py):
- ``input_ids`` [B, T] hold special-token ids everywhere; modality slots carry
  that modality's placeholder id, so ``embed_tokens`` provides a learned
  modality-type embedding at those positions.
- The continuous content is ADDED on top: slot embedding =
  ``embed_tokens(<|m|>) + encoder_m(value) + pos_embed_m(position)``.
- ``position_ids`` restart at 0 at each object; with ``attention_mask=None``
  transformers' ``create_causal_mask`` detects the packed format and applies
  block-diagonal (doc) masking (torch >= 2.6).
- Loss: Huber on each modality span, predictions read one position LEFT of
  each modality token (``<|begin_m|>`` predicts patch 0 — astroPT's
  ``starts-1`` alignment), weighted by ``loss_weight`` and averaged over the
  modalities present in the batch. No loss on special or pad tokens unless
  ``special_token_ce_weight > 0``.
- ``tokeniser: jetformer`` swaps the regression heads for per-modality
  normalizing flow + GMM heads (JetFormer/GIVT-style): patch values pass
  through ``flows[m]`` to a latent z that is both embedded and predicted,
  and the per-modality loss becomes ``mean(NLL_GMM(z) - logdet)`` — an exact
  likelihood in patch space (may be negative). Same left-shift alignment.
- ADR 0008 scalar modalities (``ModalityConfig.scalar``) are GMM-headed
  under BOTH tokenisers (``scalar_gmm_k`` mixture over the raw normalized
  value; loss is plain ``gmm_nll``, no flow, no logdet — a scalar needs no
  invertibility machinery and its odd dim cannot feed ``TinyFlow1D``).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput
from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3Model,
    SmolLM3PreTrainedModel,
)

from .configuration_astropt3 import AstroPT3Config
from .modalities import (
    Decoder,
    Encoder,
    GMMHead,
    PositionEmbedder,
    TinyFlow1D,
    gmm_nll,
)
from .tokenization import PAD_ID


@dataclass
class AstroPT3Output(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    modality_losses: Optional[dict] = None
    predictions: Optional[dict] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    # per-layer states (embeddings first, HF convention), populated only
    # when forward(output_hidden_states=True); the linear probe pools the
    # central layer from here (astroPT convention)
    hidden_states: Optional[tuple] = None


def left_shift_mask(mask: torch.Tensor) -> torch.Tensor:
    """Positions whose NEXT token is a masked-True position.

    Given a [B, T] boolean modality mask, returns a [B, T] mask that is True
    at position t iff mask[t+1] is True (last column always False). Hidden
    states at these positions predict the modality values at t+1.
    """
    shifted = torch.zeros_like(mask)
    shifted[:, :-1] = mask[:, 1:]
    return shifted


class AstroPT3Model(SmolLM3PreTrainedModel):
    config_class = AstroPT3Config

    def __init__(self, config: AstroPT3Config):
        super().__init__(config)
        self.model = SmolLM3Model(config)
        registry = config.modality_registry()
        self.modality_names = registry.names()
        # ADR 0008: scalar modalities are GMM-headed under both tokenisers
        # and never flow — one-token spans over raw normalized values
        self.scalar_names = {
            name
            for name in self.modality_names
            if getattr(registry.get_config(name), "scalar", False)
        }
        patch_names = [n for n in self.modality_names if n not in self.scalar_names]
        self.encoders = nn.ModuleDict(
            {
                name: Encoder(config.hidden_size, registry.get_config(name).input_size, config.tokeniser)
                for name in self.modality_names
            }
        )
        if config.tokeniser == "jetformer":
            # Per-modality flow (data -> latent z, with logdet) and GMM head;
            # the head replaces the regression Decoder under the same name so
            # the loss path below stays a single branch.
            self.flows = nn.ModuleDict(
                {
                    name: TinyFlow1D(
                        registry.get_config(name).input_size,
                        steps=config.jetformer_flow_steps,
                        hidden_dim=config.jetformer_flow_hidden,
                    )
                    for name in patch_names
                }
            )
            self.decoders = nn.ModuleDict(
                {
                    name: GMMHead(
                        config.hidden_size,
                        registry.get_config(name).input_size,
                        config.jetformer_gmm_k,
                    )
                    for name in patch_names
                }
            )
        else:
            self.decoders = nn.ModuleDict(
                {
                    name: Decoder(config.hidden_size, registry.get_config(name).input_size, config.tokeniser)
                    for name in patch_names
                }
            )
        for name in sorted(self.scalar_names):
            self.decoders[name] = GMMHead(
                config.hidden_size,
                registry.get_config(name).input_size,
                getattr(config, "scalar_gmm_k", 5),
            )
        self.pos_embeds = nn.ModuleDict(
            {
                name: PositionEmbedder(config.hidden_size, registry.get_config(name))
                for name in self.modality_names
            }
        )
        if config.special_token_ce_weight > 0:
            self.special_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Noise-curriculum position (jetformer, v2 sogol_branch semantics):
        # sigma = noise_max + (noise_min - noise_max) * frac, so frac 0 -> 1
        # anneals noise_max -> noise_min over training. Default 1.0 means
        # sigma = noise_min (= 0 by default): eval and existing checkpoints
        # are unaffected unless the trainer drives the curriculum.
        self.jet_noise_frac = 1.0
        self.post_init()

    def set_jet_noise_frac(self, frac: float):
        self.jet_noise_frac = float(min(max(frac, 0.0), 1.0))

    def _jet_noise_sigma(self) -> float:
        cfg = self.config
        return cfg.jetformer_noise_max + (
            cfg.jetformer_noise_min - cfg.jetformer_noise_max
        ) * self.jet_noise_frac

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def assemble_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        modality_values: dict,
        modality_masks: dict,
        modality_positions: dict,
    ) -> torch.FloatTensor:
        embeds = self.model.embed_tokens(input_ids)
        delta = torch.zeros_like(embeds)
        for name in self.modality_names:
            if name not in modality_masks or not modality_masks[name].any():
                continue
            mask = modality_masks[name]
            values = modality_values[name].to(embeds.dtype)
            content = self.encoders[name](values) + self.pos_embeds[name](modality_positions[name])
            delta = delta.index_put((mask,), content.to(embeds.dtype))
        return embeds + delta

    def forward(
        self,
        input_ids: torch.LongTensor,
        modality_values: Optional[dict] = None,
        modality_masks: Optional[dict] = None,
        modality_positions: Optional[dict] = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        compute_loss: bool = True,
        output_hidden_states: bool = False,
        **kwargs,
    ) -> AstroPT3Output:
        modality_values = modality_values or {}
        modality_masks = modality_masks or {}
        modality_positions = modality_positions or {}

        # jetformer: values -> flow -> latent z once, up front; z is both the
        # embedded input and the GMM target, and logdet joins the loss. The
        # noise curriculum perturbs only the embedded copy (training mode) —
        # the GMM target z and the logdet stay clean.
        flow_logdets = {}
        embed_values = modality_values
        if self.config.tokeniser == "jetformer" and modality_values:
            flowed = {}
            for name in self.modality_names:
                # scalars bypass the flow: their raw normalized value is both
                # the embedded input and the GMM target (ADR 0008)
                if name not in modality_values or name in self.scalar_names:
                    continue
                flow = self.flows[name]
                values = modality_values[name].to(next(flow.parameters()).dtype)
                flowed[name], flow_logdets[name] = flow(values)
            modality_values = {**modality_values, **flowed}
            embed_values = modality_values
            sigma = self._jet_noise_sigma()
            if self.training and sigma > 0:
                embed_values = {
                    **modality_values,
                    **{
                        name: z + sigma * torch.randn_like(z)
                        for name, z in flowed.items()
                    },
                }

        inputs_embeds = self.assemble_inputs_embeds(
            input_ids, embed_values, modality_masks, modality_positions
        )
        body_out = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=output_hidden_states,
        )
        hidden = body_out.last_hidden_state

        predictions = {}
        modality_losses = {}
        loss = None
        if compute_loss:
            total = hidden.new_zeros(())
            n_present = 0
            registry = self.config.modality_registry()
            for name in self.modality_names:
                if name not in modality_masks or not modality_masks[name].any():
                    continue
                mask = modality_masks[name]
                if mask[:, 0].any():
                    raise ValueError(
                        f"modality '{name}' token at sequence position 0 has no "
                        "preceding token to predict it from (missing <|bos|>?)"
                    )
                if name in self.scalar_names:
                    # ADR 0008: GMM NLL on the raw normalized scalar, both
                    # tokenisers — no flow, no logdet
                    logits_pi, mu, log_sigma = self.decoders[name](
                        hidden[left_shift_mask(mask)]
                    )
                    target = modality_values[name].to(mu.dtype)
                    mod_loss = gmm_nll(target, logits_pi, mu, log_sigma).mean()
                    pi = torch.softmax(logits_pi, dim=-1)
                    predictions[name] = (pi.unsqueeze(-1) * mu).sum(dim=-2)
                elif self.config.tokeniser == "jetformer":
                    logits_pi, mu, log_sigma = self.decoders[name](
                        hidden[left_shift_mask(mask)]
                    )
                    target = modality_values[name].to(mu.dtype)  # latent z
                    nll = gmm_nll(target, logits_pi, mu, log_sigma)
                    mod_loss = (nll - flow_logdets[name].to(nll.dtype)).mean()
                    # z-space mixture mean as the point prediction; inverse
                    # flow (self.flows[name](z, reverse=True)) maps it back.
                    pi = torch.softmax(logits_pi, dim=-1)
                    predictions[name] = (pi.unsqueeze(-1) * mu).sum(dim=-2)
                else:
                    pred = self.decoders[name](hidden[left_shift_mask(mask)])
                    target = modality_values[name].to(pred.dtype)
                    mod_loss = F.huber_loss(pred, target, delta=self.config.huber_delta)
                    predictions[name] = pred
                modality_losses[name] = mod_loss
                total = total + registry.get_config(name).loss_weight * mod_loss
                n_present += 1
            loss = total / max(n_present, 1)

            if self.config.special_token_ce_weight > 0:
                logits = self.special_head(hidden[:, :-1])
                targets = input_ids[:, 1:]
                ce = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size),
                    targets.reshape(-1),
                    ignore_index=PAD_ID,
                )
                modality_losses["special_ce"] = ce
                loss = loss + self.config.special_token_ce_weight * ce

        return AstroPT3Output(
            loss=loss,
            modality_losses=modality_losses,
            predictions=predictions,
            last_hidden_state=hidden,
            hidden_states=getattr(body_out, "hidden_states", None),
        )

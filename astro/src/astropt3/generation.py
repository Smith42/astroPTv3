"""Autoregressive sampling from a jetformer checkpoint (GIVT recipe).

The model predicts a GMM over the NEXT token's latent z (the ``starts-1``
alignment: the hidden state one position left of each modality slot). To
sample a span we walk the object's token skeleton left to right and, at each
slot of a generated modality, forward the prefix, sample z from the GMM at
the last hidden state, and invert the flow to data space — the sampled
data-space token then re-enters the next forward exactly like a real one
(the model's own forward re-applies the flow, reproducing z).

Sampling follows GIVT: a categorical draw over the mixture weights, then a
Gaussian draw from the chosen component; ``temperature`` scales sigma (the
GMM analogue of softmax temperature) and ``argmax=True`` takes the
mixture-mean point sample instead.

The token loop re-runs the full forward per generated token — no KV cache
(``use_cache`` stays False). ponytail: O(T^2) at ~400 tokens is trivial; add
caching only if generation ever becomes a hot path.
"""

import torch

from .data.packing import ObjectSeq, PackedCollator


def sample_gmm(logits_pi, mu, log_sigma, *, temperature=1.0, argmax=False, generator=None):
    """One draw per row from a diagonal GMM ([n, K], [n, K, D] -> [n, D])."""
    pi = torch.softmax(logits_pi, dim=-1)
    if argmax:
        return (pi.unsqueeze(-1) * mu).sum(dim=-2)
    k = torch.multinomial(pi, 1, generator=generator)  # [n, 1]
    idx = k.unsqueeze(-1).expand(-1, 1, mu.size(-1))
    mu_k = mu.gather(-2, idx).squeeze(-2)
    sigma_k = log_sigma.gather(-2, idx).squeeze(-2).exp()
    eps = torch.randn(mu_k.shape, generator=generator, device=mu_k.device, dtype=mu_k.dtype)
    return mu_k + temperature * sigma_k * eps


@torch.no_grad()
def generate(
    model,
    template: ObjectSeq,
    generate_modalities,
    *,
    n: int = 1,
    temperature: float = 1.0,
    argmax: bool = False,
    generator: torch.Generator | None = None,
) -> dict:
    """Sample ``n`` versions of the template's ``generate_modalities`` spans.

    ``template`` fixes the token skeleton and the positions (image indices /
    spectra wavelengths); modalities NOT in ``generate_modalities`` are
    teacher-forced from the template's values (image-to-spectra mode). Pass
    every modality to sample unconditionally. Returns
    ``{name: [n, n_tokens, input_size]}`` in data (standardized-patch) space.
    """
    if model.config.tokeniser != "jetformer":
        raise ValueError("generate() samples from GMM heads; checkpoint is not jetformer")
    generate_modalities = set(generate_modalities)
    unknown = generate_modalities - set(template.masks)
    if unknown:
        raise ValueError(f"template has no {sorted(unknown)} span to generate")
    model.eval()
    device = next(model.parameters()).device

    ids = template.input_ids.to(device)
    masks = {m: mask.to(device) for m, mask in template.masks.items()}
    positions = {m: pos.to(device) for m, pos in template.positions.items()}
    # data-space tokens so far, [n, count_m, D]; generated spans start empty
    values = {}
    for m in masks:
        v = template.values[m].to(device)
        if m in generate_modalities:
            values[m] = v.new_empty(n, 0, v.shape[-1])
        else:
            values[m] = v.unsqueeze(0).expand(n, -1, -1)

    def prefix_kwargs(t):
        kw = {
            "input_ids": ids[:t].unsqueeze(0).expand(n, -1),
            "position_ids": torch.arange(t, device=device).unsqueeze(0).expand(n, -1),
            "modality_masks": {},
            "modality_values": {},
            "modality_positions": {},
        }
        for m, mask in masks.items():
            cnt = int(mask[:t].sum())
            if cnt == 0:
                continue
            kw["modality_masks"][m] = mask[:t].unsqueeze(0).expand(n, -1)
            # [n, cnt, D] -> [n*cnt, D]: row-major (batch, time), the
            # flattening order the model's mask indexing expects
            kw["modality_values"][m] = values[m][:, :cnt].reshape(n * cnt, -1)
            pos = positions[m][:cnt]
            kw["modality_positions"][m] = pos.repeat(n) if pos.dim() == 1 else pos.repeat(n, 1)
        return kw

    for t in range(1, len(ids)):
        m_t = next((m for m in masks if masks[m][t]), None)
        if m_t is None or m_t not in generate_modalities:
            continue
        out = model(**prefix_kwargs(t), compute_loss=False)
        logits_pi, mu, log_sigma = model.decoders[m_t](out.last_hidden_state[:, -1])
        z = sample_gmm(
            logits_pi, mu, log_sigma, temperature=temperature, argmax=argmax, generator=generator
        )
        x, _ = model.flows[m_t](z, reverse=True)
        values[m_t] = torch.cat([values[m_t], x.unsqueeze(1)], dim=1)

    return {m: values[m] for m in generate_modalities}


@torch.no_grad()
def reconstruct(model, template: ObjectSeq) -> dict:
    """One-step teacher-forced predictions for every span, in data space.

    Works for affine checkpoints too (predictions are already data-space);
    jetformer point predictions (z-space mixture means) go back through the
    inverse flow. Returns ``{name: [n_tokens, input_size]}``.
    """
    model.eval()
    device = next(model.parameters()).device
    batch = PackedCollator(model.config, seq_len=len(template))([template])
    batch = {
        k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device))
        for k, v in batch.items()
    }
    out = model(**batch)
    preds = {}
    for m, pred in out.predictions.items():
        if model.config.tokeniser == "jetformer":
            pred, _ = model.flows[m](pred, reverse=True)
        preds[m] = pred
    return preds

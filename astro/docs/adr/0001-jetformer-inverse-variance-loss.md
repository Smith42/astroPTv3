# ADR 0001: Inverse-variance-weighted loss for the jetformer head

- **Status:** Rejected
- **Date:** 2026-07-15
- **References:**
  `../galactiktok/docs/adr/0002-inverse-variance-weighted-loss.md`
  (sibling repo — accepted there for an MSE-reconstruction image tokenizer)

## Question

Is the inverse-variance-weighted per-channel reduced-χ² loss from
`galactiktok` ADR 0002 worth implementing for the `tokeniser: jetformer`
model in this repo?

## Context

`galactiktok` ADR 0002 replaces a plain arcsinh-space MSE with a
per-channel reduced-χ²:

```
chi2_c = Σ [ m·valid·w·(ŷ − y)² ] / Σ [ m·valid ],   w = clamp(ivar, 0, clip)
L      = mean over active bands of chi2_c
```

motivated by heteroscedastic astronomical images: sky-dominated pixels
carry little information, surveys ship `ivar = 1/σ²`, and an unweighted
MSE spends capacity reproducing noise. It also buys band-subset
invariance (each active band contributes `1/C`) and stays consistent with
`spectrum_jeff.loss_fn` in that repo.

This repo has a second loss path: the `tokeniser: jetformer` option
(`modeling_astropt3.py`), which replaces the affine `Decoder` + Huber
head with a per-modality `TinyFlow1D` + `GMMHead` and an exact
patch-space likelihood:

```
mod_loss = mean( NLL_GMM(z) - logdet ),   z = flow_m(patch_values)
```

The flow is invertible; `generation.py` samples `z` from the GMM and
runs `flow(z, reverse=True)` back to data space.

## Decision

**Do not implement ADR 0002 for the jetformer head.** The objective it
corrects for is already exact-likelihood, and the conditions that make
the ADR viable in `galactiktok` do not hold here.

## Rationale

1. **The ADR fixes a point-prediction loss; jetformer isn't one.**
   "Unweighted MSE spends capacity reproducing noise" only bites because
   MSE/Huber forces a single point prediction and penalizes the
   irreducible noise residual. The jetformer loss is a marginal
   likelihood under a learned GMM: the mixture's `log_sigma` *is* the
   model's per-patch uncertainty. High-noise patches ⇒ wide learned
   variance ⇒ low NLL penalty, no capacity wasted chasing the exact
   value. ivar-weighting is the poor man's version of what the density
   head already does natively; layering `1/σ²` on top double-counts the
   noise model and makes the objective stop being a valid log-likelihood,
   so `generation.py`'s sampler no longer matches training.

2. **The loss lives in latent z-space; ivar lives in raw flux-space, and
   the bridge is harder than ADR 0002's own deferred Option C.**
   ADR 0002 explicitly *defers* the fixed arcsinh-Jacobian correction
   (Option C) as second-order plumbing. The jetformer is worse: the
   normalization is a *learned flow updated every step*, not a fixed
   `arcsinh`. Propagating physical ivar into z-space requires the flow's
   Jacobian at each step, and the "stay survey-agnostic, just multiply
   by ivar" trick doesn't survive a learned, moving transform.

3. **The flow must stay invertible — that's the point of jetformer here.**
   Per `AGENTS.md`, jetformer configs skip per-patch standardization so
   the record→token map stays invertible and the exact-likelihood loss
   is valid. ivar weighting (clamps, validity masks, per-band averaging)
   is non-invertible post-hoc weighting; pushing it through the flow
   breaks the generation path (`sample_gmm` → `flow(z, reverse=True)`),
   which is the jetformer's reason to exist over the affine+Huber
   default.

4. **The ADR's enabling condition is unmet for images.** ADR 0002's
   pivotal enabler is switching Legacy Survey from `…_north` to
   `…_dr10_south_21` because the north set has **no per-pixel ivar**
   (only per-band `psfdepth`). AstroPT3's pilot data is pinned to
   `mmu_ssl_legacysurvey_north` (PLAN user decision #4, verified
   schema). Image ivar — the modality the ADR targets — isn't available
   here without reversing a fixed data decision. The only ivar this repo
   carries is `spectrum.ivar` (spectra, via `data/mmu.py`), and spectra
   are not what the ADR is about.

5. **The band-subset-invariance driver doesn't apply.** ADR 0002's
   explicit ask — "same relative loss regardless of which/how many
   bands" — comes from a tokenizer that trains on variable band subsets
   with band dropout. AstroPT3 packs all 3 image bands into one
   192-dim patch token (`(c p1 p2 c)`); there is no per-band-subset
   training, so the per-band reduced-χ² averaging machinery has nothing
   to normalize.

## Where the idea *would* transfer

Spectra in the **affine + Huber** path (not jetformer), using
`spectrum.ivar` that already flows through `data/mmu.py`. The PLAN
already lists this as a config option ("Option (not pilot-default):
ivar-weighted Huber"). If spectra are ever moved off the jetformer head
to the Huber head, lift `spectrum_jeff.loss_fn`'s
`std_err = sqrt(clamp(ivar, 0, clip_ivar))·(1−badmask)` pattern verbatim
— one line at the call site, no ADR-scale vertical slice, because ivar
is already in the record and the spectrum path already carries a mask.

For the jetformer itself: skip it. The density head is the ivar.

## Consequences

- No change to `modeling_astropt3.py`, `generation.py`, the data
  pipeline, or the nanotron fork's loss block.
- The jetformer objective stays a valid exact log-likelihood, so
  sampling/generation and the `starts-1` GMM alignment remain
  consistent with training.
- If a future survey with per-pixel image ivar is adopted (overriding
  PLAN decision #4) *and* the affine+Huber path is used for images,
  revisit — that combination is exactly the `galactiktok` ADR 0002
  setup and the ADR applies directly.

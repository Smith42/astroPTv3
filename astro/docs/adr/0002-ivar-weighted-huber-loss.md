> ## ⛔ Superseded by ADR 0010 (2026-07-20)
>
> This ADR is **Superseded by [ADR 0010 — Remove the affine and aim tokenisers](0010-remove-affine-and-aim-tokenisers.md)**.
> ADR 0010 removes the `affine` tokeniser (and the fork-only `aim`) entirely,
> making `jetformer` the sole tokeniser. ADR 0002's proposal — ivar-weighted
> Huber on the `affine` + Huber path — has **no host** once the affine path is
> gone, and ADR 0001 already rejected ivar-weighting for the jetformer head
> ("the density head is the ivar"). The body below is retained as a historical
> record of the parked proposal; it is not cleared for implementation.

# ADR 0002: Inverse-variance-weighted Huber loss for the affine tokeniser

- **Status:** Superseded by ADR 0010 — see *Superseded-by-0010* banner above.
- **Date:** 2026-07-15 (facts refreshed 2026-07-16 for physical normalization;
  decision untouched — see *Superseded context* below; **superseded 2026-07-20**).
- **References:**
  - [`0001-jetformer-inverse-variance-loss.md`](0001-jetformer-inverse-variance-loss.md)
    (this repo — Rejected for the `tokeniser: jetformer` head; the
    "Where the idea *would* transfer" section is the seed of this ADR)
  - `../galactiktok/docs/adr/0002-inverse-variance-weighted-loss.md`
    (sibling repo — Accepted there for an MSE image tokenizer)

> ## 🅿️ Parked
>
> This ADR is **Proposed but Parked** — it records a design reached through a
> grilling session, not a decision cleared for implementation. The proposer is
> no longer sure the decision should be implemented as drafted.
>
> - The **root decision** (accept ivar-weighted Huber for **both** modalities and
>   commit to the DR10-south pilot switch) is **tentative** and must be
> re-examined before this ADR is moved to Accepted.
> - **Open Issue Q5** (switch justification vs config-gating posture) is
>   unresolved.
>
> **Resume here:** re-open Q5, and *before that* revisit whether the root
> decision still stands. If the root flips to defer/reject, the downstream
> branches (loss formulation, data-switch mechanics, gating) re-open.

> ## ⚠️ Superseded context (2026-07-16)
>
> This ADR was drafted against the **PU asinh stretch**, which
> `a79e4ff` replaced with **physical band-registry normalization**
> (`data/band_registry.py`): a fixed `arcsinh(flux/0.01)·0.01` knee keyed on
> the record's band names, with **no per-corpus calibration** —
> `compute_norm_stats.py` and the p1/p99 `norm_stats` path are gone.
>
> Facts below have been corrected for this; the decision, options, and open
> issues are unchanged. **This makes the proposal cheaper, not different:**
> adding `des-i` for the DR10-south switch is now a `BAND_REGISTRY` entry of
> published survey constants rather than a calibration pass over the corpus.
> The loss design is unaffected — `σ_patch²` propagation still comes from
> `per_patch_standardize`, which is unchanged and still runs after the
> stretch. O1 (root-decision doubt) should note the switch got slightly
> cheaper when it is re-examined.

## Question

Should the `tokeniser: affine` + Huber path (the default / release artifact per
PLAN Phase 3) weight its reconstruction loss by per-pixel inverse variance
(`ivar = 1/σ²`), as the `galactiktok` tokenizer now does for its MSE head — and,
if so, for which modalities, in which loss form, and on which pilot data?

## Context

ADR 0001 rejected ivar-weighting for the `tokeniser: jetformer` head: the GMM
density head already models per-patch uncertainty (so `1/σ²` double-counts the
noise model), the loss lives in latent z-space behind a learned flow whose
Jacobian is not the fixed arcsinh correction the `galactiktok` ADR defers, and
the flow must stay invertible for `generation.py`. But ADR 0001's "Where the idea
*would* transfer" section points squarely at the **affine + Huber** path:

- It is a **point-prediction loss**, so the "unweighted loss spends capacity
  reproducing noise" argument bites exactly as it does for `galactiktok`'s MSE.
- ivar composes with Huber cleanly: `huber(√w·(ŷ−y), delta)` is a robustified
  Gaussian likelihood.
- The one new piece of plumbing vs `galactiktok` is this repo's
  **per-patch standardization** after the physical band-registry stretch
  (`transforms.per_patch_standardize`), which rescales each patch by its own
  `σ_patch`. Raw-flux ivar must be propagated into loss space as
  `ivar_loss = ivar_pixel · σ_patch²` — a **fixed per-patch scalar** computed
  during the forward transform, cheap and exact (unlike the jetformer's learned
  flow Jacobian).

The affine + Huber loss today (`modeling_astropt3.py:244`):

```python
pred   = self.decoders[name](hidden[left_shift_mask(mask)])
target = modality_values[name].to(pred.dtype)
mod_loss = F.huber_loss(pred, target, delta=self.config.huber_delta)
```

weighted by per-modality `loss_weight` and averaged over modalities present.
Predictions/targets are per-patch-standardized patch vectors (image: 192-dim =
3 bands × 8 × 8; spectrum: 31 patch-256 tokens).

Two enabling facts shape the design:

- `spectrum.ivar` (7781-bin) already flows through `data/mmu.py`; the spectrum
  path already carries a mask. Spectra ivar-weighting is ~one call-site change.
- The current pilot survey `UniverseTBD/mmu_ssl_legacysurvey_north` ships **no
  per-pixel image ivar** (only per-band `psfdepth` scalars). Image ivar requires
  switching the pilot to `hugging-science/mmu_legacysurvey_dr10_south_21`
  (`image.flux` / `image.ivar` / `image.mask`, 160×160 px, 4 bands `des-g/r/i/z`,
  124M rows) — **reversing PLAN user decision #4.**

The `galactiktok` ADR's prior art (`spectrum_jeff.loss_fn`:
`std_err = sqrt(clamp(ivar, 0, clip_ivar))·(1−badmask)`,
`F.mse_loss(recon·std_err, target·std_err)`) and its chosen per-channel
reduced-χ² form the reference point for the options below.

## Decision drivers

- Spend capacity on informative pixels, not on reproducing sky noise
  (heteroscedastic astronomical data; surveys ship `ivar = 1/σ²` for this).
- Keep the loss a **Huber regressor** — the smallest change to the existing
  call site, preserving δ's robustness role and the model's regression framing.
- Stay in the per-patch-standardized space the affine path already trains in;
  propagate ivar through that fixed transform exactly (`· σ_patch²`), not
  approximately.
- Be **survey-agnostic** in the loss (see only `ivar`, never per-survey
  branches), mirroring `galactiktok` ADR 0002 and `spectrum_jeff`.
- Be consistent with ADR 0001's deferral of the asinh Jacobian (Option C) as a
  second-order correction that ≈ 1 in the sky-dominated regime.

## Options considered

### Option A — Weighted-Huber, `huber(√w·(ŷ−y), delta)`  (proposed)

Stay on Huber; weight each residual component by `√w` where
`w = clamp(ivar·σ_patch², 0, clip)`:

```
mod_loss = F.huber_loss(sqrt(w) * (pred - target), delta=huber_delta)
```

A robustified Gaussian likelihood: sound in the small-residual (weighted-L2)
regime, down-weighting outliers in the large-residual regime via δ. Smallest
change to the existing call site; δ keeps its robustness role; the loss stays a
Huber regressor. Per-patch-standardization ivar propagation (`·σ_patch²`) is
baked into the weight; the asinh Jacobian (galactiktok Option C) is deferred
second-order — with the fixed 0.01 nMgy knee it is `1/√(1+(flux/0.01)²)`, ≈1
for sky-dominated pixels and biting only on bright ones. Band-subset-invariance machinery is N/A — AstroPT3 packs all
bands into one token with no band dropout, so there is nothing to normalize;
per-component `√w` weighting inside the token carries the noise weighting.

### Option B — Per-channel reduced-χ² (galactiktok ADR 0002's choice)

`Σ(m·valid·w·res²)/Σ(m·valid)` reduced per band/component and averaged.
Dimensionless ~1 units and scale-stable across varying valid-pixel counts
(matters for spectra with per-object masks). **Not chosen:** abandons Huber's
outlier robustness (residuals un-capped), changes loss scale / effective LR, and
the band-subset-invariance machinery it buys is irrelevant here (no band
dropout; all bands packed into one token).

### Option C — Reduced-Huber hybrid

χ²-style valid-count normalization wrapping Huber's piecewise loss:
`Σ(m·valid·w·huber(√w·(ŷ−y),delta)) / Σ(m·valid)`. **Not chosen:** not present
in either prior ADR; needs a defensible statistical reading ("is a reduced
Huber a sensible likelihood?") and adds complexity for no clear gain over
Option A on this model.

### Option D — Keep unweighted Huber (status quo)

**Rejected as the proposed design** — this is the status quo the ADR exists to
upgrade, and it ignores the measurement noise the surveys provide. (Kept as the
**config-gated default** — see Decision.)

## Decision (proposed, tentative — see Parked banner)

**Accept ivar-weighted Huber for both spectra and images on the affine + Huber
path, and commit to the DR10-south pilot switch as part of the decision.**

1. **Loss formulation (Option A):** weight each residual by
   `√w`, `w = clamp(ivar·σ_patch², 0, clip)`, so the call becomes
   `huber(√w·(ŷ−y), delta)`. Per-patch-standardization ivar propagation is
   in-scope (baked into the weight); asinh Jacobian deferred second-order;
   band-subset-invariance machinery N/A.
2. **Modalities:** both `images` and `spectra`. Spectra are cheap
   (`spectrum.ivar` already in the record); images require the data switch.
3. **Data switch:** take native DR10-south geometry — 160×160 (400 patches) +
   `des-g/r/i/z` (256-dim) — and ripple `input_size` / seq-length / packing /
   configs / size YAMLs. Full ripple; gains `des-i` and the full south
   footprint. Not weight-compatible with north checkpoints (already broken by
   the survey/ivar switch regardless). **Reverses PLAN user decision #4.**
4. **Gating:** keep **unweighted Huber as the default**; ivar-weighting is
   opt-in via config (`ivar_weighted: true`), matching PLAN's "Option (not
   pilot-default)." `clip` (ivar clip ceiling) is a config knob, default off
   or mirroring `spectrum_jeff`'s `clip_ivar=100` — TBD.

### Stages (proposed)

1. **Propagate ivar through per-patch standardization.** During the forward
   transform, compute the fixed per-patch scalar `ivar_loss = ivar_pixel · σ_patch²`
   (the squared Jacobian of `(a−μ)/σ`). Cheap and exact, computed anyway.
2. **Weights and validity.** `w = clamp(ivar_loss, 0, clip)`; bad pixels folded
   into `ivar = 0` at load time (as `galactiktok` specifies), so validity is
   derived from ivar alone — loss stays survey-agnostic.
3. **Weight the Huber residual.** `mod_loss = F.huber_loss(sqrt(w)*(pred-target),
   delta=huber_delta)` per modality, then weighted by `loss_weight` and averaged
   over modalities present as today.
4. **Data switch.** Re-pin pilot `mmu_ssl_legacysurvey_north` →
   `mmu_legacysurvey_dr10_south_21`; add a `des-i` `BAND_REGISTRY` entry
   (zeropoint / pixel scale / `m_bright` from published survey constants — no
   calibration pass, and unknown bands raise until it exists);
   ripple 160px / 4-band / 400-patch / 256-dim through `tokenization.py`,
   packing, configs, and size YAMLs.
5. **Mirror to the nanotron fork.** The loss edit must mirror to the fork's loss
   block (AGENTS.md: two implementations, one weight source of truth).

## Consequences (if Accepted as proposed)

### Positive

- **Noise-aware reconstruction** — capacity spent on informative pixels, not
  sky noise; the objective is a proper robustified Gaussian likelihood.
- **Survey-agnostic** — the loss branches on nothing but ivar; after the
  Legacy switch both modalities supply per-pixel ivar + mask.
- **Consistent with `spectrum_jeff` / `galactiktok` ADR 0002** — one
  ivar-weighting convention across the sibling repos.
- **Larger, richer pilot corpus** — 14M → 124M rows, DR10, `des-i`, south
  footprint (beneficial for the larger Phase 6 sizes).
- **Smallest-loss-edit form** — Option A is a one-call-site change; δ and the
  Huber-regressor framing are preserved.

### Negative / tradeoffs

- **Reverses PLAN user decision #4** (pilot pinned to north). The full geometry
  ripple (152→160px, 3→4 bands, 361→400 patches, 192→256-dim) touches
  `tokenization.py`, packing, configs, seq-lengths, and every size YAML.
- **Not weight-compatible** with any north-trained checkpoint. Cheap now
  (pre-release); expensive once a real pilot trains.
- **9× volume** (14M→124M) — longer prep, larger corpus; image×spectra
  crossmatch yield and the image-only/image+spectra ratio change; re-eval.
- **4th band (`des-i`) registry entry** — a new `BAND_REGISTRY` entry is
  required (physical normalization raises on unknown bands), but it is
  published survey constants, not a calibration pass over the corpus.
- **Loss scale / effective LR** — weighting by `√(ivar·σ_patch²)` changes loss
  magnitude vs unweighted Huber; may need `loss_weight` / LR retune; δ
  semantics shift (δ is now in weighted-residual units).
- **No asinh Jacobian** (Option C) — exact for sky-dominated pixels,
  approximate for bright ones, same deferral as both prior ADRs.
- **Synthetic ivar = ones** — smoke tests exercise plumbing, not real weighting;
  no synthetic heteroscedastic fixture yet.
- **Config-gating coherence tension (see Open Issue Q5)** — the DR10-south
  switch + full ripple is committed-as-default while its primary stated
  justification (per-pixel image ivar) is consumed only by an opt-in flag.

## Open issues

### O1 — Root-decision doubt (user-raised, blocker for Accepted status)

The proposer is no longer sure the decision should be implemented as drafted.
The root decision (accept ivar-weighted Huber for both modalities + commit to
the DR10-south switch) is **tentative**. If it flips to defer/reject, the
downstream branches (loss formulation, data-switch mechanics, gating) re-open.

### O2 — Switch justification vs config-gating posture (Q5, parked)

The DR10-south switch + full ripple is committed-as-default, but ivar-weighting
(the loss that consumes the switch's headline justification, per-pixel image
ivar) is opt-in. Two ways to resolve:

- **(a) Independent justification:** frame the switch as justified by 9× volume,
  DR10, `des-i`, and south footprint, with per-pixel ivar as a bonus consumed by
  the opt-in loss. Config-gating is then coherent.
- **(b) Ivar-primary:** treat ivar as the primary justification; the config flag
  gates only the loss, while the data switch (and ivar flowing into the record)
  is committed regardless. The ADR states this explicitly.
- **(c) Revisit gating:** flip the default so ivar-weighted Huber is the pilot
  default, making the committed switch and the default loss consistent
  (revisits the gating decision).

**Not answered.** This is the resume entry point.

### O3 — ADR scope vs implementation plan

Does this ADR specify the nanotron-fork mirror and the synthetic-ivar fixture,
or leave those to an implementation plan? Does it record the loss-scale/LR
retune as a consequence, or leave it to first-run tuning? (Proposed: record as
consequences/consequences-flags here; defer detailed steps to an implementation
plan written if/when the root decision is confirmed.)

### O4 — `clip` default

Off by default (matching `galactiktok` ADR 0002), or mirror `spectrum_jeff`'s
`clip_ivar=100`? TBD.

## Resume here

1. Revisit **O1**: does the root decision still stand? If not, re-open the
   downstream branches and likely move this ADR to Rejected or split it
   (spectra-only is the cheap subset).
2. Re-open **O2** (Q5): pick (a) independent justification, (b) ivar-primary, or
   (c) revisit gating.
3. Resolve **O3** (scope) and **O4** (`clip` default).
4. Only then move Status to Accepted and write the implementation plan.

## References

- [ADR 0001 — jetformer inverse-variance loss](0001-jetformer-inverse-variance-loss.md)
  — Rejected for the jetformer head; seeds this ADR's "where it would transfer."
- `../galactiktok/docs/adr/0002-inverse-variance-weighted-loss.md` — Accepted
  there for an MSE image tokenizer; prior art for the per-channel reduced-χ² and
  the Legacy north→south switch.
- `spectrum_jeff.loss_fn` (`src/galactiktok/models/spectrum_jeff/modeling_spectrum_jeff.py`)
  — `std_err = sqrt(clamp(ivar, 0, clip_ivar))·(1−badmask)` prior art.
- `astro/src/astropt3/modeling_astropt3.py:244` — the affine + Huber call site.
- `astro/src/astropt3/data/transforms.py` — `per_patch_standardize` (the
  `σ_patch²` Jacobian source).
- `astro/src/astropt3/data/band_registry.py` — physical normalization
  (`a79e4ff`); the `des-i` entry the DR10-south switch needs lands here.
- `astro/src/astropt3/data/mmu.py` — `spectrum.ivar` already flows; image ivar
  arrives with the DR10-south switch.
- `astro/PLAN.md` user decision #4 — pilot pinned to north (reversed by this
  ADR as proposed).
- `astro/docs/jetformer_noise_diagnosis.md` — the capacity-waste motivation.

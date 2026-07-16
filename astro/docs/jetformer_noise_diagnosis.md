# Jetformer noise / sampling diagnosis

Diagnosis of why `astropt3-70m-jetformer-lowlr` (wandb run `y3oak0l0`) generates
poorly — especially why the per-pixel noise is badly reproduced. Measured
2026-07-14 on the **step-20000** checkpoint (`latest.txt`), on a DeltaAI GH200,
using the HF-converted checkpoint + real val shards (`shakeout_mix2/val`).

**One-line summary:** there are two independent problems — (1) the optimisation
has not converged and step 20000 is *past the best point*, and (2) the
jetformer/GIVT sampling recipe cannot produce calibrated astronomical noise at
any temperature. The visibly bad noise is mostly (2), amplified by (1).

The transformer body is *not* the problem: teacher-forced reconstructions (the
GMM **mode**) look reasonable. It is the **sampling / variance path** (flow + GMM
+ scalar temperature + objective) that is broken.

---

## Setup / architecture facts (this checkpoint)

- Body: SmolLM3, 23 layers, hidden 512, 8 heads (70M). `tokeniser: jetformer`.
- Image modality: (3,152,152) flux → asinh stretch → patch-8 → **361 patches ×
  192 floats** (3·8·8). **Per-patch standardisation is skipped** (deliberate, so
  the record→token map is invertible for the exact-likelihood loss).
- Per-modality flow: `TinyFlow1D` = **4 RealNVP affine-coupling steps**, hidden
  128, scale bounded `tanh(s)*1.5`. Splits the 192-dim patch 96/96.
- Head: `GMMHead`, **K=4** diagonal Gaussian components, `log_sigma` clamped to
  **[−7, 2]** (σ ∈ [9e-4, 7.4]).
- Loss (per modality): `mean(NLL_GMM(z) − logdet)` — exact patch-space
  likelihood, can go negative.
- Sampling (`astropt3.generation`): draw component k, `z = μ_k + T·σ_k·ε`, invert
  flow. `argmax` uses the mixture mean. Scalar `temperature` scales σ only.
- Config: peak LR 1e-4 → cosine → min 1e-5 (at step 20k), warmup 200,
  `clip_grad=1.0`. Noise curriculum `jetformer_noise_max=0.1 → _min=0.0`
  (perturbs only the *embedded* z copy in training, not the GMM target/logdet).
- Training data is effectively **image-only** (`spectra_loss=0` throughout; the
  val shards contain no spectrum-bearing records) — so the spectra head is
  untrained; ignore spectra outputs.

---

## Problem 1 — training not converged / unstable (the "not doing great" root)

wandb history for `y3oak0l0`:

| step  | images_loss | grad_norm | lr     |
|-------|-------------|-----------|--------|
| 8     | 719.5       | 5135      | 4e-6   |
| 2000  | 51.4        | 377       | 9.8e-5 |
| 4000  | 30.9        | 1067      | 9.2e-5 |
| 8000  | 24.7        | 1615      | 7.0e-5 |
| 12000 | 0.13        | 1623      | 4.2e-5 |
| 16000 | **−0.76**   | 1743      | 1.9e-5 |
| 19000 | −0.20       | 1266      | 1.1e-5 |
| 20000 | **+1.02**   | 1186      | 1.0e-5 |

- `grad_norm` sits at **1000–1700 against `clip_grad=1.0`** for the entire second
  half → ~99.9% of every gradient is clipped → effective steps are tiny.
- The loss is **non-monotonic**: it bottoms at −0.76 (step 16k) then **climbs
  back to +1.0** by step 20k. **The checkpoint we sample (20k) is worse than
  16k.**
- Negative loss is a tell: the objective is `NLL_GMM(z) − logdet`, and the flow
  can push loss below zero by **inflating logdet** without improving samples.
  That logdet-gaming is a plausible source of the gradient blow-up.
- This is the OLMo-style grad-norm growth already flagged in `JETFORMER_PLAN.md`.
  It **persists at this 10×-lower LR**, so it is not just an LR-knob issue.

## Problem 2 — noise generation is uncalibrated at any temperature (measured)

All numbers in the **model output space** (asinh-stretched patch space, i.e. what
the model actually generates — no inverse-sinh, to avoid its nonlinearity). Noise
isolated by subtracting a 3×3 box blur (high-pass). Real = 16 val images;
sampled = 8 unconditional draws.

**Per-channel noise std** (real ≈ [0.30, 0.22, 0.43]):

| sampling | ch0  | ch1  | ch2  | vs real            |
|----------|------|------|------|--------------------|
| T=0.2    | 0.07 | 0.07 | 0.03 | ~5× too **small** → blur |
| T=1.0    | 5.9  | 5.8  | 1.9  | ~15–20× too **large** → speckle |

→ **No temperature reproduces the real amplitude.** A single scalar σ-scale
cannot separate structural variance from noise variance (GIVT's structural
limitation), now quantified: low-T kills the noise, high-T explodes it, nothing
in between is right.

**Spatial texture** (lag-1 horizontal autocorr of the noise):

| sampling | ch0    | ch1    | ch2    |
|----------|--------|--------|--------|
| real     | +0.29  | +0.31  | −0.03  |
| T=0.2    | +0.10  | +0.09  | −0.08  |
| T=1.0    | −0.18  | −0.17  | −0.13  |

→ real noise is spatially **positively** correlated (PSF-blurred, not white); the
T=1 samples are **anti**-correlated (white checkerboard). Even at the correct
amplitude it would not look like sky noise.

**Ruled out by measurement:**

- **Not a head defect.** Teacher-forced `log_sigma` on real patches is *not*
  saturated: percentiles p5/50/95/99 = [−3.46, −0.93, 0.69, 0.99] (median σ≈0.4
  in z), **0% at the +2 ceiling**, 1% above σ=2.7. Mixture is *not* collapsed:
  weight entropy 1.29 / max 1.386 (all 4 components used).
- **Not cross-channel decorrelation.** Avg off-diagonal channel corr of the noise
  is ~0 for **both** real (+0.003) and sampled (+0.015…+0.035). The "primary-
  colour dots" look is just per-pixel independent noise at the wrong (huge)
  amplitude — real noise is decorrelated across bands too, just small.

**Mechanism:** the shallow 4-step affine-coupling flow reaches low *z*-space NLL,
but low z-space likelihood does **not** induce a calibrated *pixel*-space
sampling distribution — its inverse amplifies noise ~15–20× and whitens/anti-
correlates it spatially. Exact likelihood in z is being achieved partly through
the logdet term (see Problem 1), which does not constrain sample quality.

## What we're missing (the deeper mismatch)

**We are asking the generative head to *learn* the sensor noise — wasteful and
the wrong tool for astro data.**

- Astro cutouts are **background/noise-dominated**. Exact per-patch likelihood is
  therefore dominated by the NLL of irreducible noise that carries no
  information. That (a) burns most of the model's capacity/loss budget on noise,
  (b) makes the loss heavy-tailed and unstable — **feeding Problem 1**, and (c)
  buries the galaxy signal.
- Astro data has a **known noise model** (per-pixel variance / ivar — MMU spectra
  literally carry `ivar`; images have sky + shot noise). The standard correct
  move is to model the *signal* and treat noise as a known additive process, or
  to **whiten / down-weight the loss by ivar**. We currently do neither.
- **Skipping per-patch standardisation** (the jetformer invertibility choice)
  forces the flow to also model absolute per-patch flux scale on top of noise →
  worse-conditioned target → more instability. There is a real tension here
  between "invertible token map" and "well-conditioned loss".

## Suggested priorities

1. **Fix convergence first — the model is not trained.** Instrument the loss
   *per term* (GMM-NLL vs logdet separately); the logdet is the likely culprit.
   Try penalising/regularising `|logdet|`, warming up the flow before the GMM, or
   a separate/looser grad clip for the flow. Lowering LR further will not fix it
   (already tried — this *is* the low-LR run). Meanwhile, eval/sample from step
   ~16k, not 20k.
2. **Stop memorising noise.** Down-weight / whiten the patch loss by the known
   noise (ivar), or train the head on a denoised / heteroscedastic target and add
   calibrated noise at sample time. This attacks Problem 1, the amplitude
   problem, and the capacity waste at once. *(The user is bringing in a
   normalisation approach — `PolymathicAI/galactiktok@feat/norm` — for the image
   modality; that is the natural place to fold this in.)*
3. **Don't expect temperature to give realistic noise.** If samples need it, add
   it from the noise model post-hoc rather than via the GMM σ.
4. If keeping jetformer as-is: deeper flow (more steps) with **channel-mixing**
   so the sampled noise texture can be non-white, and reconsider the
   no-standardisation choice.

---

## Reproduction

All on a GH200 with the plain `astro` uv venv (`--device cuda`; torch
2.12.1+cu130 sees the GPU — no module load needed). Convert the nanotron
checkpoint to HF first (`/work/nvme/bfvh/msmith10/astroPTv3_tools/convert_cpu.py
--checkpoint <step_dir> --save_path <hf_dir>`).

- Temperature sweep / image-completion figures + the noise-statistics and
  `log_sigma`-saturation measurements were one-off scratchpad scripts (node-local
  `/tmp`, now gone). Key knobs: `astropt3.generation.generate(model, template,
  {"images"}, n, temperature)`; image completion = seed `generate`'s
  `values["images"]` with the first K real patches (K = rows·19 of the 19×19
  patch-8 grid) and only sample at image-patch index ≥ K; hand-rolled loops must
  be `@torch.no_grad()`. Figures were logged to wandb `y3oak0l0` under
  `generation/*` (temp_sweep, image_completion_top{5,10}rows,
  argmax_conditioning_sweep).

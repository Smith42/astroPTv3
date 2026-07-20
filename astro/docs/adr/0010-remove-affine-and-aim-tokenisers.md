# ADR 0010: Remove the affine and aim tokenisers; jetformer becomes the sole tokeniser

- **Status:** Accepted
- **Date:** 2026-07-20
- **References:**
  - [`0001-jetformer-inverse-variance-loss.md`](0001-jetformer-inverse-variance-loss.md) — Rejected ivar-weighting for the jetformer head; established that the GMM density head "is the ivar."
  - [`0002-ivar-weighted-huber-loss.md`](0002-ivar-weighted-huber-loss.md) — **Superseded by this ADR.** Was a Parked proposal to add ivar-weighting to the affine + Huber path; moot once the affine path is removed.
  - [`0008-scalar-modalities.md`](0008-scalar-modalities.md) — Scalars use `GMMHead` under **both** tokenisers; orthogonal to this ADR and explicitly out of scope here.

## Question

Should AstroPTv3 carry multiple tokenisers — the `affine` default plus `jetformer` plus the fork-only `aim` — or collapse to a single tokeniser by removing `affine` (HF + nanotron fork) and `aim` (fork-only), leaving `jetformer` as the sole tokeniser everywhere?

## Context

AstroPTv3 today ships three tokenisers, routed by a `tokeniser` config field (`AstroPT3Config.tokeniser: str = "affine"`, validated to `"affine" | "jetformer"` on the HF side and `"affine" | "aim" | "jetformer"` in the nanotron fork):

- **Affine (the default)** — linear `Encoder`/`Decoder` (`src/astropt3/modalities.py`) + Huber loss + per-patch standardization (`data/transforms.py:per_patch_standardize`, gated at `data/packing.py:90`). 23 of 30 configs use it; it is the default enum value and the release artifact per PLAN Phase 3.
- **Jetformer (the product direction)** — per-modality `TinyFlow1D` flow + flow-`GMMHead` + a noise curriculum (`train_smoke.py:81`) + an exact patch-space likelihood `mean(NLL_GMM(z) − logdet)` that can go negative + inversion (jetformer configs **skip** per-patch standardization to keep the record→token map invertible) + `astropt3.generation.generate()` / `scripts/generate.py`. Seven configs today: `test-tiny-jetformer` plus six 70m nanotron pilots. Empirically outperforms affine, which is the motivation for this ADR.
- **`aim` (fork-only dead code)** — a 2-layer tanh/GELU MLP `Encoder`/`Decoder` variant of affine that exists **only** in the nanotron fork (`src/nanotron/models/astropt3.py`, 22 references; `src/nanotron/config/astropt3_config.py`). It is referenced by **zero** configs (`tokeniser: aim` matches nothing) and was already dropped from the HF side in commit `cb1b3ca` ("kept only for back-compat"). Its code surface is parallel to affine's; removing affine covers the same lines.

The `tokeniser` field is read in nine HF files (`configuration`, `modalities`, `modeling`, `data/packing`, `data/nanotron_loader`, `generation`, `eval/samples`, `train_smoke`) and 22 fork references — it is a genuine router, not a label.

Two facts bound the risk of the removal:

- **Checkpoints are external to git, not tracked.** `.gitignore` excludes `wandb/`; `git ls-files` has zero `.safetensors`/`.bin`/`.pt`. Pretrained checkpoints live on `/beegfs/.../astroPTv3_checkpoints/` and (per PLAN.md) are converted + uploaded to HF Hub as scheduled checkpoints land. The `/beegfs` dir currently holds only **pilot runs** — six `astropt3-70m-jetformer*` variants (~29G each, 32 steps) plus `astropt3-160m-real-shakeout`. No production fleet exists in either tokeniser; what is on disk is overwhelmingly jetformer pilots. → A repo change is **code + configs + tests only**; external `/beegfs`/Hub assets are out of scope for this ADR's diff.
- **Scalars are orthogonal.** `Z`, `ebv`, and `photometry` modalities use `GMMHead` under **both** tokenisers (ADR 0008); the scalar loss route (`modeling_astropt3.py` scalar-branch) bypasses the tokeniser entirely, and `data/scalar_registry.py` is tokeniser-independent. Removing affine/aim does not touch scalar loss.

Tests currently pin affine: `tests/test_saveload.py:18` asserts `config.tokeniser == "affine"`; `tests/test_jetformer.py` compares affine-vs-jetformer standardization; `tests/test_generate.py:100-105` tests that affine *rejects* `generate()`. Rewriting these to a jetformer-only world is part of the change.

## Decision

**Remove the `affine` and `aim` tokenisers. Jetformer becomes the sole tokeniser in both the HF transformers implementation and the nanotron fork.** Concretely:

1. **Delete the affine path** — the linear `Encoder`/`Decoder`, the Huber loss call site, the per-patch-standardization gate, and all `if tokeniser == "affine"` / `tokeniser != "jetformer"` guards — from `src/astropt3/` and the fork's `src/nanotron/models/astropt3.py`.
2. **Delete the `aim` path** — the `elif tokeniser == "aim"` MLP branches and the `"aim"` enum value — from the nanotron fork. (Same code surface as affine; nothing extra.)
3. **Make jetformer unconditional** — `TinyFlow1D` + flow-`GMMHead` + noise curriculum + exact-likelihood loss + `generate()` become the only path. The jetformer-only guards in `generation.py:58` and `eval/samples.py:200` collapse (no rejection branch, sampling always available). Per-patch standardization is **gone** (jetformer needs an invertible record→token map).
4. **Delete the `tokeniser` config field entirely** from `AstroPT3Config` (HF) and the fork's `astropt3_config.py`, including the enum validation asserts. All `if config.tokeniser == "jetformer"` conditionals become unconditional. This is the real simplification: the router vanishes, not just collapses.
5. **Remove the `nanotron_loader.py` back-fills in the same change.** `getattr(config, "tokeniser", "affine")` at `data/nanotron_loader.py:65,72,374` defaults to the removed affine path — if the field is deleted but these back-fills are left, flat micro-batch dict loading silently routes to the removed affine path and explodes. Deleting the field without removing the back-fills is a landmine; they go together.
6. **Rewrite the 23 affine YAMLs to jetformer in place**, authoring the **seven missing large-size jetformer configs** (160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b — none exist today) as best-guess defaults scaled from the 70m jetformer recipe. (Beyond 70m + `test-tiny-jetformer`, jetformer configs do not exist; the rewrite is per-size jetformer hyperparameter authoring, not a tokeniser flip.)
7. **Rewrite the affine-pinning tests** to a jetformer-only world: `test_saveload.py`, the affine-vs-jetformer comparison in `test_jetformer.py`, and the affine-rejects-`generate()` assertions in `test_generate.py`.
8. **Scalars are out of scope** — their `GMMHead` path is tokeniser-independent and unchanged.
9. **Single unified change** — HF code/configs/tests + nanotron fork mirror in one atomic PR. No half-purged intermediate state where affine is gone from HF but lingers in the pretraining fork.
10. **Mark ADR 0002 Superseded by 0010.** Its proposal (ivar-weighted Huber on the affine + Huber path) has no host once affine is removed; ADR 0001 already rejected ivar-weighting for the jetformer head ("the density head is the ivar").

## Rationale

1. **Jetformer outperforms affine; affine is now defunct code with real maintenance cost.** Two tokenisers means two Encoder/Decoder surfaces, two loss branches, two standardization stances, two generation contracts, and a `tokeniser` router threaded through nine HF files and 22 fork references — all maintained for a path that no longer ships. Removing the defunct path is the simplification: a single tokeniser, no dead `Encoder`/`Decoder`, no driver conditionals.
2. **Affine's planned improvement is already subsumed.** ADR 0002's ivar-weighted Huber was the headline upgrade for the affine path; ADR 0001 established that the jetformer GMM density head already models per-patch uncertainty natively (rationale: "the density head is the ivar"). With jetformer as the sole path, the upgrade affine was meant to receive is moot — jetformer has the property by construction.
3. **`aim` is pure dead code, safe to delete with affine.** It is unreachable from the HF side (dropped in `cb1b3ca`), referenced by zero configs, and architecturally parallel to affine (same Encoder/Decoder conditionals). Removing affine and leaving `aim` would not achieve "single tokeniser"; removing both does.
4. **Deleting the field is the root-cause fix, not a single-valued default.** Keeping `tokeniser` as a single-valued `"jetformer"` default leaves the router as dead infrastructure and preserves the `nanotron_loader` `getattr` back-fills as latent misfire points (they default to the removed affine). Deleting the field removes the seam entirely and forces the back-fill removal the same change needs anyway.
5. **The removal destroys no trained model fleet.** Checkpoints are external to git, and what exists on `/beegfs` is jetformer pilots plus one ambiguous shakeout run — no production affine checkpoint set is confirmed. The repo change is code + configs + tests; external `/beegfs`/Hub assets (and any decision to retire them) are explicitly out of scope here.
6. **Scalars stay untouched.** Because scalar loss bypasses the tokeniser (ADR 0008), this ADR changes nothing about `scalar_registry.py` or the scalar `GMMHead` route — removing affine/aim cannot regress redshift/ebv/photometry prediction.

## Consequences

1. **`nanotron_loader.py` back-fill removal is mandatory in the same change.** The `getattr(config, "tokeniser", "affine")` calls at lines 65, 72, 374 default to the removed affine path; deleting the `tokeniser` field without removing them makes flat micro-batch dict loading silently route to a deleted path. This is the single most important implementation hazard and is not derivable from any prior ADR (ADR 0004 documents the *inverse* precedent — adding a field with a backfill default — but nothing covers field *deletion*).
2. **Stale `tokeniser` field in existing jetformer checkpoint `config.json`.** Once the field is deleted from `AstroPT3Config`, existing jetformer checkpoints' `config.json` files carry a now-unknown field. `AutoConfig.from_pretrained` may drop the unknown field silently (transformers' default behavior for unknown keys) rather than reject; implementers should verify the load behavior on a real jetformer checkpoint and document it. The `nanotron_loader` back-fills are removed (not left to default) so the loader no longer paper-overs this.
3. **Large-size jetformer configs (160m→12b) are best-guess defaults pending HP-tuning.** This ADR authors them by scaling the 70m jetformer recipe (noise-curriculum schedule, flow dims, exact-likelihood deltas); they are **not validated by training** and should be treated as starting points. Actual jetformer training/HP-tuning at each size is a separate later effort; the ADR only guarantees the repo *ships a loadable jetformer config for every size*.
4. **ADR 0002 is Superseded by this ADR.** Its Parked proposal (ivar-weighted Huber on the affine path) loses its host; its Status is flipped to *Superseded by ADR 0010*. (ADR 0001 stays Rejected — it was already decided against for the jetformer head.)
5. **Per-patch standardization is gone.** The `packing.py:90` gate and the affine-only branch of `per_patch_standardize` are deleted; jetformer's invertibility requirement already mandated skipping standardization, so this loses no jetformer behavior. The noise curriculum (`train_smoke`) and `generate()` guard become unconditional.
6. **Existing affine/aim-tagged checkpoints become unloadable** from this codebase (their `Encoder`/`Decoder` weights have no classes). As noted, no production affine/aim fleet is confirmed on disk; retiring any that exist is an out-of-band `/beegfs` + HF Hub operation, not part of this ADR's diff.

## Validation / Acceptance

The unified purge is complete when **all** of the following pass (these are the phase verification gates plus the purge-specific checks):

- **CPU gates pass post-remove:**
  - `uv run pytest` — the rewrite of the affine-pinning tests (`test_saveload`, `test_jetformer`, `test_generate`) to a jetformer-only world passes, and the `@pytest.mark.gpu` suite is unaffected.
  - `uv run python scripts/count_params.py` — size table asserts ±10% nominal for every size; the affine→Encoder/Decoder removal and jetformer-only flow + GMMHead widths must keep counts in range.
  - `uv run python -m astropt3.train_smoke --config configs/model/<jetformer-config> --steps 50 --assert-decrease` — loss decreases (now unconditional jetformer; noise curriculum unconditional).
- **Every config loads:** each of the rewritten (70m + `test-tiny`) and newly authored (160m, 410m, 1b, 1.4b, 2.8b, 6.9b, 12b) jetformer configs loads via `config_io.load_model_config` and `AutoConfig` without the `tokeniser` field.
- **Back-fill removal verified:** `grep -rn 'tokeniser' astro/src/astropt3/data/nanotron_loader.py` returns nothing; a flat micro-batch dict load lands on the jetformer path (no silent affine default).
- **GPU smoke in jetformer (run on the training machine, not here):** the `@pytest.mark.gpu` suite — `test_jetformer_gpu.py` TP=2 replicated-grad check, 50-step smoke, nanotron→HF conversion — passes with the fork's affine/aim branches removed.
- **HF↔nanotron parity:** `convert_nantron_to_hf` / `hf_to_nanotron` round-trip on a jetformer checkpoint preserves weights and the (now-fieldless) config, on both sides of the unified change.
# ADR 0010: Remove the affine and aim tokenisers; jetformer becomes the sole tokeniser

- **Status:** Accepted
- **Date:** 2026-07-20
- **References:**
  - [`0001-jetformer-inverse-variance-loss.md`](0001-jetformer-inverse-variance-loss.md) — Rejected ivar-weighting for the jetformer head; established that the GMM density head "is the ivar."
  - [`0002-ivar-weighted-huber-loss.md`](0002-ivar-weighted-huber-loss.md) — **Superseded by this ADR.** Its proposal (ivar-weighted Huber on the affine path) has no host once affine is gone.
  - [`0008-scalar-modalities.md`](0008-scalar-modalities.md) — Scalars use `GMMHead` under both tokenisers; unaffected by this ADR.

## Question

Keep three tokenisers — `affine` (the default), `jetformer`, and the fork-only
`aim` — or collapse to `jetformer` alone?

## Context

The `tokeniser` field routes three paths: `affine` (linear `Decoder` + Huber +
per-patch standardization), `jetformer` (`TinyFlow1D` + `GMMHead` + exact
patch-space likelihood + noise curriculum + `generate()`), and `aim` (a fork-only
MLP variant of affine, referenced by zero configs, already dropped from the HF
side in `cb1b3ca`).

Jetformer outperforms affine — **TODO: cite the pilot run and metric.** That
result is the whole reason for this ADR and is the one claim here not verifiable
from the tree; name the run so it survives the next eighteen months.

Bounds on the risk:

- **`aim` is 4 lines** (`astropt3_config.py:88,131`; `models/astropt3.py:90,111`)
  and unreachable from HF.
- **No affine fleet exists.** `/beegfs/.../astroPTv3_checkpoints/` holds six
  `astropt3-70m-jetformer*` pilots plus `astropt3-160m-real-shakeout`; git tracks
  no weights. This is a code + configs + docs change.
- **Scalars bypass the tokeniser** (ADR 0008) and are untouched.

## Decision

Delete `affine` and `aim`; delete the `tokeniser` field itself, so the router
vanishes rather than collapsing to a single-valued default.

1. **Delete the affine path** — the `Decoder` class, the Huber call site, the
   standardization gate, and every `tokeniser` guard — from `src/astropt3/` and
   the fork's `models/astropt3.py`. **`Encoder` stays**: it is a single
   `nn.Linear` used under jetformer for all modalities including scalars
   (`modeling_astropt3.py:93`). Only its `tokeniser` argument and validator go.
2. **Delete `aim`** — the two `elif tokeniser == "aim"` branches and the enum
   value, fork-only.
3. **Delete `data/transforms.py`** (20 lines, sole caller is the gate at
   `packing.py:117,134`). Per-patch standardization goes with it; jetformer's
   invertibility already required skipping it.
4. **Delete the `tokeniser` field** from `AstroPT3Config` and the fork's
   `astropt3_config.py`, including the enum asserts — *and in the same commit*
   the `getattr(config, "tokeniser", "affine")` back-fills at
   `nanotron_loader.py:65,72,374` and `packing.py:90`. `PretrainedConfig` absorbs
   unknown kwargs silently, so a leftover back-fill would route flat micro-batch
   loading to a deleted path with no error. This is the one hazard in the change.
5. **Drop the `tokeniser:` line from the 22 affine YAMLs** (9 `configs/model/`,
   13 `configs/nanotron/`). Nothing else: all ten existing jetformer configs use
   the `AstroPT3Config` defaults verbatim (`flow_steps 4, flow_hidden 128,
   gmm_k 4, noise_max 0.1, noise_min 0.0`), so there is no per-size recipe to
   author. Merge `test-tiny{,-jetformer}.yaml` and the nanotron pair into one
   file each.
6. **Delete the affine-only tests, don't rewrite them** — the affine tail of
   `test_jetformer.py:171-178`, the affine half of
   `test_generate.py:test_generate_rejects_affine_and_missing_span`, and
   `test_saveload.py:18`. In `test_modality_order`, `test_physical_norm`,
   `test_spectral_norm`, and `test_scalar_modalities`, the
   `AstroPT3Config(**{**tiny_config.to_dict(), "tokeniser": "jetformer"})`
   wrapper collapses to `tiny_config`.
7. **Drop `--tokeniser` from `scripts/tp2_grad_check.py:81`** (it defaults to
   affine) and the `--tokeniser=jetformer` argument at
   `tests/test_jetformer_gpu.py:126`.
8. **Update the prose**: `AGENTS.md` (the router and standardization gate),
   `PLAN.md:118,160,183,188,263`, `README.md:8`, and flip ADR 0002 to
   *Superseded by ADR 0010*.
9. **Fork first, then bump the submodule pointer** in the HF PR — a commit cannot
   span a repo and its pinned submodule (currently `51d74e5e`). Ordering, not
   atomicity, is what prevents a half-purged state.

## Rationale

1. **Affine is defunct code with a live maintenance cost** — a `tokeniser` router
   threaded through eight HF files, `tp2_grad_check`, six test files, and 37 fork
   references, all for a path that no longer ships.
2. **Affine's planned upgrade is already subsumed.** ADR 0002's ivar-weighted
   Huber was affine's headline improvement; ADR 0001 established that jetformer's
   GMM density head has that property by construction.
3. **Deleting the field beats defaulting it.** A single-valued `tokeniser` leaves
   dead infrastructure and keeps the back-fills alive as silent misfire points.
4. **The cost of being wrong is a revert.** No trained fleet depends on affine,
   so the fallback if jetformer misbehaves at 1b+ is `git revert`, not a retrain.

## Consequences

1. **Affine/aim-tagged checkpoints become unloadable** (their `Decoder` weights
   have no class). None are known to exist; retiring any is an out-of-band
   `/beegfs`/Hub operation, out of scope here.
2. **Existing jetformer checkpoints are unaffected.** A stale `tokeniser:
   jetformer` key in their `config.json` survives `AutoConfig.from_pretrained` as
   an inert attribute — verified, neither dropped nor rejected.
3. **Modality extras grow ~3.9×** (70m: 681k → 2.64M params) as every size moves
   to flow + `GMMHead`. Measured across all sizes: worst deviation +2.9% at 70m,
   well inside `count_params`' ±10%.

## Validation

The three standing gates, unchanged — `uv run pytest`, `uv run python
scripts/count_params.py` (which already loads every `configs/model/*.yaml`
through `load_model_config`), and `train_smoke --assert-decrease` — plus the
`@pytest.mark.gpu` suite on the training machine. No purge-specific checks: if
the back-fills or a config survive the change, these fail.

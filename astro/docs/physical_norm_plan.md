# Implementation Plan: Port galactiktok `feat/norm` Physical Image Normalization

Replace astroPTv3's data-driven Platonic Universe (PU) asinh stretch with
galactiktok's physical, band-registry-keyed normalization for the image
modality. Spectra are unchanged. Each chunk is independently verifiable and
gated on the CPU test suite (`uv run pytest`); the smoke gate
(`train_smoke --assert-decrease`) runs after the data-side chunks are complete.

**Source of truth:** `../galactiktok` branch `feat/norm`,
`src/galactiktok/models/image_transformer/band_registry.py` + the
`_normalize`/`_physical_factors`/`decode` methods in
`modeling_image_transformer.py`.

**Target:** `astro/src/astropt3/` (this transformers implementation + CPU test
target). The nanotron fork mirrors the loader change only (thin fork).

---

## Chunks (in dependency order)

### Chunk 1 — Port `band_registry.py` (new file, no callers yet)

**Files:**
- NEW `astro/src/astropt3/data/band_registry.py` — verbatim port of galactiktok's
  `band_registry.py`: `LEGACYSURVEY_REFERENCE_ZP = 22.5`,
  `LEGACYSURVEY_REFERENCE_SCALE = 0.262`, the full 15-band `BAND_REGISTRY`
  (`des-g/r/z`, `hsc-g/r/i/z/y`, `jwst-f090w..f444w`), `RAW_BANDS`
  (`rgb-r/g/b`), `rescale_factors(band)`, `clamp_flux(band)`, the module
  constant `_DIV_FACTOR = 0.01` (with the ADR 0001 / band_registry citation
  comment), and the `__main__` self-check (all bands finite + des identity
  rescale).

**Also add** the `_normalize` / `_physical_factors` functions here (ported from
galactiktok's `ImageTransformerTokenizer` methods), as module-level functions
taking `(flux, bands, device, dtype)` and returning tensors. `_normalize`:
`flux * rescale → clamp(-cf, +cf) → arcsinh(x/0.01)·0.01`; `RAW_BANDS`
passthrough; unknown bands raise `NotImplementedError`. Add an
`empty-bands guard` (return flux unchanged with a comment, or assert non-empty
— see Tests). Add a `physical_inverse(flux, bands)` helper: clamp to arcsinh
ceiling → `sinh(x/0.01)·0.01` → `/rescale`.

**Verify:** `uv run python -m astropt3.data.band_registry` runs the `__main__`
self-check. No other code touched yet, so `uv run pytest` is unaffected.

---

### Chunk 2 — Wire physical norm into `transforms.py` + `packing.py`

**Files:**
- `astro/src/astropt3/data/transforms.py` — remove `ASINH_ALPHA`,
  `asinh_params_from_percentiles`, `asinh_stretch`, `_broadcast_band`. Keep
  `per_patch_standardize` (still used by the affine path + spectra). Repoint
  the module docstring at the physical-norm story (or delete the PU docstring;
  `band_registry.py` now owns the normalization description).
- `astro/src/astropt3/data/packing.py` — `ObjectSequencer.__init__`:
  - Remove `image_p1`, `image_p99`, `alpha` params and the `asinh_offset`/
    `asinh_scale` computation.
  - Keep `spiral` and the `standardize = tokeniser != "jetformer"` flag.
- `ObjectSequencer._images_tokens`: replace
  `asinh_stretch(flux, self.asinh_scale, self.asinh_offset)` with
  `physical_normalize(flux, record["image"]["band"])` (import from
  `band_registry`). Then `patchify_image` → (standardize if affine) →
  (spiralise if spiral). Records already carry `image.band` from
  `mmu.normalize_record` / `synthetic.IMAGE_BANDS`.

**Verify:** `uv run pytest` — expect failures in `test_transforms.py` (whole
file, retired functions) and `test_jetformer::test_jetformer_skips_per_patch_standardization`
(expected-flux assertion). These are fixed in Chunk 7. Other tests
(`test_packing`, `test_model`, `test_generate`, `test_train_smoke`) should pass
once fixtures are rescaled (Chunk 3) — but run them now to see the real
breakage surface.

---

### Chunk 3 — Rescale synthetic fixtures to nMgy-scale

**Files:**
- `astro/src/astropt3/data/synthetic.py` — in `make_record`:
  - `amps = rng.uniform(0.01, 0.1, size=3)` (was `20.0, 80.0`).
  - `flux += rng.normal(0.0, 0.001, ...)` (was `0.1`).
  - Keep the blob + redshift-correlated sigma structure unchanged.
  - Update the module docstring's flux description (arbitrary units → nMgy).

**Rationale:** Under the 0.01 nMgy divisor, `asinh(20/0.01)≈8.6` saturates;
the 4% output range + per-patch standardization would blow up noise into the
only signal → smoke training fails. nMgy-scale puts the 0.01 knee in the faint
regime, galaxy cores at ~0.1 nMgy (`asinh(10)≈3`, off saturation), and the
~398 nMgy clamp never fires on fixtures.

**Verify:** `uv run pytest tests/test_mmu.py tests/test_packing.py` (fixtures
still schema-valid). `uv run python -m astropt3.train_smoke --config
configs/model/test-tiny.yaml --steps 50 --assert-decrease` — the smoke gate;
verifies nMgy fixtures + physical norm let the model learn. **This is the key
material gate for the data-side change.**

---

### Chunk 4 — Remove `norm_stats` plumbing (full clean removal)

This is the biggest diff. Do it as one chunk because the callers are coupled.

**Files:**
- `astro/src/astropt3/config_io.py` — delete
  `sequencer_kwargs_from_data_config`. Keep `load_data_config` /
  `load_model_config` / `resolve_data_root`.
- `astro/src/astropt3/data/nanotron_loader.py` — remove the `norm_stats`
  constructor arg (line 137) and the
  `sequencer_kwargs = sequencer_kwargs_from_data_config(...)` block
  (lines 164-166); construct `ObjectSequencer(config)` directly. Remove the
  `from ..config_io import ... sequencer_kwargs_from_data_config` import.
  Remove the `norm_stats=getattr(dataset_args, "norm_stats", None)` line 372.
- `astro/src/astropt3/eval/val_loss.py` — remove `norm_stats` from
  `val_batches`, `evaluate`, and the `from_checkpoint` wrapper; remove the
  `--norm-stats` argparse arg (line 122).
- `astro/src/astropt3/eval/linear_probe.py` — remove `norm_stats_kwargs` from
  `collect_probe_objects`; construct `ObjectSequencer(config)` directly.
- `astro/scripts/run_probe_sweep.py` — remove `--norm-stats` arg and the
  `norm_stats=args.norm_stats` call (line 91).
- `astro/scripts/check_pilot_data.py` — remove the
  `sequencer_kwargs_from_data_config` import (line 30) and the
  `seq_kwargs = ...` block (line 138-140, including the "run
  compute_norm_stats.py first" note). Construct `ObjectSequencer(config)`.

**13 nanotron training yamls** — strip the `norm_stats:
astro/configs/data/pilot_images_spectra.yaml` line from each:
`configs/nanotron/astropt3-{70m,6p9b,160m-shakeout,70m-jetformer-lowlr,12b,1b,
1p4b,70m-baby,70m-jetformer,410m,160m,2p8b,2p8b-tp2-dryrun}.yaml`.

**Verify:** `uv run pytest` — `test_eval.py`, `test_loader_resume.py`,
`test_nanotron_loader.py` will fail on the removed `norm_stats=None` arg; fixed
in Chunk 7. `grep -rn "norm_stats\|sequencer_kwargs" astro/src astro/scripts
astro/configs` should return no hits.

---

### Chunk 5 — Rewrite `generate.py` inverse path

**Files:**
- `astro/scripts/generate.py`:
  - Remove `--norm-stats` arg (line 126).
  - Remove the `sequencer_kwargs`/`asinh_params` block (lines 144-161) and the
    `from astropt3.config_io import ... sequencer_kwargs_from_data_config` +
    `from astropt3.data.transforms import asinh_params_from_percentiles`
    imports.
  - Construct `ObjectSequencer(model.config)` directly.
  - Replace `maybe_sinh(imgs)` (lines 185-188) with
    `physical_inverse(imgs, record["image"]["band"])` (imported from
    `band_registry`). The template record always carries `image.band`
    (`load_template_record` → synthetic or `MMUIterableDataset`), so band
    names are free at decode.
  - Update the module docstring (the `--norm-stats` paragraph → physical
    inverse keyed by template bands; the "exact for jetformer / qualitative
    for affine" note still holds, since affine standardization discards
    per-patch mean/std).

**Verify:** `uv run pytest tests/test_generate.py` — the test constructs
`ObjectSequencer(jet_config)` with no stats (already does); should pass after
the fixture rescale. Run `test_generate` modes manually if the test doesn't
cover the inverse path.

---

### Chunk 6 — Config mirror (`image_norm_divisor`)

**Files:**
- `astro/src/astropt3/configuration_astropt3.py` — add
  `image_norm_divisor: float = 0.01` to `AstroPT3Config.__init__` (default
  matches the `band_registry._DIV_FACTOR` constant). No size-yaml changes
  (none set it; default applies).
- `astro/src/astropt3/config_io.py` `load_model_config` — ensure old
  checkpoints get `image_norm_divisor = _DIV_FACTOR` (the module constant) if
  absent, so `config.json` is self-describing. (transformers' `from_pretrained`
  already fills defaults, but be explicit for checkpoints saved before this
  field existed.)

**Verify:** `uv run pytest tests/test_saveload.py` — checkpoint round-trip
preserves `image_norm_divisor`. `uv run python scripts/count_params.py` —
no-op (band_registry adds no params; the field is config-only); should still
pass.

---

### Chunk 7 — Replace/fix tests

**Files:**
- DELETE `astro/tests/test_transforms.py` — all 5 tests cover retired PU
  functions (`asinh_params_from_percentiles`, `asinh_stretch`, `ASINH_ALPHA`,
  sequencer-with/without-stats).
- NEW `astro/tests/test_physical_norm.py` — cover:
  1. `band_registry` constants: `rescale_factors("des-g") == (1.0, 1.0)`;
     all 15 bands finite + positive rescale/clamp.
  2. `_normalize` output range for des: `arcsinh(flux/0.01)·0.01`, finite,
     within `±arcsinh(clamp/0.01)·0.01`.
  3. `RAW_BANDS` passthrough: flux unchanged for `["rgb-r","rgb-g","rgb-b"]`.
  4. Unknown band (`"euclid-g"`) → `NotImplementedError`.
  5. Empty band list `[]` → flux unchanged (the `all(...)` vacuous-True guard;
     add the guard in Chunk 1 and assert it here).
  6. Encode→decode round-trip up to the clamp: `physical_inverse(physical_normalize(flux,
     bands), bands) ≈ flux` for flux below the clamp; clamped flux is lossy.
- `astro/tests/test_jetformer.py` line ~144 — fix the expected flux:
  `flux = torch.asinh(torch.as_tensor(record["image"]["flux"]))` →
  `flux = physical_normalize(torch.as_tensor(record["image"]["flux"]),
  record["image"]["band"])`. The rest of the assertion (jetformer tokens ==
  patchify of the normalized flux; affine standardizes to zero mean) holds.
- `astro/tests/test_eval.py` line 30 — drop `norm_stats=None`.
- `astro/tests/test_loader_resume.py` line 118 — drop `norm_stats=None`.
- `astro/tests/test_nanotron_loader.py` — drop any `norm_stats` arg.
- `astro/tests/conftest.py` — `ObjectSequencer(tiny_config)` unchanged (the
  stats params just go away).

**Verify:** `uv run pytest` — the material gate. All tests pass. This is the
primary verification for the whole change.

---

### Chunk 8 — Delete `compute_norm_stats.py` + yaml `normalization` block; move lambda check

**Files:**
- DELETE `astro/scripts/compute_norm_stats.py`.
- `astro/configs/data/pilot_images_spectra.yaml` — delete the `normalization`
  block (`asinh_alpha`, `image_p1`, `image_p99`, `stats_provenance`). Keep
  `name`/`version`/`sources`/`paths`/`split`/`packing`.
- `astro/scripts/check_pilot_data.py` — add the spectrum lambda-range sanity
  print (port from `compute_norm_stats.py`: iterate records, track
  `lambda_min`/`lambda_max`, print range + warn if outside 3600–9824 Å).

**Verify:** `uv run pytest` (no test imports `compute_norm_stats`).
`uv run python scripts/check_pilot_data.py --help` (smoke); run against
synthetic if a real shard isn't available.

---

### Chunk 9 — Docs + nanotron fork mirror

**Docs (astro/):**
- `README.md` line 55 — remove the `compute_norm_stats.py` step; describe
  physical norm (band-registry-keyed, no calibration run).
- `docs/training.md` lines 79-86, 134, 327 — remove the norm_stats/asinh
  calibration section + the yaml `norm_stats:` arg + the
  `compute_norm_stats.py` troubleshooting reference.
- `docs/architecture.md` lines 39-59 — replace the PU asinh stretch
  description (step 2) with the physical-norm story (rescale → clamp →
  arcsinh/0.01); keep the per-patch standardization step.
- `PLAN.md` lines 92, 98, 107, 139, 242, 266, 380 — update the file tree
  (remove `compute_norm_stats.py`, add `band_registry.py`; update
  `transforms.py` description) and the phase notes (asinh p1/p99 → physical
  norm).

**nanotron fork** (`nanotron/` submodule, branch `main`):
- Mirror the `nanotron_loader.py` `norm_stats` removal in the fork's loader
  copy (the fork consumes flat micro-batch dicts from `data/nanotron_loader.py`;
  physical norm lives in `astro/` so the fork stays thin).
- No change to `src/nanotron/models/astropt3.py` — physical norm is data-side;
  PP=1/TP-replicated modality modules are unaffected.
- `tools/astropt3/convert_*` — no change (conversion is weight-level, not
  normalization-level).

**Migration note:** add a one-line note to `docs/architecture.md` or a new
`docs/migration.md`: old PU-trained checkpoints are **incompatible** with
physical norm (different normalization target); `load_model_config` back-fills
`image_norm_divisor` but the weights were trained on a different target.
Declare incompatible, retrain.

**Verify:** `uv run pytest` (docs don't affect tests). `grep -rn
"compute_norm_stats\|norm_stats\|asinh_params\|image_p1\|image_p99" astro/
--include="*.py" --include="*.yaml" --include="*.md"` should return only the
historical `wandb/` run logs (untouched) and the migration note.

---

## Validation gates (run after Chunk 7, again after Chunk 9)

1. **`uv run pytest`** — the material gate. All CPU tests pass
   (`@pytest.mark.gpu` excluded via `addopts`).
2. **`uv run python -m astropt3.train_smoke --config configs/model/test-tiny.yaml
   --steps 50 --assert-decrease`** — the smoke gate. Verifies nMgy-scale
   fixtures + physical norm let the model learn (the synthetic-fixture rescale
   is gated here).
3. **`uv run python scripts/count_params.py`** — no-op (band_registry adds no
   params; `image_norm_divisor` is a config field). Still passes; confirms no
   accidental param drift.

---

## Edge cases & risks (named, not blocking)

- **Unknown band** → `NotImplementedError` (registry + passthrough guard).
  Covered by `test_physical_norm`.
- **RAW_BANDS passthrough** — `all(b in RAW_BANDS for b in bands)` → flux
  unchanged. Covered.
- **Empty band list** — vacuous-True guard; covered.
- **Mixed RAW + registry bands** — `all(...)` is False if any band isn't RAW,
  then `_physical_factors` raises on the RAW band. Mixed records raise; not a
  real case for us (records are all-`des-*` or all-`rgb-*`). Named in the
  `band_registry` docstring.
- **nanotron fork parity** — physical norm lives in `astro/`; the fork mirrors
  only the loader's `norm_stats` removal. PP=1/TP-replicated modality modules
  unaffected (norm is data-side).
- **Old PU-trained checkpoints** — incompatible; migration note in docs.

---

## What's NOT in scope

- Spectra normalization (unchanged: raw flux → patchify → standardize).
- The nanotron fork's `models/astropt3.py` (no change; norm is data-side).
- The `tokenization.py` special-token ids (frozen per AGENTS.md).
- Any GPU/training run (CPU tests/smoke only on this machine).
- galactiktok's `ImageTransformerTokenizer` (its encode/decode/MAE machinery
  is not ported — only the normalization primitives).

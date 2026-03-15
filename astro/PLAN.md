# SmolLM AstroPT — Incremental Implementation Plan

See `smollm_astropt_plan.md` (repo root) for the full architectural vision.
This document tracks the concrete, incremental steps to get there.

---

## Phase 1 — Data Loading: Smith42/Galaxies  ← **current**

Goal: prove the data pipeline works end-to-end before any training.

### 1a — `astro/data/galaxy_dataset.py`
- [x] `GalaxyIterableDataset`: wraps `datasets.load_dataset("Smith42/galaxies",
  revision="v2.0", streaming=True)` as a PyTorch `IterableDataset`.
  - v2.0 includes all metadata columns inline — no separate join needed.
- [x] Each item is a dict with PIL image + 14 curated metadata fields (redshift,
  stellar mass, SFR, DESI grz photometry, Sérsic n/b/a, Galaxy Zoo fractions,
  Petrosian half-light radius). Missing/NaN fields are silently dropped.
- [x] Configurable split (`train` / `validation` / `test`) and optional shard offset
  for multi-worker / multi-GPU use.

### 1b — `astro/data/smolvlm_adapter.py`
- [x] `GalaxySmolVLMDataset`: thin wrapper around `GalaxyIterableDataset` that
  converts each galaxy into a SmolVLM conversation and runs it through the HF
  `processor`, producing `{input_ids, attention_mask, labels, pixel_values}`.
- Training objective: image → short metadata description
  (`"Galaxy {dr8_id}. Angular size proxy: {galaxy_size} px²."`)
  Loss computed on the assistant response only (user/system masked).
- Default: full `image` (512×512) matching original AstroPT; `image_crop`
  (256×256) available via `use_crop=True` for faster smoke tests.

### 1c — `astro/test_galaxy_load.py`
- [x] CLI script: loads N batches, prints shapes and a sample decoded text.
- Verifies: images load, processor runs, pixel_values shape is correct.

**Note on training objective**: the original AstroPT `LLMModalityDataset`
(https://github.com/Smith42/astroPT) uses the same pattern:
- User turn = image patches (`<|begin_images|>…<|end_images|>`)
- Assistant turn = scalar metadata properties
This confirms the metadata-as-text-target approach in Phase 1 is consistent
with the direction of the original codebase.

**Open question for Mike**: should the assistant target include more fields
(e.g. photometric redshift, morphology flags) once cross-matched MMU data is
available, or stay as simple size metadata for Phase 1?

---

## Phase 2 — Stage 1 Training: Connector Warmup (backbone frozen)

Goal: train only the SigLIP → MLP projection on 135M SmolLM backbone.

### 2a — YAML mixture config
- `astro/configs/galaxies_135m_stage1.yaml`
- Point to `GalaxySmolVLMDataset`; set `sampling_strategy: all`.

### 2b — Training script
- Extend `vision/smolvlm2/smolvlm/train/train.py` (or a thin wrapper).
- Freeze `language_model` and `vision_tower`; train only `connector`.
- Target: 1 epoch over validation split (~86k examples) as smoke test.
- Then full train split (~8.5M) once smoke test passes.

### 2c — Logging & checkpoints
- W&B run tagged `astro-galaxies-135m-stage1`.
- Save checkpoint every 1000 steps.

---

## Phase 3 — Stage 2 Training: Full Continual Pretraining

Goal: unfreeze SmolLM backbone; jointly train connector + backbone.

### 3a — Lower learning rate schedule
- Follow SmolVLM recipe: `language_model_lr` ≈ 2e-5, `connector_lr` ≈ 1e-4.
- Mix in ~5% astronomy text (ADS abstracts) to preserve language capability.

### 3b — Scale up
- 135M → 360M → 1.7B → 3B (sequentially, pending compute).

---

## Phase 4 — Additional MMU Modalities

Dependency: Phase 0 from `smollm_astropt_plan.md` (MMU upload + cross-matching).

### 4a — FITS multi-band images
- `astro/data/fits_dataset.py`
- Linear channel projection N → 3 before SigLIP.

### 4b — Spectra (wavelength shuffle)
- `astro/data/spectra_dataset.py`
- 1D patching + wavelength shuffle (4× default) + MLP projection.

### 4c — Light curves (temporal shuffle)
- `astro/data/lightcurve_dataset.py`
- Same pattern as spectra, applied to time axis.

### 4d — Scalar catalog values
- Direct linear projection; negligible complexity.

### 4e — Cross-modal training example assembly
- Combine image + spectrum + scalars for same object into one context window.

---

## Phase 5 — Evaluation

- Linear / MLP probes on physical properties (redshift, stellar mass, SFR,
  morphology) following Sanjaripour, Smith et al. (ICLR 2026).
- Direct comparison vs. AstroPT 89M, AION-1 3B.
- Tokenization ablations (affine vs. MLP; shuffle ratios 2×/4×/8×).

---

## Key Files

| Path | Purpose |
|------|---------|
| `astro/data/galaxy_dataset.py` | HuggingFace streaming loader |
| `astro/data/smolvlm_adapter.py` | SmolVLM conversation converter |
| `astro/test_galaxy_load.py` | End-to-end smoke test |
| `astro/configs/` | YAML dataset mixture configs |
| `vision/smolvlm2/smolvlm/train/train.py` | Existing training entry point (reuse) |

# AGENTS.md

This file provides guidance to Agents when working with code in this repository.

## What this repo is

AstroPTv3: a from-scratch suite of multimodal astronomical foundation models
(70M–12B, Pythia-mirrored) — a SmolLM3 decoder body fed continuous image/
spectra patch tokens with per-modality regression heads, pretrained on the
Multimodal Universe. The repo is a fork of `huggingface/smollm`: `text/`,
`vision/`, `tools/` are **read-only upstream reference**; all project code
lives in `astro/`. The approved phase plan (decisions are fixed) is
`astro/PLAN.md`; hard constraints are in `AGENTS.md` — the critical ones:
**no GPU or training runs on this machine** (CPU tests/smoke only), tests
may use the network but only `network`-marked ones may require it
(ADR 0006 streams the corpus live; deselect with `-m 'not network'` when
the HF hub is down), special-token ids in
`astro/src/astropt3/tokenization.py` are frozen, and dependencies are
managed only through uv in `astro/pyproject.toml`.

## Commands

All from `astro/`:

```bash
uv sync --extra dev                       # create/update the venv
uv run pytest                             # CPU suite (gpu-marked tests excluded via addopts)
uv run pytest tests/test_model.py::test_pad_invariance   # single test
uv run python scripts/count_params.py     # size table; asserts ±10% of nominal
uv run python -m astropt3.train_smoke \
    --config configs/model/test-tiny.yaml --steps 50 --assert-decrease
```

These three (pytest, count_params, train_smoke) are the phase verification
gates and must all pass before any phase is declared done. `@pytest.mark.gpu`
tests exist to be run on the training machine, never here.

## Architecture

Data flows record → sequence → packed batch → model; the contract between
stages is implicit and easy to break, so understand it before editing:

1. **Records** are MMU-schema dicts (`image.flux` float32 (3,152,152);
   `spectrum` with 7781-bin `flux/lambda/ivar/mask`). `data/synthetic.py`
   generates schema-identical fixtures so everything runs offline; both
   modalities are optional per record (image-only is the common case, ~13M
   of 14M; spectrum-only rows are the non-crossmatched ZWARN==0 DESI
   spectra, ADR 0005). Real records are **streamed live from the HF hub**
   by `data/streaming.py` (ADR 0006, which deleted the local reshard and
   its `PILOT_FEATURES` schema): three lsdb/HATS sources — images-only,
   spectra-only, and the `how="inner"` crossmatch — interleaved **per
   record** by fixed weights (0.60/0.15/0.25, provisional) using a repeating
   draw pattern, never a sampler, so no RNG state is ever checkpointed.
   `row_to_record` is the sole decode adapter over MMU-native rows.
   Partitions are addressed by index (`CatalogStream._delayed_partitions`),
   split across `world_size × num_workers` by modulo, and buffered whole —
   so resume is `(epoch, partition cursor, row offset)` per source, all
   ints, and stays exactly no-replay. `data_root` is `synthetic` or `mmu`;
   a stale path to the old corpus raises. Val reserves the first
   `VAL_PARTITIONS` partitions of every source (whole partitions ⇒
   spatially disjoint).
2. **`ObjectSequencer`** (`data/packing.py`) turns a record into an
   `ObjectSeq`: central 96×96 crop (`packing.IMAGE_CROP`; JWST cubes are
   already 96×96) + physical band-registry normalization
   (`data/band_registry.py`: rescale to nanomaggies → bright-pixel clamp →
   `arcsinh(x/0.01)` — tokens are flux in knee units of 0.01 nMgy = 10 pMgy,
   O(1) values, keyed on the record's band names — no per-corpus
   calibration; unknown bands raise; spectra get the symmetric ADR 0007
   treatment in `data/spectral.py`: DESI f_λ → AB nMgy via `f_ν = f_λ·λ²/c`
   on the fixed DESI grid, then `arcsinh(f_ν/10 nMgy)`, invertible with no
   side info; unknown grids raise) + patchify (`tokenization.py`) + per-patch
   standardization (`data/transforms.py`) per modality (jetformer configs
   SKIP the standardization — the exact-likelihood loss needs an invertible
   record -> token map, and standardization discards each patch's
   mean/std), wrapped in frozen
   special tokens: `<|bos|> <|begin_m|> …placeholders… <|end_m|>` per
   modality, spans serialized in a **uniform random order seeded on
   `crc32(object_id) ^ epoch`** (resume-exact, always on — every
   conditional among the present spans lands in training; ADR 0008,
   generalizing 0005's bimodal 50/50 flip; pre-rule fixed-order
   checkpoints are incompatible). Images → 144 patch-8 tokens
   (192 floats); spectra → 31 patch-256 tokens with normalized per-patch
   mean wavelength as a continuous position; ADR 0008 adds one-token
   **scalar modalities** (`Z` gated on ZWARN==0, `ebv`, joint 3-band
   `photometry`; `data/scalar_registry.py` fixed transforms, loss_weight
   0.1) predicted by `GMMHead`s (`scalar_gmm_k`) under BOTH tokenisers —
   no flow, no logdet, plain `gmm_nll` on the raw normalized value. The
   linear probe builds scalar-free sequences (`include_scalars=False`)
   so R² stays a representation metric; `eval/scalar_head.py` is the
   ask-the-model metric (`nmad`/`outlier_frac`/`coverage_1sig`).
3. **`PackedCollator`** greedily packs whole objects (never split) into
   fixed-length rows. Two invariants matter:
   - `position_ids` restart at 0 per object and **are the document mask**:
     the model passes `attention_mask=None` and transformers'
     `create_causal_mask` detects the packed format from the restarts
     (torch ≥ 2.6). Pads get position 0, isolating each as its own segment.
   - Flattened `modality_values`/`modality_positions` are concatenated in
     row-major (batch, time) order — exactly the order boolean-mask
     indexing produces. The model relies on this to align values without
     indices.
4. **`AstroPT3Model`** (`modeling_astropt3.py`): 64-id `embed_tokens` (no
   text vocab, no lm_head) + additive deltas
   `encoder_m(value) + pos_embed_m(position)` at placeholder slots →
   `SmolLM3Model(inputs_embeds=…)` → per-modality `Decoder` heads. Loss is
   Huber at positions one left of each modality token (`<|begin_m|>`
   predicts patch 0 — astroPT's `starts-1` alignment), via
   `left_shift_mask`; weighted mean over modalities present. The
   `tokeniser: jetformer` option (JetFormer/GIVT, ported from astroPT v2's
   sogol_branch) instead flows each modality's patch tokens through a
   per-modality `TinyFlow1D` and predicts them with a per-modality `GMMHead`;
   loss is exact patch-space likelihood `mean(NLL_GMM(z) - logdet)`, which
   can go negative. A noise curriculum (`jetformer_noise_max` -> `_min`,
   driven via `set_jet_noise_frac`) perturbs only the embedded z copy in
   training mode. The nanotron fork mirrors all of it (flows/GMM heads on
   the embedding/head blocks, z+logdet routed to the loss, TP-synced noise);
   sampling lives in `astropt3.generation` + `scripts/generate.py`.
5. **Config**: `AstroPT3Config(SmolLM3Config)` carries a `modalities` list
   of dicts; `import astropt3` registers the Auto classes, so it must be
   imported before `AutoModel.from_pretrained` on a checkpoint. Size YAMLs
   in `configs/model/` are loaded by `config_io.load_model_config`.

**Two implementations, one weight source of truth**: this transformers
implementation is the release/probing artifact and CPU test target; actual
pretraining happens in the nanotron fork (`nanotron/` git submodule, branch
`main`) that consumes flat micro-batch dicts built by
`data/nanotron_loader.py` (`{m}_values`/`{m}_positions`/`{m}_mask` +
`input_ids`/`position_ids` — flat because nanotron's device mover only
transfers top-level tensors). The fork adds
`src/nanotron/{models/astropt3.py,config/astropt3_config.py}`, the
`astropt3_streaming` dataset type in `run_train.py`, and
`tools/astropt3/convert_{nanotron_to_hf,hf_to_nanotron}.py`; PP=1 and
`tp_mode: ALL_REDUCE` are asserted (modality modules are TP-replicated via
nanotron's tied-parameter mechanism). `nanotron_loader.py` must stay
importable without nanotron; keep all modality/packing logic in `astro/` so
the fork stays thin. gpu-marked tests (`tests/test_nanotron_gpu.py`,
`tests/test_phase4_gpu.py`) cover HF↔nanotron parity, TP=2 replicated
grads, 50-step smoke + conversion, the Pythia checkpoint schedule, and
kill/resume — see PLAN Phase 3 notes for the venv recipe.

**Checkpointing & eval (Phase 4)**: `checkpoints.checkpoint_schedule:
pythia` saves at steps 1,2,4,…,512 plus every `checkpoint_interval`
(canonical schedule in `checkpoint_schedule.py`, lazy-imported by the fork's
trainer). Each checkpoint stores the stream position under
`dataset_state/dp_{rank}.pt` — state is captured at the START of the current
partial packing row, so resume re-draws the untrained partial row and
continues the exact micro-batch sequence at any `num_loading_workers`
(workers > 0 rides torchdata's StatefulDataLoader and must resume with the
same worker count; `object_id_log` writes the per-object no-replay audit
trail). Evaluation
never runs in the trainer: `scripts/run_probe_sweep.py` polls a run's
checkpoint dir (gated on `latest.txt`), converts each step to HF, and runs
`astropt3.eval.val_loss` (fixed deterministic val batches; synthetic val
uses record indices ≥ 10M), `astropt3.eval.linear_probe` (ridge probe of
redshift `Z` from mean-pooled hidden states), and `astropt3.eval.samples`
(fixed-template image/spectrum panels per checkpoint, mirrored to the
sweep's own wandb run with `--wandb` — ADR 0003) — run it on a spare GPU
alongside training (`EVAL_GPU=<id>` on the launch scripts co-launches it).

A behavior to remember when touching data or fixtures: per-patch
standardization turns flat/noise-only patches into irreducible N(0,1)
targets — synthetic data must contain patch-scale structure or smoke
training cannot learn. Synthetic image flux is nMgy-scale (cores ~0.1,
noise ~0.001) so the physical normalization's fixed 0.01 nMgy arcsinh knee
lands in the same regime as on real LegacySurvey data; pre-physical-norm
(PU-asinh) checkpoints are incompatible — retrain.

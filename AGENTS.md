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
**GPU work and training runs are allowed here** (this box has 2×A100 80GB
and slurm), but the CPU suite stays the fast gate and multi-day runs belong
on the training cluster by preference; tests
may use the network but only `network`-marked ones may require it
(ADR 0015 streams the corpus live via LSDB; deselect with `-m 'not network'`
when the HF hub is down), special-token ids in
`astro/src/astropt3/tokenization.py` are frozen, and dependencies are
managed only through uv in `astro/pyproject.toml`.

## Commands

All from `astro/`:

```bash
uv sync --extra dev                       # create/update the venv
uv run pytest                             # CPU suite (gpu-marked tests excluded via addopts)
uv run pytest tests/test_model.py::test_pad_invariance   # single test
uv run python scripts/count_params.py     # size table; asserts ±10% of nominal
uv run pytest -m network tests/test_lsdb_stream.py   # live hub stream check
```

These two (pytest, count_params) are the phase verification gates and must
pass before any phase is declared done; ADR 0015 retired the `train_smoke`
gate together with the local-corpus path it exercised. `@pytest.mark.gpu`
tests run here too (`uv run pytest -m gpu` in the GPU venv); the node is
shared, so pin a device and check `nvidia-smi` before claiming one.

## Architecture

Data flows record → sequence → packed batch → model; the contract between
stages is implicit and easy to break, so understand it before editing:

1. **Records** are MMU-schema dicts (`image.flux` float32 (3,152,152);
   `spectrum` with 7781-bin `flux/lambda/ivar/mask`); both modalities are
   optional per record (image-only is the common case). Offline fixtures
   live in `tests/legacy_fixture.py` (ADR 0015 deleted `data/synthetic.py`
   with the rest of the local-corpus path). Real records are **streamed
   live from the HF hub inside each DataLoader worker** by
   `data/nanotron_loader.py` via LSDB (ADR 0015): a plain
   `lsdb.streams.InfiniteStream` over LegacySurvey, or — with
   `crossmatch_desi: true` — a `CrossMatchStream` (lsdb PR #1584, tracked
   as a git dependency on `selective-x-match`) with DESI on the left and
   LegacySurvey joined right through `OuterKdTreeCrossmatch`
   (`data/outer_crossmatch.py`), which also recovers the unmatched Legacy
   images a plain join would drop. `count_fraction_threshold` is pinned to
   0.0: 0.5 saved 31% Legacy bytes but cost ~20% tokens/s and image
   coverage (EXPERIMENTS.md §9). Each worker owns a dask `LocalCluster`
   (threads only, `partitions_per_chunk=4`) so the next draw's fetch
   overlaps the current one's decode; rows decode through
   `NestedFrame.map_rows`, never `iterrows`. `num_loading_workers` counts
   concurrent LSDB readers per DP rank — watch box-wide RSS. The stream is
   **cursorless**: checkpoints restore weights/optimizer/scheduler/RNG,
   never the record position, so resume opens a fresh stream and may
   revisit records (ADR 0015; `docs/training.md` §5). Recognized
   transport/storage errors discard the iterator and reopen under bounded
   backoff; decode/validation and unknown errors fail immediately.
   Provenance (lsdb/hats versions, catalog, columns) is logged at startup
   and `uv.lock` pins the resolved LSDB commit.
2. **`ObjectSequencer`** (`data/packing.py`) turns a record into an
   `ObjectSeq`: central 96×96 crop (`packing.IMAGE_CROP`; JWST cubes are
   already 96×96) + physical band-registry normalization
   (`data/band_registry.py`: rescale to nanomaggies → bright-pixel clamp →
   `arcsinh(x/0.01)` — tokens are flux in knee units of 0.01 nMgy = 10 pMgy,
   O(1) values, keyed on the record's band names — no per-corpus
   calibration; unknown bands raise; spectra get the symmetric ADR 0007
   treatment in `data/spectral.py`: DESI f_λ → AB nMgy via `f_ν = f_λ·λ²/c`
   on the fixed DESI grid, then `arcsinh(f_ν/10 nMgy)`, invertible with no
   side info; unknown grids raise) + patchify (`tokenization.py`) per
   modality — no per-patch standardization: the jetformer tokeniser's
   exact-likelihood loss needs an invertible record -> token map, and
   standardization would discard each patch's mean/std — wrapped in frozen
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
   0.1) predicted by `GMMHead`s (`scalar_gmm_k`) directly — no flow, no
   logdet, plain `gmm_nll` on the raw normalized value. The
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
   `SmolLM3Model(inputs_embeds=…)` → per-modality regression heads. The
   jetformer tokeniser (JetFormer/GIVT, ported from astroPT v2's
   sogol_branch) is the sole regression head: each patch modality's tokens
   flow through a per-modality `TinyFlow1D` to a latent z that is both
   embedded and predicted by a per-modality `GMMHead`, with loss the exact
   patch-space likelihood `mean(NLL_GMM(z) - logdet)` at positions one left
   of each modality token (`<|begin_m|>` predicts patch 0 — astroPT's
   `starts-1` alignment), via `left_shift_mask`; the loss can go negative;
   weighted mean over modalities present. A noise curriculum
   (`jetformer_noise_max` -> `_min`, driven via `set_jet_noise_frac`)
   perturbs only the embedded z copy in training mode. Scalar modalities
   (`Z`, `ebv`, `photometry`) bypass the flow and go straight to a
   `GMMHead` on the raw normalized value. The nanotron fork mirrors all of
   it (flows/GMM heads on the embedding/head blocks, z+logdet routed to
   the loss, TP-synced noise); sampling lives in `astropt3.generation` +
   `scripts/generate.py`.
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
`tests/test_jetformer_gpu.py`) cover HF↔nanotron forward/loss parity, exact
weight-conversion round-trips, TP=2 replicated gradients (including the
noise curriculum), and a 50-step smoke run with checkpoint conversion — see
PLAN Phase 3 notes for the venv recipe.

**Checkpointing & eval**: `checkpoints.checkpoint_schedule: pythia` saves
at steps 1,2,4,…,512 plus every `checkpoint_interval` (canonical schedule
in `checkpoint_schedule.py`, lazy-imported by the fork's trainer);
`latest.txt` is written last, so any step dir it covers is complete.
Checkpoints carry weights/optimizer/scheduler/RNG and **nothing else** —
the LSDB stream is cursorless, so resume opens a fresh stream and records
may be revisited (ADR 0015; `docs/training.md` §5). Evaluation never runs
in the trainer: convert a checkpoint with the fork's
`tools/astropt3/convert_nanotron_to_hf.py`, then score the HF artifact
with `astropt3.eval` (`val_loss`, `linear_probe`, `scalar_head`,
`samples`); ADR 0015 removed the automated `run_probe_sweep` sidecar.

A behavior to remember when touching data or fixtures: per-patch
standardization turns flat/noise-only patches into irreducible N(0,1)
targets — synthetic fixtures must contain patch-scale structure or smoke
training cannot learn. The fixtures' image flux is nMgy-scale (cores ~0.1,
noise ~0.001) so the physical normalization's fixed 0.01 nMgy arcsinh knee
lands in the same regime as on real LegacySurvey data; pre-physical-norm
(PU-asinh) checkpoints are incompatible — retrain.

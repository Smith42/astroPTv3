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
must pass without network access, special-token ids in
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
   generates schema-identical fixtures so everything runs offline; records
   may lack `spectrum` (image-only is the common case, ~13M of 14M). Real
   records flow through `data/mmu.py`: `scripts/prepare_pilot_data.py`
   (login node, `[data]` env, network) lsdb-LEFT-crossmatches the two MMU
   HATS collections into local parquet shards (train/val by hashing coarse
   HEALPix tiles — spatially disjoint), and `MMUIterableDataset` streams
   them back offline, sharded by DP rank and DataLoader worker. The asinh
   p1/p99 calibration in `configs/data/pilot_images_spectra.yaml` is
   written by `scripts/compute_norm_stats.py`; `scripts/check_pilot_data.py`
   is the sanity/throughput gate.
2. **`ObjectSequencer`** (`data/packing.py`) turns a record into an
   `ObjectSeq`: asinh stretch + patchify (`tokenization.py`) + per-patch
   standardization (`data/transforms.py`) per modality, wrapped in frozen
   special tokens: `<|bos|> <|begin_m|> …placeholders… <|end_m|>` per
   modality in **alphabetical registry order**. Images → 361 patch-8 tokens
   (192 floats); spectra → 31 patch-256 tokens with normalized per-patch
   mean wavelength as a continuous position.
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
   `left_shift_mask`; weighted mean over modalities present.
5. **Config**: `AstroPT3Config(SmolLM3Config)` carries a `modalities` list
   of dicts; `import astropt3` registers the Auto classes, so it must be
   imported before `AutoModel.from_pretrained` on a checkpoint. Size YAMLs
   in `configs/model/` are loaded by `config_io.load_model_config`.

**Two implementations, one weight source of truth**: this transformers
implementation is the release/probing artifact and CPU test target; actual
pretraining happens in a nanotron fork (`nanotron/` git submodule, arrives
Phase 3) that consumes the same batch dicts. Keep all modality/packing
logic in `astro/` so the fork stays thin.

A behavior to remember when touching data or fixtures: per-patch
standardization turns flat/noise-only patches into irreducible N(0,1)
targets — synthetic data must contain patch-scale structure or smoke
training cannot learn, and the real asinh scale is calibrated from data
(Phase 2 `compute_norm_stats.py`), not by eye.

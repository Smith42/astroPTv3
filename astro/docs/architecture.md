# AstroPTv3 architecture

*Background for anyone picking up the project. The operational counterpart is
[`training.md`](training.md); the authoritative phase plan with all fixed
decisions is [`../PLAN.md`](../PLAN.md).*

## What this is

AstroPTv3 (NAIRR260009) is a from-scratch suite of **multimodal astronomical
foundation models** spanning 70M–12B parameters, mirroring the Pythia suite's
sizes and checkpoint schedule so that scaling behaviour can be studied across
the whole family. The recipe combines two lineages:

- **AstroPT** (Smith et al.): autoregressive *next-token regression* over
  continuous embeddings of astronomical data — no quantization, no text
  vocabulary. The model predicts the next image/spectrum patch directly and
  is trained with a Huber loss.
- **SmolLM3**: the transformer body — a modern decoder stack with grouped-
  query attention (GQA), RoPE with NoPE every 4th layer, RMSNorm, SwiGLU,
  and document-masked packed sequences.

The pretraining corpus is the **Multimodal Universe** (MMU) pilot:
DESI Legacy Survey north images (~14.2M galaxies, 3×152×152 flux cubes)
LEFT-crossmatched with DESI EDR SV3 spectra (~1.1M, 7781-bin), so roughly
1-in-14 objects carries both modalities and the rest are image-only. Both
situations are first-class: an object contributes whatever modalities it has.

## From a galaxy to a token sequence

Every training example ("object") is one astronomical source. The path from
survey data to model input:

1. **Record**: an MMU-schema dict. `image.flux` is float32 `(3, 152, 152)`
   (g/r/z bands); `spectrum` (when present) has 7781-bin `flux`, `lambda`,
   `ivar`, `mask`. Records stream from local parquet shards
   (`data/mmu.py::MMUIterableDataset`) or from the synthetic generator
   (`data/synthetic.py`, schema-identical, used by every test).

2. **Physical normalization** (`data/band_registry.py`, ported from
   galactiktok `feat/norm`): image flux is normalized physically, keyed on
   the record's band names — rescale to LegacySurvey nanomaggies
   (zeropoint + pixel-area factors from the surveys' documentation) →
   clamp survey-flagged bright pixels (per-band `m_bright` ceiling) →
   `arcsinh(flux/0.01)·0.01`. No data-driven calibration; unknown bands
   raise, `rgb-*` composites pass through raw. Invertible up to the clamp
   (`physical_inverse`). Spectra are not stretched; masked bins are zeroed.

   > **Migration note:** checkpoints trained before this change (on the
   > percentile-calibrated Platonic-Universe asinh stretch) are
   > **incompatible** — the normalization target is different. The config
   > field `image_norm_divisor` is back-filled on load, but the weights were
   > trained on a different target: declare incompatible, retrain.

3. **Patchify** (`tokenization.py`):
   - images → **361 patches of 192 floats** (8×8×3; patch 8 because
     152 = 8×19 — 16 does not divide 152), integer patch-index positions;
   - spectra → pad 7781→7936 → **31 patches of 256 floats**, each with a
     *continuous* position: the patch's mean wavelength, normalized
     `(λ−3000)/7000`.

4. **Per-patch standardization**: each patch is normalized to zero mean, unit
   variance *individually*. This makes the regression target scale-free, but
   it has a consequence worth internalizing: a flat or noise-only patch
   becomes an irreducible N(0,1) target. Data must contain patch-scale
   structure for the loss to be reducible — this is why the synthetic
   fixtures contain blobs/continua rather than pure noise, and why the real
   asinh calibration matters.

5. **Sequence assembly** (`data/packing.py::ObjectSequencer`): modalities are
   wrapped in special tokens, in **alphabetical registry order**:

   ```
   <|bos|> <|begin_images|> p0 … p360 <|end_images|>
           <|begin_spectra|> s0 … s30 <|end_spectra|>
   ```

   397 tokens for a both-modality object, 364 for image-only. The special
   vocabulary is **frozen at 64 ids** (`tokenization.py`): 0 `<|pad|>`,
   1 `<|bos|>`, then 3 ids per modality alphabetically (images 2–4, spectra
   5–7); ids 8–63 are reserved so future modalities never resize the
   embedding. There is no text vocabulary and no lm_head.

6. **Packing** (`PackedCollator`): whole objects (never split) are packed
   greedily into fixed rows of `sequence_length` (4096 → ~10 objects/row);
   the tail is padded. Two invariants the model depends on:
   - `position_ids` restart at 0 for each object and **double as the
     document mask**: the HF model passes `attention_mask=None` and
     transformers builds the block-diagonal causal mask from the restarts;
     nanotron's flash-attention varlen path uses the same signal
     (`_use_doc_masking: true`). Pads get position 0, so each pad is its own
     one-token document, unable to attend to (or be attended by) anything.
   - flattened per-modality `values`/`positions` are concatenated in
     row-major (batch, time) order — exactly the order boolean-mask indexing
     produces — so the model aligns values to slots without index tensors.

## The model

Both implementations share this spec (`modeling_astropt3.py` is the readable
reference):

- **Input assembly**: `embed_tokens(input_ids)` over the 64-id vocab, plus an
  *additive* delta at each modality-placeholder slot:
  `embed(<|m|>) + encoder_m(value) + pos_embed_m(position)`. The placeholder
  embedding acts as a learned modality-type embedding; nothing is overwritten
  in place.
- **Encoders/decoders** are per-modality affine maps (`tokeniser: affine`,
  a single `nn.Linear` each way; an `aim` MLP variant is selectable in
  config). Image positions go through an `nn.Embedding` (index type), spectra
  positions through a small affine layer (continuous type).
- **Body**: the stock SmolLM3 decoder stack, consuming `inputs_embeds`.
  RoPE θ=100k, **NoPE every 4th layer** (`no_rope_layer: 4` /
  `no_rope_layer_interval: 4`), GQA, RMSNorm ε=1e-6, SwiGLU, bf16 training.
- **Loss**: Huber (δ=1.0) computed at positions **one to the left** of each
  modality token — `<|begin_m|>` predicts patch 0, patch *i* predicts patch
  *i+1* (AstroPT's `starts−1` alignment, implemented via `left_shift_mask`).
  Per-modality means are combined by `loss_weight` (both 1.0 for the pilot).
  Special tokens and pads carry no loss.
- **Init**: stock SmolLM3 `_init_weights`, normal(0, 0.02).

### Size family (Pythia-mirrored totals, no vocab head)

Verified by `scripts/count_params.py` (asserts ±10%; all within 1.8%):

| Name | layers | hidden | heads | kv heads | head_dim | intermediate | total |
|------|--------|--------|-------|----------|----------|--------------|-------|
| 70M  | 23 | 512  | 8  | 2 | 64  | 1536  | 70.0M |
| 160M | 25 | 768  | 12 | 4 | 64  | 2048  | 158.3M |
| 410M | 27 | 1024 | 16 | 4 | 64  | 4096  | 411.9M |
| 1B   | 22 | 2048 | 16 | 4 | 128 | 5632  | 994.8M |
| 1.4B | 31 | 2048 | 16 | 4 | 128 | 5632  | 1.401B |
| 2.8B | 36 | 2048 | 16 | 4 | 128 | 11008 | 2.815B (exact SmolLM3-3B body) |
| 6.9B | 38 | 4096 | 32 | 8 | 128 | 11008 | 6.740B |
| 12B  | 42 | 5120 | 40 | 8 | 128 | 14336 | 11.90B |

Modality extras are tiny (~2560×hidden ≈ 1M params at 70M, 13M at 12B); the
small sizes gain a layer or two over Pythia to hit nominal totals.

## Two implementations, one weight source of truth

- **nanotron fork** (`Smith42/nanotron`, branch `astropt3`, git submodule at
  repo root `nanotron/`): the *training* implementation.
  `src/nanotron/models/astropt3.py` is the branch's Qwen2/SmolLM3 stack with
  the vocab-embedding block replaced by the 64-id + modality assembly, and
  the lm_head + sharded-CE loss replaced by modality decoders + masked Huber.
  `run_train.py` gains an `astropt3_streaming` dataset type that calls back
  into this package's `data/nanotron_loader.py`.
- **transformers implementation** (`src/astropt3/`): the release/probing
  artifact and CPU test target. `import astropt3` registers the Auto
  classes, so `AutoModel.from_pretrained(<converted checkpoint>)` works.
- **Converters** (`nanotron/tools/astropt3/convert_{nanotron_to_hf,
  hf_to_nanotron}.py`): every released checkpoint is converted
  nanotron→HF, exactly as SmolLM3 ships. The conversion roundtrip is
  bitwise; *forward* parity between the two stacks is bf16-tolerance only,
  because nanotron applies RoPE at absolute packed-row positions while HF
  restarts per object (RoPE is relative, so attention agrees, but float
  trajectories differ), and flash-attn vs sdpa kernels differ.

The design rule that keeps this manageable: **all modality/packing logic
lives in `astro/`**; the fork only consumes flat micro-batch dicts
(`{m}_values`, `{m}_positions`, `{m}_mask`, `input_ids`, `position_ids` —
flat because nanotron's device mover only transfers top-level tensors).
`nanotron_loader.py` must stay importable without nanotron installed.

## Parallelism semantics

- **PP = 1 everywhere** (asserted). Even the 12B fits without pipeline
  stages (24GB bf16 weights /TP8 + ZeRO-1 optimizer shards); this keeps
  modality tensors on every rank and avoids pipeline complexity.
- **TP**: the transformer body shards as upstream. The tiny modality
  encoders/decoders/position-embedders stay **replicated** across TP ranks,
  riding nanotron's stock tied-parameter mechanism
  (`mark_unsharded_params_as_tied_across_tp`, `reduce_op=None` — identical
  inputs give identical grads by construction; `scripts/tp2_grad_check.py`
  asserts this). This is why **`tp_mode: ALL_REDUCE` is asserted by the
  model**: REDUCE_SCATTER would shard the hidden stream over the sequence
  and break replication.
- **DP + ZeRO-1**: standard data parallelism with optimizer-state sharding.
  Note that the flattened modality tensors have *different shapes on each DP
  rank* (different objects → different patch counts), which is why
  `general.ignore_sanity_checks: true` is required — nanotron's DP
  input-difference check all-gathers tensors assuming equal shapes.
- **Data sharding**: the object stream splits by DP rank
  (`split_dataset_by_node`, identical within a TP group) and then across
  DataLoader workers (HF datasets assigns each worker a disjoint subset of
  the rank's parquet shards).

## Training routine

Pythia-style, adapted to a smaller corpus:

- bf16, fused AdamW (β=0.9/0.95, ε=1e-8), weight decay 0.1 on ≥2D params
  (norms excluded), grad clip 1.0, fp32 gradient accumulation.
- LR: linear warmup min(2000 steps, 1%), cosine decay to 0.1× peak.
  Peak LR by size (Pythia values): 1e-3, 6e-4, 3e-4, 3e-4, 2e-4, 1.6e-4,
  1.2e-4, 1.2e-4 for 70M → 12B.
- GBS 2M tokens (512×4096) at cluster scale; the pilot corpus is ~5.7B
  tokens/epoch, so multi-epoch training is accepted (consistent with
  AstroPTv1 findings).
- **Checkpointing**: `checkpoint_schedule: pythia` saves at steps
  1, 2, 4, …, 512 and then every `checkpoint_interval` (1000) — the log2-
  spaced early checkpoints are what make learning-dynamics studies possible.
  Each checkpoint carries the data-stream position (`dataset_state/`), so a
  resumed run continues the *exact* micro-batch sequence with no replay.
- **Evaluation never blocks training**: `scripts/run_probe_sweep.py` runs in
  a separate process (spare GPU), converting each checkpoint to HF and
  computing (a) a fixed-batch validation loss and (b) a ridge linear probe
  of redshift `Z` from mean-pooled hidden states. Val/train splits are
  **spatially disjoint** (whole order-7 HEALPix tiles hash to one split), so
  near-duplicates cannot leak.

## Roadmap context

Phases 1–4 (package, data pipeline, nanotron fork, checkpoint/eval
machinery) are complete and verified; Phase 5 is the 70M/160M pilots, and
Phase 6 scales up (410M → 12B) and adds time-series (`mmu_tess_spoc`) and
tabular (`mmu_gaia_gaia`) modalities **config-only** via the reserved token
ids. See `PLAN.md` for the full phase log with verification notes.

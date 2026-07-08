# AstroPTv3: SmolLM3 × MMU multimodal pretraining — Implementation Plan

## Context

AstroPTv3 (NAIRR260009) trains an open, from-scratch suite of multimodal
astronomical foundation models (70M–12B, Pythia-mirrored sizes and
checkpointing) by porting the AstroPT approach — autoregressive next-token
**regression** over continuous embeddings of images/spectra — onto the SmolLM3
architecture. This repo (the `astroPTv3` checkout)
is a fork of `huggingface/smollm`. Scope: **architecture + pre-training only**.

Reference implementation: `../astroPT` (branch `multi-gpu-llm`) —
`src/astropt/model.py` has `ModalityConfig`/`ModalityRegistry`,
`Encoder`/`Decoder`, Huber next-token regression, and an LLM-backbone path
using `<|begin_mod|>`/placeholder/`<|end_mod|>` special tokens;
`src/astropt/local_datasets.py` has patchify + collate semantics
(targets shifted to `starts-1`).

## User decisions (fixed)

1. **Training framework: nanotron** (fork of `huggingface/nanotron@smollm3` —
   branch verified to exist, `run_train.py` at root). Parallelism strategy per
   the **Ultra-Scale Playbook**. The **transformers implementation of the model
   is kept as the release/probing artifact** (nanotron→HF conversion, exactly
   how SmolLM3 itself ships).
2. **From scratch** at all sizes; Pythia-style checkpoint schedule.
3. **Minimal special vocab** (64 ids); no natural-language text; no 128k
   lm_head — outputs are per-modality regression decoders.
4. **Pilot data**: `UniverseTBD/mmu_ssl_legacysurvey_north` ×
   `UniverseTBD/mmu_desi_edr_sv3` via lsdb crossmatch (schemas verified below);
   time series + tabular later as config-only extensions.
5. **Affine tokeniser default** (single `nn.Linear` per direction); aim MLP
   selectable via config.
6. **uv** manages environments; new deps in `astro/pyproject.toml`; upgrading
   existing venv packages is fine.
7. **No training runs on this machine.** Development + CPU unit/smoke tests
   here; all GPU work (including nanotron smoke runs) on the training machine.
   Deliver launch scripts + docs, don't execute.

## Verified pilot-data schemas (checked 2026-07-07 via HF datasets-server + MMU builders)

**`UniverseTBD/mmu_ssl_legacysurvey_north`** — 14,174,203 rows, ~4 TB:
- `image` struct: `bands=["des-g","des-r","des-z"]`, `flux` float32 **(3, 152, 152)**
  (builder constant `_image_size=152`, `_pixel_scale=0.262`), `psf_fwhm`, `scale`.
- Scalars: `flux_{g,r,z}`, `fiberflux_{g,r,z}`, `psfdepth_{g,r,z}`, `ebv`, `z_spec`.
- `ra` f64, `dec` f64, `object_id` string, `_healpix_29` int64.
- ⚠ Raw flux (not JPG), 152 px — **not divisible by 16**; patch size must change vs AstroPTv1.

**`UniverseTBD/mmu_desi_edr_sv3`** — 1,126,441 rows, ~86 GB:
- `spectrum` struct of length-**7781** sequences: `flux`, `lambda`, `ivar`,
  `lsf_sigma` (float32), `mask` (bool).
- `Z`, `ZERR`, `ZWARN`, `FLUX_*`/`FIBERFLUX_*` photometry, `EBV`, `ra`, `dec`,
  `object_id`, `_healpix_29`.
- Crossmatched pilot corpus is bounded by DESI: expect ~0.5–1M matched pairs;
  the remaining ~13M image-only objects are still usable as single-modality
  sequences (design requirement below).

## Design

### Two model implementations, one weight source of truth

- **nanotron fork** (`Smith42/nanotron`, branch `astropt3`, forked from
  `huggingface/nanotron@smollm3`): training-time model with TP + ZeRO-1 DP.
  Added as a **git submodule at repo root `nanotron/`**, installed editable via
  a `[train]` extra (needs flash-attn; training machine only).
- **transformers implementation** (`astro/src/astropt3/`): release artifact,
  probing/eval, and CPU tests. `AstroPT3Config(SmolLM3Config)` +
  `AstroPT3Model(SmolLM3PreTrainedModel)`, registered with Auto classes
  (SmolLM3 classes verified present in transformers 4.57.1).
- **Conversion scripts both ways** + tiny-config forward-equivalence test.
  Every released checkpoint is converted nanotron→HF (as SmolLM3 does).

### Repo layout

```
astroPTv3/
├── nanotron/                        # git submodule → Smith42/nanotron@astropt3
│   └── (fork adds:)
│       src/nanotron/models/astropt3.py        # SmolLM3-style body; modality embed/loss blocks
│       src/nanotron/config/astropt3_config.py # modalities section in nanotron YAML config
│       tools/astropt3/convert_{nanotron_to_hf,hf_to_nanotron}.py
├── astro/
│   ├── pyproject.toml               # uv project "astropt3": torch, transformers>=4.57,
│   │                                # datasets>=4.3, einops, pyyaml, pytest, wandb
│   │                                # [data]: lsdb, hats, nested-pandas   (prep env, login node)
│   │                                # [train]: nanotron (editable ../nanotron), flash-attn (training machine)
│   ├── src/astropt3/
│   │   ├── __init__.py              # AutoConfig/AutoModel registration
│   │   ├── configuration_astropt3.py
│   │   ├── modeling_astropt3.py     # HF release/probing model
│   │   ├── modalities.py            # ModalityConfig/Registry, Encoder, Decoder, PositionEmbedder
│   │   ├── tokenization.py          # patchify/unpatchify, normalization, SPECIAL_TOKENS (frozen)
│   │   ├── data/
│   │   │   ├── mmu.py               # MMUIterableDataset (parquet streaming, rank/worker sharding)
│   │   │   ├── packing.py           # ObjectSequencer + PackedCollator (shared by HF & nanotron paths)
│   │   │   ├── nanotron_loader.py   # adapter: PackedCollator batches → nanotron micro-batch dicts
│   │   │   ├── synthetic.py         # network-free fixtures matching the verified MMU schemas
│   │   │   └── transforms.py        # asinh stretch + per-patch standardization
│   │   ├── train_smoke.py           # tiny plain-torch CPU loop (validation only, NOT the trainer)
│   │   └── eval/{val_loss.py,linear_probe.py}
│   ├── configs/
│   │   ├── nanotron/astropt3-{70m,160m,410m,1b,1p4b,2p8b,6p9b,12b}.yaml   # full nanotron configs
│   │   ├── model/…yaml (HF-side mirrors + test-tiny)
│   │   └── data/pilot_images_spectra.yaml
│   ├── scripts/
│   │   ├── prepare_pilot_data.py    # lsdb crossmatch → parquet shards (login node, [data] env)
│   │   ├── compute_norm_stats.py
│   │   ├── count_params.py          # asserts each size within 10% of nominal
│   │   ├── launch_slurm.sbatch      # torchrun → nanotron run_train.py, multi-node
│   │   └── run_probe_sweep.py       # async linear probes over converted HF checkpoints
│   └── tests/                       # CPU-only by default; @pytest.mark.gpu for nanotron parity
└── text/, vision/, tools/           # upstream smollm, untouched (reference)
```

### Model (both implementations share this spec)

Tiny 64-id special-token embedding + SmolLM3 decoder stack (GQA, NoPE every
4th layer, RMSNorm, SwiGLU, doc masking) + per-modality affine
Encoder/Decoder/PositionEmbedder dicts. No lm_head.

- Slot embedding is **additive**: `embed(<|m|>) + encoder_m(value) + pos_m(position)`
  (placeholder = learned modality-type embedding; no in-place overwrite).
- Loss: Huber(delta=1.0) on each modality span, predictions taken one position
  left of each modality token (`<|begin_m|>` predicts patch 0 — astroPT
  `starts-1` semantics), `loss_weight`-weighted mean over modalities. No loss
  on special/pad tokens (`special_token_ce_weight` hook kept for later
  variable-length modalities).
- Positions: SmolLM3 RoPE/NoPE unchanged over the flat packed sequence with
  per-object-reset `position_ids` (doubles as nanotron's doc-masking signal for
  FA varlen); per-modality embeddings added at input.
- Init: stock SmolLM3 `_init_weights` (normal 0.02).

Special vocab (frozen in `tokenization.py`): `0 <|pad|>`, `1 <|bos|>`, then 3
ids per modality alphabetically: images 2–4, spectra 5–7; 8–63 reserved so new
modalities never resize the embedding.

### Modality tokenization (pinned to verified schemas)

- **Images**: `image.flux` (3,152,152) float32 → asinh stretch → einops
  `"c (h p1) (w p2) -> (h w) (p1 p2 c)"` with **patch 8** → **361 tokens** of
  `input_size=192`; per-patch standardization; integer patch-index positions
  (spiral option ported). Patch 8 chosen because 152 = 8×19 (16 doesn't divide
  152) and higher tokens/object stretches the limited corpus.
- **Spectra**: `spectrum.flux` (7781) → pad to 7936 → **31 patches** of
  `input_size=256`; per-patch standardization; position = per-patch mean
  `lambda` normalized `(λ-3000)/7000` through a small affine PositionEmbedder
  (`pos_type="continuous"`). `mask==True` bins zeroed before patching.
  Option (not pilot-default): ivar-weighted Huber.
- **Missing modalities are allowed**: `ObjectSequencer` emits only the
  modalities present, so image-only objects (~13M) train alongside ~1M
  image+spectra pairs.
- Per object: `<|bos|> <|begin_images|> ×361 <|end_images|> <|begin_spectra|> ×31 <|end_spectra|>`
  = 397 tokens (image-only: 364). Greedy packing into seq 4096 (~10
  objects/seq), tail padded, pad excluded from loss/attention.

### Model-size table (named size ≈ TOTAL params; no vocab head)

GQA, NoPE interval 4, seq 4096. With the affine default, modality extras are
~2560×hidden (≈1M at 70M → ≈13M at 12B); small sizes gain a layer or two to hit
nominal totals — finalized in Phase 1 by `count_params.py` (±10% assert).

FINAL (Phase 1, verified by count_params.py on meta device; all within 1.8% of nominal):

| Name | layers | hidden | heads | kv | head_dim | inter | total (measured) |
|------|--------|--------|-------|----|----------|-------|------------------|
| 70M  | 23 | 512  | 8  | 2 | 64  | 1536  | 70.04M (+0.1%) |
| 160M | 25 | 768  | 12 | 4 | 64  | 2048  | 158.3M (−1.0%) |
| 410M | 27 | 1024 | 16 | 4 | 64  | 4096  | 411.9M (+0.5%) |
| 1B   | 22 | 2048 | 16 | 4 | 128 | 5632  | 994.8M (−1.5%) |
| 1.4B | 31 | 2048 | 16 | 4 | 128 | 5632  | 1.401B (−0.7%) |
| 2.8B | 36 | 2048 | 16 | 4 | 128 | 11008 | 2.815B (+0.5%, exact SmolLM3-3B body) |
| 6.9B | 38 | 4096 | 32 | 8 | 128 | 11008 | 6.740B (−1.8%) |
| 12B  | 42 | 5120 | 40 | 8 | 128 | 14336 | 11.90B (+0.4%) |

### Nanotron fork surgery (branch `astropt3`)

1. `src/nanotron/models/astropt3.py`: copy the branch's SmolLM3 (qwen2-style)
   model; replace the vocab `TensorParallelEmbedding` block with the 64-id
   embedding + modality-encoder assembly; replace the lm_head
   `TensorParallelColumnLinear` + sharded-CE `Loss` block with per-modality
   affine decoders + masked Huber loss block.
2. **PP=1 everywhere** (playbook: don't take on pipeline complexity unless
   memory forces it — it doesn't, see recipe table). This means modality
   tensors never cross pipeline stages; the whole batch dict goes to every rank.
3. **TP**: transformer body sharded as upstream; modality encoders/decoders are
   tiny affine layers kept **replicated** across TP ranks (identical inputs →
   identical grads; verify nanotron's `NanotronParameter` reduce semantics for
   replicated modules in the tiny parity test — this is the one subtle bit).
4. Config: `modalities` section added to the nanotron model config dataclass;
   our YAMLs otherwise mirror `text/pretraining/smollm3/stage1_8T.yaml`
   conventions (`_use_doc_masking: true`, etc.).
5. Dataloader: new dataset type `astropt3_streaming` wired into
   `run_train.py`'s `get_dataloader`, implemented by
   `astro/src/astropt3/data/nanotron_loader.py`: MMUIterableDataset →
   ObjectSequencer → PackedCollator → micro-batch dicts
   (`input_ids`, modality values/masks/positions, `position_ids`,
   `label_*` targets), sharded by **DP rank** (`split_dataset_by_node`),
   identical stream within a TP group.
6. Checkpointing: patch the trainer's interval check with
   `should_checkpoint(step)` — Pythia schedule {1,2,4,…,512} then every 1000
   steps (~2B tokens at GBS 2M) + final. Save `datasets` stateful-iterable
   `state_dict()` alongside nanotron's checkpoint for resume.
7. Conversion: `tools/astropt3/convert_nanotron_to_hf.py` (backbone weight map
   + modality modules), modeled on the fork's existing llama converters.

### Parallelism recipe (Ultra-Scale Playbook; H100 80GB, seq 4096, GBS 2M tokens = 512×4096)

| Size | GPUs (grant) | TP | DP | ZeRO | Activation recompute | Notes |
|------|--------------|----|----|------|----------------------|-------|
| 70M–410M | 32 | 1 | 32 | 1 | none | max micro-batch that fits |
| 1B–1.4B  | 64 | 1 | 64 | 1 | selective | |
| 2.8B     | 64 | 2 | 32 | 1 | selective | |
| 6.9B     | 128 | 4 | 32 | 1 | selective/full | |
| 12B      | 256 | 8 | 32 | 1 | full | TP inside NVLink node; **no PP** (24GB bf16 weights /8 + ZeRO-1 optim shards fit comfortably) |

Playbook practices baked into the plan: TP never crosses the node boundary;
prefer ZeRO-1 + activation recomputation before adding PP; benchmark
micro-batch size and grad-accum to hit GBS; profile the first ~100 steps
(torch profiler) before committing node-hours; track tokens/s/GPU + MFU
(target ~30–45% depending on size); FA2 varlen doc masking via
position_ids; bf16 + fused AdamW (β 0.9/0.95, wd 0.1 on ≥2D params), grad clip
1.0, cosine to 0.1× peak, warmup min(2000 steps, 1%).
Peak LR by size (Pythia): 1e-3 / 6e-4 / 3e-4 / 3e-4 / 2e-4 / 1.6e-4 / 1.2e-4 / 1.2e-4.
Corpus is small vs Pythia's 300B tokens (~5.7B tokens/epoch pilot); multi-epoch
training is accepted (per AstroPTv1 findings); token budget per size set at
launch time in the YAML.

### Data pipeline

- **Prep (login node, `[data]` env)**: `prepare_pilot_data.py` —
  `lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north")`
  LEFT-crossmatch `mmu_desi_edr_sv3` (≤1″): all images kept, spectra attached
  where matched → ~256MB parquet shards at
  `$ASTROPT3_DATA_ROOT/{train,val}/` (default `../astroPTv3_data/pilot_v1`
  beside the repo) + provenance json.
  Fallback if lsdb won't install: HF streaming + homemade HEALPix join (same
  output schema).
- **Train/val split by coarse HEALPix pixel** (from `_healpix_29`) — spatially
  disjoint, no near-duplicate leakage.
- Training streams the local shards with `load_dataset("parquet", ...,
  streaming=True)` + `HF_DATASETS_OFFLINE=1` — no network/lsdb on compute nodes.
- `compute_norm_stats.py` (100k sample) → asinh scale + normalization stats
  into the data YAML; verify flux histograms before/after stretch.
- `synthetic.py` generates records matching the **verified schemas** above —
  all tests and the CPU smoke loop run networkless.

## Phases (each a reviewable PR)

### Phase 1 — `astro/` package: modalities, tokenization, packing, HF model
Create `astro/` scaffold (uv), port from `../astroPT/src/astropt/model.py`
(ModalityConfig/Registry l.41-71; Encoder/Decoder l.272-315 — affine default)
and `local_datasets.py` (patchify/spiralise); implement `tokenization.py`
against the verified MMU shapes, `packing.py`, `synthetic.py`,
`configuration_astropt3.py` + `modeling_astropt3.py`, size YAMLs,
`count_params.py`, `train_smoke.py`.
**Verify (this machine, CPU)**: pytest green, no network — patchify↔unpatchify
roundtrip exact on (3,152,152) and (7781,) fixtures; packing never splits an
object and handles image-only objects; masked positions contribute zero loss;
doc mask blocks cross-object attention (perturb object A → object B hidden
states identical); target alignment on a hand-built example; tiny-model
forward/backward finite with grads on every param; `save_pretrained` →
`AutoModel.from_pretrained` → identical outputs; `count_params.py` ±10% per
size; 50-step CPU smoke on synthetic: loss < 0.7× initial.

### Phase 2 — Pilot data prep + streaming dataset
`prepare_pilot_data.py`, `compute_norm_stats.py`, `data/mmu.py`,
`configs/data/pilot_images_spectra.yaml`.
**Verify**: crossmatch logs matched/unmatched counts (expect ~0.5–1M matched,
~13M image-only); decoded-object sanity print (patch stats ~N(0,1) after
stretch, λ range 3600–9824Å); dataloader-only throughput ≥2× training
consumption; 2 ranks × 2 workers yield disjoint object_ids. (lsdb runs on the
login node here; nothing GPU.)

### Phase 3 — Nanotron fork: model, config, dataloader, conversion
Fork `huggingface/nanotron@smollm3` → `Smith42/nanotron@astropt3`; submodule at
`nanotron/`; implement fork items 1–7 above; `nanotron_loader.py`;
`convert_nanotron_to_hf.py` (+ reverse); nanotron YAMLs for test-tiny + 70M.
**Verify (training machine, 1 GPU, gpu-marked tests)**: tiny-config parity —
convert HF↔nanotron and match forward losses on a fixed synthetic batch to
bf16 tolerance; replicated-module gradient check across TP=2; 50-step nanotron
run on synthetic data: loss decreases; conversion of that checkpoint loads via
`AutoModel.from_pretrained` and reproduces val loss.

DONE (verified on a shared A100 node, GPU-pinned, tiny configs only). Fork
items 1–5 + 7 delivered; item 6 (Pythia checkpoint schedule) is Phase 4 per
the phase split. Notes:
- Modality encoders/decoders/pos-embedders ride nanotron's stock
  `mark_unsharded_params_as_tied_across_tp` (replicated across TP,
  `reduce_op=None` under ALL_REDUCE — "synced by design", asserted identical
  grads in `astro/scripts/tp2_grad_check.py`). `tp_mode: ALL_REDUCE` is
  asserted by the model: REDUCE_SCATTER shards the hidden stream over the
  sequence and would break replication (revisit only if throughput demands).
- nanotron applies RoPE at absolute row positions in the packed row; HF
  restarts per object. RoPE is relative so attention agrees — parity is
  therefore bf16-tolerance, not bitwise (conversion roundtrip IS bitwise).
- GPU env recipe (no prebuilt flash-attn for torch 2.12/cu13 yet):
  `uv venv --python 3.13; uv pip install torch==2.8.0
  <flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp313 wheel from GitHub>
  -e nanotron -e astro psutil` then `pytest -m gpu astro/tests/test_nanotron_gpu.py`.
- Fork carries 4 small compat/bug fixes vs upstream (functorch tree_map,
  flash-attn 2.8 rotary signature, unbound `tied_name` in weight-decay
  exclusion, consumption-stats hasattr guards) + relaxed numpy pin, psutil dep.
- `Smith42/nanotron` fork + `astropt3` branch exist locally in `nanotron/`;
  pushing to GitHub needs credentials (create the fork, then
  `git -C nanotron push origin astropt3` and push the recorded gitlink).

### Phase 4 — Checkpoint schedule, resume, eval hooks
Pythia `should_checkpoint` patch + dataset-state save; `eval/val_loss.py`
(fixed 512 val batches per eval interval) + `eval/linear_probe.py` +
`run_probe_sweep.py` (async over converted HF checkpoints — never blocks
training; ridge probe → redshift `Z`, which the pilot data carries natively).
**Verify (training machine)**: kill-at-step-137/resume overlays the
uninterrupted loss curve without sample replay (object_id hash log);
checkpoint dirs at exactly steps 1,2,4,…,512,1000,…; each converts and loads.

DONE (verified on the reserved A100s, tiny synthetic configs,
2026-07-08; gpu gates in `tests/test_phase4_gpu.py`). Notes:
- Schedule: `checkpoints.checkpoint_schedule: pythia` (new optional
  `CheckpointsArgs` field) = powers of two ≤ 512 ∪ multiples of
  `checkpoint_interval`; canonical implementation in
  `astropt3/checkpoint_schedule.py`, lazy-imported by the fork's trainer.
  1000-step run produced dirs at exactly 1,2,4,…,512,1000; all 11 convert
  and load via `AutoModel.from_pretrained`.
- Stream state: `PackedMicroBatches.state_dict()` = position at the START
  of the current partial packing row (partial rows are untrained, so resume
  re-draws them — no tensor serialization, exact continuation). Saved per
  DP rank to `{ckpt}/dataset_state/dp_{rank}.pt` before `latest.txt`;
  loaded in `run_train.py` from `trainer.init_checkpoint_path`. Requires
  `num_loading_workers: 0` (else `state_dict()` is None and the trainer
  skips it — revisit with torchdata StatefulDataLoader if Phase 5
  throughput demands workers). MMU resume is exact for
  `shuffle_buffer_size: 0`; with a buffer it skips at most the in-flight
  buffer and never replays a trained record (HF shuffle semantics).
- Kill/resume gate: checkpoint at 137, SIGKILL a few steps later, resume →
  the step-138 loss (pure forward on restored state) matches the killed
  run's own value at log precision (0.0791 = 0.0791) and tracks the
  independent uninterrupted run to 0.7% mean over steps 138–200;
  `object_id_log` (new dataset arg, yield-time append) proves the resumed
  stream is exactly the uninterrupted tail, no replay. Note: compare a
  resumed run against its OWN trajectory for tight tolerances — two
  independent runs drift a few % by step 137 (nondeterministic backward).
- 2026-07-08 (later): rebased the fork commit onto `origin/astropt3` after
  the user merged upstream `main` there (fork `main` now exists,
  PR #1 = astropt3). The merge required two more compat fixes: moe.py must
  not require `grouped_gemm` at import time (it is imported for dense
  models via scaling.parametrization), and the flash-attn ≥ 2.8 rotary
  signature fix from 179c18f0 had to be restored (merge took upstream's
  rotary.py wholesale). Full GPU suite re-verified post-rebase.
- Fork also needed: per-stage `consumed_tokens_per_dataset_folder` kept in
  sync by hand for astropt3 streams, else `TrainingMetadata`'s
  consumed-tokens invariant fails on checkpoint LOAD (upstream only updates
  it for BlendableDataset).
- Eval is fully outside the trainer: `run_probe_sweep.py` polls a run dir
  (gated on `latest.txt`), converts, then `eval/val_loss.py` (fixed
  deterministic batches; synthetic val = record indices ≥ 10M) and
  `eval/linear_probe.py` (closed-form ridge, inner-split λ, test R²).
  Tiny-run sweep: val loss 0.456→0.051 monotone, images/spectra within 2×,
  redshift probe R² 0.42→0.79 across checkpoints.

### Phase 5 — Slurm launch + 70M/160M pilots
`launch_slurm.sbatch` (torchrun → `nanotron/run_train.py --config-file
astro/configs/nanotron/astropt3-70m.yaml`), per-size YAMLs from the recipe
table, profiling run instructions.
**Verify (training machine)**: 100-step dry run at target node count first
(playbook rule) with tokens/s/GPU + MFU logged; then 70M and 160M pilots to
completion: monotone-ish val loss per modality; image/spectra losses within
~5× after warmup (else tune loss_weight); probe R² for redshift improves
across checkpoints.

IN PROGRESS (2026-07-08, dev node, user's GPU reservation): real-data
70M shakeout ahead of the cluster pilots.
- Pilot data prep RUNS ON THE DEV NODE too (network confirmed): the full
  `prepare_pilot_data.py` run is filling `../astroPTv3_data/pilot_v1`
  (5,596 partitions, ~75 obj/s ≈ multi-day; journalled, resume any time,
  any machine). A finished 2°-cone prep around the DESI SV3 rosette
  (217.97, 32.62) sits in `../astroPTv3_data/pilot_sv3cone`: 26,452
  objects, 7,354 with spectra (27.8%) — spectra-rich subset for probing.
  NEVER point a cone run at the canonical dir: cone partitions are
  row-filtered and would poison the resume journal.
- `compute_norm_stats.py` ran on 10k real images → asinh p1/p99 now in
  the data yaml (provisional: day-one sky region; re-check when the full
  corpus lands). `check_pilot_data.py`: real images decode to exact
  N(0,1) patches, spectra to 31 patches λ 3702–9784 Å; dataloader ~1,000
  obj/s ≈ 400k tok/s per process at 8 workers (≥2× gate passes at DP=2).
- Blocker found+fixed by the first 70M execution: upstream nanotron's
  DDP + fp32-accum + ZeRO-1 comm hook routes to a dead reduce-scatter
  branch (NotImplementedError) — every ZeRO-1 DDP run would have crashed
  at step 1. Fork fix: all-reduce path (nanotron commit 0668f369).
- 100-step DP=2 dry run (astropt3-70m-shakeout.yaml): loss 0.459→0.259,
  ~240k tok/s total, 123 model TFLOPs/GPU (~39% MFU), peak 31.7 GiB/80.
  20k-step run (2.6B tokens, Pythia checkpoints) + async probe sweep
  launched on the reserved pair.

### Phase 6 — Scale-up + modality extension
410M → 1.4B (TP=1), then 2.8B/6.9B/12B per the recipe table (dry run before
each). Add time series (`mmu_tess_spoc`) and tabular scalars (`mmu_gaia_gaia`)
**config-only** via reserved token ids + registry entries; add a config-only
test proving no model-code change was needed.
**Scientific sanity check**: loss-vs-tokens curves order correctly across
sizes (bigger = lower at matched tokens).

## Risks

- **Nanotron surgery** is the deep end: embedding/loss pipeline blocks, TP
  semantics for replicated modality modules, custom dataset type. Mitigated by
  PP=1, the tiny HF↔nanotron parity test, and keeping all modality/packing
  logic in the shared `astro` package (nanotron only consumes batch dicts).
- **lsdb not installed anywhere yet**; only the login-node prep env needs it;
  HEALPix-join fallback specified.
- **Pilot corpus is small** (~5.7B tokens/epoch): fine for 70M–410M; larger
  sizes need the Phase 6 modality/survey extensions or many epochs — flag at
  the Month-3 scope checkpoint.
- **flash-attn / nanotron env** exists only on the training machine; all
  gpu-marked tests deferred there; CPU tests must stay green here.
- **BeeGFS checkpoint pressure**: prune non-schedule intermediates; convert +
  upload scheduled checkpoints to HF hub as they land.
- **Streaming-resume fidelity**: pin `datasets`, record the pin in checkpoint
  metadata.

## Reference files

- `../astroPT/src/astropt/model.py` — ModalityConfig/Encoder/Decoder/optimizer groups to port
- `../astroPT/src/astropt/local_datasets.py` — patchify, sequence structure, collate semantics
- `text/pretraining/smollm3/stage1_8T.yaml` — SmolLM3-3B nanotron config conventions
- `vision/m4/models/vllama3/modeling_vllama3.py` — in-fork placeholder-scatter reference
- transformers 4.57.1 `models/smollm3/` — SmolLM3Config/Model (verified present)
- Ultra-Scale Playbook: https://huggingface.co/spaces/nanotron/ultrascale-playbook

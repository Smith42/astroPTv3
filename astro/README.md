# AstroPT3

SmolLM3-architecture multimodal astronomical foundation models, pretrained
from scratch on the [Multimodal Universe](https://huggingface.co/collections/UniverseTBD/multimodal-universe-hats)
with AstroPT-style continuous-token regression (NAIRR260009).

- **Architecture**: SmolLM3 decoder body (GQA, NoPE every 4th layer) with a
  64-id special-token vocabulary; images/spectra enter as affine-projected
  continuous patch tokens and leave through per-modality regression decoders
  (Huber next-token loss). See `src/astropt3/modeling_astropt3.py`.
- **Training** runs on the nanotron fork (git submodule at `../nanotron`,
  Phase 3+); the transformers implementation here is the release/probing
  artifact and the CPU test target.
- **Pilot data**: `UniverseTBD/mmu_ssl_legacysurvey_north` (3×152×152 flux
  cubes → 361 patch-8 tokens) × `UniverseTBD/mmu_desi_edr_sv3` (7781-bin
  spectra → 31 patch-256 tokens), lsdb-crossmatched offline.

## Setup

```bash
cd astro
uv sync --extra dev          # CPU-safe: model, packing, tests
uv sync --extra data         # + lsdb/hats (login-node data prep only)
uv sync --extra train        # + nanotron/flash-attn (training machine only)
```

## Develop / verify (no GPU, no network)

```bash
uv run pytest                          # unit tests (gpu-marked tests excluded)
uv run python scripts/count_params.py  # size table, ±10% assert
uv run python -m astropt3.train_smoke --config configs/model/test-tiny.yaml \
    --steps 50 --assert-decrease       # end-to-end CPU smoke on synthetic data
```

Model-size configs live in `configs/model/` (Pythia-mirrored 70M–12B).
The implementation plan (phases, verification, parallelism recipes) is
[`PLAN.md`](PLAN.md); per-phase PRs land on feature branches.

**Docs**: [`docs/architecture.md`](docs/architecture.md) — what the model is
and why (data→tokens→model, size family, parallelism semantics);
[`docs/training.md`](docs/training.md) — how to run it (environments, data
prep, launching, checkpoint/resume, eval, troubleshooting).

## Pilot data (login node, network)

Both MMU sources are HATS collections with margin caches, so the crossmatch
is a single lsdb call. Prepare once, then everything downstream is offline:

```bash
uv run --extra data python scripts/prepare_pilot_data.py
    # LEFT-crossmatch (1", nearest) → ~256MB parquet shards under
    # {root}/{train,val}/ + provenance.json; resumable per partition;
    # logs matched/image-only counts. Smoke: --cone RA DEC RADIUS_ARCSEC
uv run python scripts/check_pilot_data.py --target-tokens-per-sec N
    # decoded-object sanity (~N(0,1) patches, λ range) + dataloader
    # throughput bench (want ≥2× training consumption)
```

Image normalization is physical (band-registry-keyed rescale → bright-pixel
clamp → arcsinh; `data/band_registry.py`), so there is no per-corpus
calibration step.

Training streams the shards with `astropt3.data.mmu.MMUIterableDataset`
(`HF_DATASETS_OFFLINE=1`; DP-rank and DataLoader-worker sharded; keep
`num_workers ≤ n_shards / world_size`).

## Training (nanotron fork) + async eval

Pretraining runs on the `Smith42/nanotron` fork (submodule at `../nanotron`;
configs in `configs/nanotron/`). Smoke run:

```bash
cd .. && CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --nproc_per_node=1 nanotron/run_train.py \
    --config-file astro/configs/nanotron/astropt3-test-tiny.yaml
```

- **Checkpoints**: `checkpoints.checkpoint_schedule: pythia` saves at steps
  1,2,4,…,512 plus every `checkpoint_interval` (schedule source:
  `src/astropt3/checkpoint_schedule.py`). Each checkpoint carries the data
  stream position (`dataset_state/dp_{rank}.pt`), so setting
  `checkpoints.resume_checkpoint_path` resumes the exact micro-batch
  sequence — no sample replay, no gap (requires `num_loading_workers: 0`;
  set `object_id_log` on the dataset to audit this).
- **Eval never blocks training** — run the sweep beside it on a spare GPU:

```bash
python astro/scripts/run_probe_sweep.py \
  --checkpoints-dir <run_ckpt_dir> --out-dir <eval_dir> \
  --data-root <val_shards|synthetic> --watch --until-step <train_steps>
    # per checkpoint: convert to HF -> fixed-batch val loss
    # (astropt3.eval.val_loss) -> ridge redshift probe
    # (astropt3.eval.linear_probe) -> one line in probe_results.jsonl
```

GPU-marked tests (training machine / reserved GPU):
`pytest -m gpu tests/test_nanotron_gpu.py tests/test_phase4_gpu.py`.

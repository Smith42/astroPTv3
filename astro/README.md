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
uv sync --extra data         # + lsdb (match-index build only; not needed to train)
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

## Pilot data (streamed, ADR 0006)

The corpus is **streamed live from the HF hub at train time** — there is no
prep step and no local copy. Three HATS sources are interleaved per
record: images-only (~14M LegacySurvey), spectra-only (~1.1M DESI EDR SV3),
and their 1" inner crossmatch, at provisional weights 0.60/0.15/0.25.

Reads are `hats` (partition enumeration) + `pyarrow` (row groups); lsdb runs
only offline, to build the match-index that the pairs source joins on:

```bash
uv run --extra data python scripts/build_match_index.py --out match_index.parquet
    # ~200 crossmatch partitions, ~1h; ids only (tens of MB, no pixels copied)
uv run pytest tests/test_streaming.py    # cursor logic offline + one live check
```

Without a match index there is no pairs source and the corpus degrades to
images + spectra — visible in the logs rather than silent.

`data_root` is `synthetic` (tests, smoke) or `mmu` (real training); a path
to the retired local corpus raises. Partitions are addressed by index, so
resume skips without downloading and replays nothing; the whole stream
state is a handful of ints. Val reserves whole HEALPix partitions, so
train/val stay spatially disjoint.

Image normalization is physical (band-registry-keyed rescale → bright-pixel
clamp → arcsinh; `data/band_registry.py`), so there is no per-corpus
calibration step.

Network is a hard training dependency: hub downtime stalls training with no
local fallback. That is the deliberate trade for dropping the reshard.

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

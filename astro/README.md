# AstroPT3

SmolLM3-architecture multimodal astronomical foundation models, pretrained
from scratch on the [Multimodal Universe](https://huggingface.co/collections/UniverseTBD/multimodal-universe-hats)
with AstroPT-style continuous-token regression (NAIRR260009).

- **Architecture**: SmolLM3 decoder body (GQA, NoPE every 4th layer) with a
  64-id special-token vocabulary; images/spectra enter as affine-projected
  continuous patch tokens and leave through per-modality regression decoders
  (Huber next-token loss). See `src/astropt3/modeling_astropt3.py`.
- **Training** runs on the nanotron fork (git submodule at `../nanotron`);
  the transformers implementation here is the release/probing artifact and
  the CPU test target.
- **Pilot data (ADR 0015, experimental)**: the uncrossmatched
  `UniverseTBD/mmu_ssl_legacysurvey_north` HATS catalog, streamed through
  LSDB's `InfiniteStream` — image only, with the image-side catalog scalars.

## Setup

```bash
cd astro
uv sync --extra dev          # CPU-safe: model, packing, tests, lsdb
uv sync --extra train        # + nanotron/flash-attn (any GPU box, this one included)
```

## Develop / verify (CPU, no network)

```bash
uv run pytest                          # unit tests (gpu-marked tests excluded)
uv run python scripts/count_params.py  # size table, ±10% assert
```

The synthetic-dependent `train_smoke` gate is **temporarily gone** under the
ADR 0015 cutover; this branch is experimental and is not declared complete
against the old repo contract until an LSDB-backed smoke exists (ADR 0015
§Interim evidence). Offline decode/sequence/pack fixtures live in
`tests/legacy_fixture.py`; one bounded `pytest.mark.network` test opens the
live catalog (`tests/test_lsdb_stream.py`).

Model-size configs live in `configs/model/` (Pythia-mirrored 70M–12B).
The implementation plan (phases, verification, parallelism recipes) is
[`PLAN.md`](PLAN.md).

**Docs**: [`docs/architecture.md`](docs/architecture.md) — what the model is
and why; [`docs/training.md`](docs/training.md) — how to run it.

## Training data (LSDB InfiniteStream, ADR 0015)

The training-record path is:

```
lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north")
→ lsdb.streams.InfiniteStream(client=None, partitions_per_chunk=1, seed)
→ decode rows → ObjectSequencer → greedy packing → torch DataLoader → nanotron
```

- **One catalog, no crossmatch.** Image rows plus the image-side scalars the
  active registry uses. The full multimodal model shape is retained so later
  crossmatches don't need an architecture migration.
- **Concurrency is the DataLoader's.** LSDB runs synchronously with
  `client=None`; keep `num_loading_workers` at the configured 6–8 per rank.
  Each consumer derives its stream seed from `(run seed, dp rank, worker id,
  retry generation)`. Overlap between consumers, repeats, and incomplete
  coverage are accepted.
- **No dataset checkpoint state.** Resume restores model/optimizer/scheduler
  state and starts a fresh stream. There is no cursor and no replay audit.
- **Failure policy.** Recognized transport/storage errors reopen a fresh
  stream under bounded exponential backoff; decode/validation and unknown
  errors fail immediately.
- **Floating revisions.** LSDB and the catalog float to latest; `uv.lock`
  records the resolved package graph and runs log the resolved LSDB version.
  Worker memory is the known risk: `InfiniteStream` can retain ~2 whole
  pandas partitions, so lower `num_loading_workers` on an observed OOM.

Evaluation (validation loss, probes, sample sweeps) is deferred (ADR 0015
§Evaluation and generation): `astropt3.eval` keeps only the pure model-side
functions consuming provided records/objects/batches; the source-driven
CLIs, `run_probe_sweep.py`, and the eval co-launch hooks are removed.

Image normalization is physical (band-registry-keyed rescale → bright-pixel
clamp → arcsinh; `data/band_registry.py`), so there is no per-corpus
calibration step.

## Training (nanotron fork)

```bash
cd .. && CUDA_DEVICE_MAX_CONNECTIONS=1 \
  torchrun --nproc_per_node=1 nanotron/run_train.py \
    --config-file astro/configs/nanotron/<config>.yaml
```

All run configs use `is_astropt3_streaming: true` — no dataset block knobs
beyond `ar_replicas`/`replica_placement`. Checkpoints store weights
(bf16), optimizer + LR-scheduler state, RNG states, and `model_config.json`
under the Pythia schedule (`checkpoint_schedule.py`); `latest.txt` marks the
last complete checkpoint.

GPU-marked tests (any reserved GPU, this box included):
`pytest -m gpu tests/test_nanotron_gpu.py`.

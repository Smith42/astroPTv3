# Training AstroPTv3 models with nanotron

*The operational guide: environments, data, launching, checkpoint/resume.
Background on the model is in [`architecture.md`](architecture.md). The
current record-source decision is
[`adr/0015-lsdb-infinite-stream-training.md`](adr/0015-lsdb-infinite-stream-training.md)
— **experimental/incomplete**: the old contract's `train_smoke` gate is
absent on this branch.*

## 1. Environments

| env | where | contents | used for |
|-----|-------|----------|----------|
| `uv sync --extra dev` | anywhere | torch (CPU ok), transformers, lsdb | unit tests |
| GPU venv | training machine | torch + **flash-attn** + nanotron (editable `nanotron/`) + astro (editable `astro/`) + psutil | training, GPU tests, conversion |

flash-attn wheels are the constraint for the GPU venv: pick a torch version
with a prebuilt wheel for your CUDA (never compile it on a shared box).

```bash
uv venv gpuenv --python 3.13
uv pip install torch==2.8.0 \
  <flash_attn wheel from GitHub releases> \
  -e nanotron -e astro psutil
```

Verify the env on a GPU box: `pytest -m gpu tests/test_nanotron_gpu.py`.

## 2. Data (LSDB InfiniteStream — ADR 0015)

There is **no prep step** and no local corpus. Each DataLoader worker opens

```
lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north", columns=<active registry projection>)
→ lsdb.streams.InfiniteStream(client=None, partitions_per_chunk=1, seed)
```

decodes rows, sequences/packs them, and feeds nanotron through a plain
`DataLoader(batch_size=None)`.

Facts to know:

- **Network is a hard dependency.** Hub downtime stalls training with no
  local fallback.
- **Image only, uncrossmatched.** Records carry the image cube and the
  catalog's image-side scalars (`ebv`, `flux_g/r/z`, optionally fiber/psf
  fields when active). The multimodal model shape is retained so future
  crossmatches don't need an architecture migration; spectra stay allocated
  but untrained (rates, losses read as `spectra: 0.0`).
- **Concurrency is the DataLoader's.** LSDB runs synchronously
  (`client=None`) — do not create a Dask client. `num_loading_workers` is
  6–8 per rank. Seeds are `(run seed, dp rank, worker id, retry generation)`.
- **No cursors, no coverage guarantee.** Overlap between consumers,
  repeated records, and revisits after retries/resumes are accepted.
- **Memory risk.** `InfiniteStream` can hold ~2 whole pandas partitions
  (~1.5 GiB) per worker. If RSS overflows, lower `num_loading_workers` —
  this is runbook, not a preflight gate.
- **Floating revisions.** LSDB and the catalog float; `uv.lock` records
  what's installed in a checkout, and `[data]` startup lines log the
  resolved LSDB version per worker.

Streaming specifics:

- **Resume is fresh.** Checkpoints restore weights/optimizer/scheduler/RNG;
  the record stream is cursorless and restarts on resume. Old checkpoint
  `dataset_state/` files are ignored.
- **Transient failures rebuild.** Recognized transport/storage errors
  (HTTP, OS, the hub's closed-client RuntimeError) discard the iterator and
  reopen with bounded exponential backoff (60 retries, ≤120 s cap,
  `gc.collect()` before reopening). Decode/validation/unknown errors fail
  immediately.

## 3. Configs

Full nanotron run configs live in `astro/configs/nanotron/`. The dataset
block is minimal:

```yaml
data_stages:
- data:
    dataset:
      is_astropt3_streaming: true   # the only source (ADR 0015)
      # ar_replicas: 1              # optional: distinct AR factorisations
      # replica_placement: decorrelated
    num_loading_workers: 8          # concurrent LSDB readers per DP rank
    seed: 42
```

Everything else — catalog, columns, stream policy — is fixed in
`astropt3.data.nanotron_loader`. The old knobs (`data_root`,
`match_index`, synthetic fractions, `object_id_log`) are gone.

Governing knobs, top to bottom:

```yaml
general:
  ignore_sanity_checks: true   # REQUIRED with DP>1: per-rank modality tensor
                               # shapes differ; the DP input check would crash
model:
  model_config:
    is_astropt3_config: true   # dispatches to AstroPT3ForTraining
    _use_doc_masking: true     # position_ids restarts = document boundaries
parallelism:
  pp: 1                        # asserted — do not change
  tp_mode: ALL_REDUCE          # modality modules are TP-replicated
  dp: <n>
tokens:
  sequence_length: 4096
  micro_batch_size: 16
  batch_accumulation_per_replica: 1
checkpoints:
  checkpoint_schedule: pythia  # steps 1,2,4,...,512 + every interval
  checkpoint_interval: 1000
  resume_checkpoint_path: null # set to the checkpoints dir to resume
```

Per-size recipes are unchanged (the PLAN table in
[`docs/adr/0014-byte-efficiency-and-mfu-programme.md`](adr/0014-byte-efficiency-and-mfu-programme.md)
and `architecture.md`).

## 4. Launching

Single node (from the repo root):

```bash
CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=<gpus> --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    nanotron/run_train.py --config-file astro/configs/nanotron/<config>.yaml
```

Multi-node via slurm:

```bash
sbatch --nodes=<N> astro/scripts/launch_slurm.sbatch astro/configs/nanotron/<config>.yaml
# dry run: sbatch --nodes=<N> --export=ALL,DRY_RUN_STEPS=100 astro/scripts/launch_slurm.sbatch <config>
```

The launcher sources `$ASTROPT3_ENV` (default `../astroPTv3_gpuenv`) and
rendezvous on the first node. **Always dry-run first**, checking
tokens/s/GPU, MFU, and memory, before committing a real run.

**Notes after the ADR 0015 cutover:**

- `HF_DATASETS_OFFLINE=1` must NOT be set — the stream needs the hub.
- The eval sidecar (`run_probe_sweep.py`) and the `EVAL_GPU` co-launch hook
  are removed; §Evaluation is deferred to a future LSDB-backed seam.

## 5. Checkpoints and resume

Checkpoint dirs `{1,2,4,...,512,1000,...}` hold model weights (bf16),
optimizer + LR-scheduler state, RNG states, `model_config.json`;
`latest.txt` is written last, so any step dir it covers is complete.

Resume:

```yaml
checkpoints:
  resume_checkpoint_path: <checkpoints dir>   # reads latest.txt
```

The run restores model/optimizer/scheduler/RNG **and nothing else**. The
LSDB stream is cursorless: resume opens a fresh stream and records may be
revisited immediately. There is no exact-sequence continuation, no replay
audit, and no worker-count constraint on resume.

Watch the logs: `lm_loss`, per-modality `images_loss` (spectra_loss stays
untrained), `tokens_per_sec_per_gpu`, `model_tflops_per_gpu`, memory lines.

## 6. Evaluation status

Deferred. `astropt3.eval` keeps pure model-side functions — `evaluate`
(mean loss over provided batches), `embed_objects` + `ridge_r2` (linear
probe on objects you supply), `scalar_head_metrics`, sampling/rendering —
that callers drive with their own records or batches. The source-backed
CLIs (`val_loss`, `linear_probe`, `scalar_head` mains, run_probe_sweep,
generate) and their GPU sweep tests are removed until an LSDB-backed
evaluation seam is designed.

## 7. Verification gates

1. `uv run pytest` (CPU suite) green in `astro/`.
2. `uv run python scripts/count_params.py` green.
3. The synthetic-dependent `train_smoke` gate is suspended (see top note).

A bounded live check exists for hub outages:
`uv run pytest -m network tests/test_lsdb_stream.py`.

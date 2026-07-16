# Training AstroPTv3 models with nanotron

*The operational guide: environments, data, launching, checkpoint/resume,
evaluation, and the traps we already hit so you don't hit them again.
Background on the model itself is in [`architecture.md`](architecture.md).*

## 0. TL;DR — a complete tiny run

```bash
# from the repo root, in a GPU env (see §1)
CUDA_DEVICE_MAX_CONNECTIONS=1 torchrun --nproc_per_node=1 \
    nanotron/run_train.py --config-file astro/configs/nanotron/astropt3-test-tiny.yaml
```

That trains a 4-layer toy on synthetic (offline, schema-identical) data for
50 steps and writes a checkpoint. Everything below is the same thing with
real data, real sizes, and more GPUs.

## 1. Environments

Three distinct environments; do not mix them.

| env | where | contents | used for |
|-----|-------|----------|----------|
| `uv sync --extra dev` | anywhere | torch (CPU ok), torchdata, transformers, datasets | unit tests, CPU smoke, eval code |
| `uv sync --extra data` | machine with network | + lsdb, hats, nested-pandas | data prep (crossmatch) only |
| GPU venv | training machine | torch + **flash-attn** + nanotron (editable `nanotron/`) + astro (editable `astro/`) + psutil + torchdata | training, GPU tests, conversion |

flash-attn wheels are the constraint for the GPU venv: pick a torch version
with a prebuilt wheel for your CUDA (never compile it on a shared box).
Known-good recipe (A100, 2026-07):

```bash
uv venv gpuenv --python 3.13
uv pip install torch==2.8.0 \
  <flash_attn-2.8.3.post1+cu12torch2.8cxx11abiTRUE-cp313 wheel from GitHub releases> \
  -e nanotron -e astro psutil
```

Verify the env with the GPU test suite (one GPU, ~20 min; TP=2 test wants two):

```bash
pytest -m gpu astro/tests/test_nanotron_gpu.py astro/tests/test_phase4_gpu.py
```

## 2. Data

### 2.1 Prepare the pilot corpus (network, `[data]` env)

```bash
cd astro
uv run --extra data python scripts/prepare_pilot_data.py            # full corpus
uv run --extra data python scripts/prepare_pilot_data.py \
    --cone 217.9688 32.6198 600                                     # 10' smoke test
```

This lsdb-LEFT-crossmatches the MMU image catalog (~14M rows, ~4TB) with
DESI EDR SV3 spectra (1″ radius) and writes ~shard-sized parquet under
`../astroPTv3_data/pilot_v1/{train,val}/` (override with
`$ASTROPT3_DATA_ROOT` or `--out`). Facts to know:

- It is **journalled and resumable**: each HEALPix partition is logged in
  `progress.jsonl` only after its shards are atomically renamed into place.
  Kill it anytime; rerun the same command anywhere with the same `--out` and
  it continues with no loss and no duplicates. The full run is a multi-day
  job (~70 obj/s single-process); partial corpora are usable immediately.
- Train/val assignment hashes **whole order-7 HEALPix tiles** — spatially
  disjoint splits, stable across reruns (do not change the salt mid-corpus).
- **Never point a `--cone`/`--limit-partitions` smoke run at the canonical
  corpus directory**: cone runs row-filter partitions, and a partition
  journalled as "done" from filtered rows would be silently skipped (=data
  loss) by the full run later. Give smoke runs their own `--out`.
- DESI coverage is patchy: most partitions have 0% spectra matches, SV3
  rosettes have ~25%. Expect ~7% matched overall.

### 2.2 Image normalization (no calibration step)

Image flux is normalized physically, keyed on each record's band names
(`data/band_registry.py`): rescale to LegacySurvey nanomaggies → clamp
survey-flagged bright pixels → `arcsinh(flux/0.01)` — tokens are flux in
knee units (0.01 nMgy = 10 picomaggies), O(1) values. The constants come
from the surveys' own documentation, so there is nothing to calibrate per
corpus — unknown bands raise `NotImplementedError` (add them to
`BAND_REGISTRY`).

### 2.3 Gate the data before burning GPU-hours

```bash
HF_DATASETS_OFFLINE=1 uv run python scripts/check_pilot_data.py \
    --workers 8 --seq-len 4096 --target-tokens-per-sec <training tok/s>
```

Two checks: decoded objects must show per-patch mean ≈ 0 / std ≈ 1 and
spectra λ within 3600–9824 Å; and dataloader-only throughput should be
**≥2× the training consumption** you expect (see §4 for reference numbers;
remember each DP rank runs its own workers, so per-process throughput
multiplies by DP).

## 3. Configs

Full nanotron run configs live in `astro/configs/nanotron/`:

- `astropt3-test-tiny.yaml` — 4-layer toy, synthetic data, 1 GPU, 50 steps.
- `astropt3-{70m,160m,410m,1b,1p4b,2p8b,6p9b,12b}.yaml` — the pilot recipes
  for all eight sizes (dims from the PLAN size table, parallelism + peak LR
  from the recipe table below, GBS 512×4096 everywhere).
- `astropt3-70m-shakeout.yaml` — the 70M model on 2 GPUs (DP=2), real
  day-one data; the template for "I have a small box, not a cluster".

The interesting knobs, top to bottom:

```yaml
general:
  ignore_sanity_checks: true   # REQUIRED with DP>1: per-rank modality tensor
                               # shapes differ; the DP input check would crash
model:
  model_config:
    is_astropt3_config: true   # dispatches to AstroPT3ForTraining
    # modalities: omitted -> pinned pilot defaults (images 144x192 patch 8,
    #                        spectra 31x256 continuous-λ positions)
    tokeniser: affine          # or "aim" (MLP)
    _use_doc_masking: true     # position_ids restarts = document boundaries
parallelism:
  pp: 1                        # asserted by the model — do not change
  tp_mode: ALL_REDUCE          # asserted — modality modules are TP-replicated
  dp: <n>                      # main scaling axis
data_stages:
- data:
    dataset:
      data_root: <shard dir>   # or the literal string "synthetic"
      shuffle_buffer_size: 2000
      # object_id_log: <path>  # audit trail: one object_id/line as trained
    num_loading_workers: 8     # resume-exact at any value (torchdata
                               # StatefulDataLoader); keep it FIXED per run
optimizer:
  zero_stage: 1                # ZeRO-1 everywhere per the recipe table
tokens:
  sequence_length: 4096
  micro_batch_size: 16         # 70M/A100-80GB: peak ~32GiB — headroom to raise
  batch_accumulation_per_replica: 1  # raise to hit GBS on fewer GPUs
checkpoints:
  checkpoint_schedule: pythia  # steps 1,2,4,...,512 + every interval
  checkpoint_interval: 1000
  resume_checkpoint_path: null # set to the checkpoints dir to resume
```

To adapt a recipe to fewer GPUs: reduce `dp`, raise
`batch_accumulation_per_replica` to keep the global batch size, and leave
everything else alone. Model-size hyperparameters for all eight sizes are in
`astro/configs/model/*.yaml` and the table in `architecture.md` — copy the
70M nanotron yaml and swap the `model_config` block.

### Per-size recipe (Ultra-Scale Playbook; 80GB GPUs, seq 4096, GBS 2M tokens)

| Size | GPUs | TP | DP | ZeRO | recompute | peak LR |
|------|------|----|----|------|-----------|---------|
| 70M–410M | 32 | 1 | 32 | 1 | none | 1e-3 / 6e-4 / 3e-4 |
| 1B–1.4B  | 64 | 1 | 64 | 1 | selective | 3e-4 / 2e-4 |
| 2.8B     | 64 | 2 | 32 | 1 | selective | 1.6e-4 |
| 6.9B     | 128 | 4 | 32 | 1 | selective/full | 1.2e-4 |
| 12B      | 256 | 8 | 32 | 1 | full | 1.2e-4 |

TP never crosses the node boundary; no PP at any size. Warmup
min(2000 steps, 1%), cosine to 0.1× peak.

## 4. Launching

Single node:

```bash
HF_DATASETS_OFFLINE=1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
torchrun --nproc_per_node=<gpus> --rdzv-backend=c10d --rdzv-endpoint=localhost:0 \
    nanotron/run_train.py --config-file astro/configs/nanotron/<config>.yaml
```

(`--rdzv-endpoint=localhost:0` auto-picks a free port — handy when several
runs share a box. `HF_DATASETS_OFFLINE=1` keeps `datasets` from touching the
network on compute nodes.)

Multi-node goes through the launcher (from the repo root):

```bash
sbatch --nodes=<N> astro/scripts/launch_slurm.sbatch astro/configs/nanotron/<config>.yaml
# dry run (100 steps, checkpoints redirected to *-dryrun):
sbatch --nodes=<N> --export=ALL,DRY_RUN_STEPS=100 astro/scripts/launch_slurm.sbatch <config>
```

It sources `$ASTROPT3_ENV` (default `../astroPTv3_gpuenv`), rendezvous on
the first node, and runs one `torchrun` per node via `srun`; node counts per
size are in its header. The config's `dp*tp` must equal the allocated GPU
count.

**Always dry-run first** (playbook rule): run ~100 steps at the target
topology and check tokens/s/GPU, MFU, and memory before committing
node-hours. Reference points measured on 2×A100-80GB, DP=2, mbs 16,
seq 4096, real data, 8 workers/rank:

- 70M: ~0.6 s/step at GBS 131k tokens → ~240k tokens/s total,
  **123 model TFLOPs/GPU (~39% MFU)**, peak 31.7 GiB. Loss 0.46 → 0.26 in
  100 steps from scratch.
- Target MFU band across sizes: ~30–45%. If you're far under, suspect the
  dataloader (see §7).

What to watch in the logs: `lm_loss` (total), per-modality `images_loss` /
`spectra_loss` (should sit within ~5× of each other after warmup — if not,
revisit `loss_weight`), `tokens_per_sec_per_gpu`, `model_tflops_per_gpu`,
and the memory lines from the first steps.

## 5. Checkpoints and resume

With `checkpoint_schedule: pythia` you get dirs
`{1,2,4,...,512,1000,2000,...}` under `checkpoints_path`, each containing
model weights (bf16), optimizer + LR-scheduler state, RNG states,
`model_config.json`, and `dataset_state/dp_{rank}.pt` — the data-stream
position. `latest.txt` is written last, so any step dir it covers is
complete (the eval sweep keys off this).

To resume after a crash or kill:

```yaml
checkpoints:
  resume_checkpoint_path: <checkpoints dir>   # reads latest.txt
```

and relaunch the same command. The run restores weights/optimizer/RNG *and*
the stream position, then continues with **exactly the micro-batch sequence
an uninterrupted run would have consumed** — no replayed samples, no gap.
Constraints and semantics:

- Exact stream resume works at **any** `num_loading_workers`: with workers
  the stream position lives in the worker processes, so the loader is
  torchdata's `StatefulDataLoader` and the checkpoint stores its per-worker
  snapshots (torchdata must be installed in the GPU env — without it,
  workers > 0 refuses to start rather than train unresumably, which is how
  the 20k real-data shakeouts silently lost their stream state). The saved
  state maps per-worker, so **resume with the same `num_loading_workers`
  as the saving run** — a mismatch is rejected at load. Checkpoints written
  before this change (dataset-format) still resume, at workers 0 only.
- With `shuffle_buffer_size > 0`, resume skips at most the in-flight buffer
  and never replays a trained record (HF shuffle semantics).
- Set `dataset.object_id_log` to get a per-rank file with one `object_id`
  per trained object — the audit trail we use to *prove* no-replay in the
  GPU tests.
- When comparing a resumed run's losses to something, compare to **its own
  pre-kill trajectory**, not an independent run: two identical runs drift a
  few percent within ~100 steps from nondeterministic flash-attn backwards.

## 6. Evaluation (never blocks training)

The sweep runs as a separate process, ideally pinned to a spare GPU:

```bash
CUDA_VISIBLE_DEVICES=<spare> python astro/scripts/run_probe_sweep.py \
  --checkpoints-dir <run's checkpoints_path> \
  --out-dir <eval dir> \
  --data-root <val shard dir> \
  --norm-stats astro/configs/data/pilot_images_spectra.yaml \
  --seq-len 4096 --val-batches 512 --probe-objects 2048 \
  --watch --until-step <train_steps>
```

For each completed checkpoint it: converts to HF (`{out}/hf/{step}`), scores
a **fixed deterministic set of validation batches** (val loss comparable
across steps), ridge-probes redshift `Z` from mean-pooled hidden states
(test-split R²), and appends one JSON line to `{out}/probe_results.jsonl`.
Healthy runs show monotone-ish falling val loss and rising probe R² across
the Pythia checkpoints (the tiny reference run went val 0.456→0.051,
R² 0.42→0.79 over 1000 steps).

The pieces also run standalone against any converted checkpoint:

```bash
python -m astropt3.eval.val_loss    --checkpoint <hf_dir> --data-root <val dir> --seq-len 4096
python -m astropt3.eval.linear_probe --checkpoint <hf_dir> --data-root <val dir> --target Z
```

Note: the probe needs objects that *carry* the target — early partial
corpora may have very few spectra in val, in which case probe against a
spectra-rich train subset and treat R² as relative, not absolute.

Manual conversion, if you need just one checkpoint:

```bash
torchrun --nproc_per_node=1 nanotron/tools/astropt3/convert_nanotron_to_hf.py \
    --checkpoint_path=<checkpoints>/<step> --save_path=<hf_dir>
# then:  import astropt3; AutoModel.from_pretrained(<hf_dir>)
```

## 7. Troubleshooting / hard-won facts

**Crashes at the first backward with `NotImplementedError` in
`fp32_accum_hook`** — you are on a nanotron without fork commit `0668f369`.
Upstream's DDP + fp32-grad-accumulation + ZeRO-1 path routes into an
unfinished reduce-scatter branch; the fork forces the all-reduce path.

**Crashes at import with "Grouped GEMM is not available"** — upstream
`nn/moe.py` demanded `grouped_gemm` at import time even for dense models;
the fork defers it to MoE construction. If it reappears after an upstream
sync, re-apply (fork commit `831045ff`).

**`TypeError: RotaryEmbedding.__init__` takes 2 to 6 args** — flash-attn
≥2.8 dropped `pos_idx_in_fp32` from its rotary constructor; the fork's
`nn/rotary.py` passes the survivors by keyword. Historically reverted once
by an upstream merge — check after every sync.

**Crash on resume: "Mismatch between the total consumed tokens…"** —
`TrainingMetadata` asserts the per-stage token ledger matches the global
counter *at load time*; only BlendableDataset updated the ledger upstream.
The fork's trainer syncs it for astropt3 streams.

**DP>1 crashes in a sanity check about differing tensor shapes** — set
`general.ignore_sanity_checks: true` (required; see §3).

**Throughput far below the reference numbers** — almost always the
dataloader. Check `check_pilot_data.py` throughput vs consumption, raise
`num_loading_workers` (exact resume survives workers now), make sure the
shard count per rank ≥ workers (HF assigns whole shards to workers), and
keep `pin_memory` on (default). The synthetic generator is CPU-bound too — don't
benchmark compute with `data_root: synthetic` and 0 workers.

**Loss stuck ≈1.0 on synthetic-looking data** — per-patch standardization
turned structureless patches into irreducible N(0,1) targets. Check that the
data's bands are in `band_registry.BAND_REGISTRY` (so the physical
normalization applies at the right flux scale) and that the data has
patch-scale structure.

**Don't** use `datasets.IterableDataset.shuffle()` anywhere in the pipeline:
in datasets 5.x it collapses `n_shards` to 1 and silently destroys
rank/worker sharding. `MMUIterableDataset` implements its own seeded
shard-order + buffer shuffle for exactly this reason.

**BeeGFS/checkpoint pressure**: a 70M checkpoint dir is ~1GB (weights +
ZeRO-1 optimizer state); the Pythia schedule to 20k steps is ~30 dirs.
Prune non-schedule intermediates and convert+upload scheduled checkpoints
as they land at larger sizes.

## 8. Verification gates before calling a run "real"

1. `uv run pytest` (CPU suite) green in `astro/`.
2. `pytest -m gpu astro/tests/test_nanotron_gpu.py astro/tests/test_phase4_gpu.py`
   green in the GPU env.
3. `check_pilot_data.py` decode sanity + ≥2× throughput on the actual corpus.
4. 100-step dry run at the target topology: tokens/s/GPU + MFU logged and
   sane, memory has headroom, per-modality losses within ~5×.
5. Then launch, with the probe sweep watching from a spare GPU.

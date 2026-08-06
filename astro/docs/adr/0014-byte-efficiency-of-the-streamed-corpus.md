# ADR 0014 — Byte efficiency of the streamed corpus

**Status:** Accepted (2026-08-06). Supersedes `2026-08-06-useful-bytes-per-megabyte.md`,
which held the full working and was deleted after this record was extracted
(recover it from git history if a number here needs its derivation).

## Context

Training on the ADR 0006/0011/0013 live stream is **transfer-bound, not
compute-bound**. On the five-spoke North corpus the GPU idles at every cell
boundary while ~1.18 GB of partner data arrives: median 0.68 s/step against a
mean of 7.80 s/step, with 93% of wall clock inside stalls. Three independent
attempts to fix that with *concurrency* all failed (below). The only lever
that matters is therefore **loss-bearing token bytes per megabyte
transferred**.

## What a megabyte buys (measured, single row group per catalogue)

- LegacySurvey North anchor row group: 283 KB/object, of which **`image.flux`
  is 92.3%**. Every other anchor column is <0.1% — projecting them is
  pointless.
- `image.flux` is 3×152×152; `packing.IMAGE_CROP = 96` tokenises the central
  96×96. **60% of every image byte is discarded before it reaches a token.**
- DESI spectra: 114 KB/spectrum, of which `spectrum.ivar` is **41%** and is
  never read outside `synthetic.py`.
- KB per loss-bearing token: anchor/PROVABGS scalars ~0 (same row we already
  pay for), full-frame image 0.73, ivar-projected spectrum 1.15, today's 96×96
  crop 1.82, today's spectrum 1.93, **HSC 3.56** — the most expensive token in
  the corpus by 5×.
- The discarded periphery is **not** empty sky: median per-patch std is within
  1% of the crop's, and only 12.7% of full-frame patches fall below the crop's
  own 10th percentile. Full-frame tokenisation would be 2.5× more tokens of
  roughly equal quality at zero extra bytes.

## Decisions

**Selected and carried forward:**

1. **AR span-order replay** — *built and measured 2026-08-06*. `ar_replicas`
   in `nanotron_loader.py` re-emits each downloaded record under a different
   ADR 0008 span order (already a pure function of `(object_id, epoch)`, so
   the suffixed id *is* the reseed — no new RNG, nothing extra to checkpoint,
   resume stays exact). Replica 0 keeps the original `object_id`, replicas get
   `#n`, so the no-replay audit still catches accidental duplication.
   Measured over identical first 117 steps, five-spoke, 8 workers, dp=2:
   `ar_replicas: 1` → 7.80 s/step mean, 21/116 slow; `ar_replicas: 2` → 2.51
   s/step mean, 11/116 slow. **~3× faster per step.** Total exposures per
   object over fixed `train_steps` are **unchanged** (8.7 at every replica
   count) — only the arrangement differs, and higher replicas move *toward*
   fewer epochs. Residual risk is within-batch correlation (document masking
   means replicas cannot attend to each other, so it is purely a
   gradient-diversity effect); a shuffle buffer would decorrelate them.
2. **Per-band tokenisation** — selected, not yet built. 8×8×3 = 192 floats
   becomes three 8×8×1 = 64-float tokens: **3× tokens from identical bytes**,
   and cross-band structure is learned autoregressively instead of pre-fused.
   Open before building: object length triples (check p95/p99 against
   `sequence_length: 4096`), the jetformer flow/GMM dims are tied to
   `input_size` so the flow narrows, band order becomes an AR ordering
   decision, and checkpoints are incompatible — needs its own A/B at matched
   tokens *and* matched bytes.

**Standing, unselected:**

3. Project out `spectrum.ivar` — free and correct, but **demoted**: DESI is
   6.3% of five-spoke anchors (was 92.2% at three spokes), so it now saves 40%
   of the spectrum bytes of 6% of rows.
4. The image crop (tokenise the full 152×152, or foveate: patch 8 centre +
   patch 16 annulus ≈ 198 tokens for all pixels) is the **largest remaining
   lever** — but it is a modelling change, and belongs in its own ADR with an
   A/B, not a perf pass.
5. Epoch byte reuse (a transient, evictable local cache) is in tension with
   ADR 0006/0011 and dominates everything else at multi-epoch scale. Raise it
   explicitly before multi-epoch training makes it urgent.

**Rejected — measured, do not repeat:**

- **Cell-boundary prefetch thread.** Not merely neutral: at
  `num_loading_workers: 8` on the five-spoke corpus it hung a rank for 20
  minutes and NCCL's watchdog killed the job. It adds a *second* concurrent
  large read per loader process (~32 in flight instead of 16) and creates no
  bandwidth. Fine at 2 workers, which is itself a bad operating point.
- **fsspec `cache_type=background` + pyarrow `pre_buffer`.** Within noise of
  baseline (3.73 vs 2.88 s/step mean, identical slow-step count).
- **Row-group / page skipping in partner partitions.** Every DESI partition is
  a *single* row group (up to 520 MB) and no page index is written. There is
  no sub-file granularity to skip to.
- **Filtering "empty sky" patches** (refuted by the periphery measurement),
  **HATS index tables** (solve lookup, which the match index already gives),
  **harder anchor column projection** (<0.1% of bytes).

**Inconclusive:**

- **Block-shuffled cell order.** Built and run at block 256. Note
  `shuffled()` runs *before* `owned_by_rank`, which deals cells round-robin to
  `dp × num_loading_workers` consumers — so a block smaller than the consumer
  count strips the locality straight back out. Over 36 steps: identical median
  (0.69 vs 0.68 s), identical slow-step count. **No signal either way**;
  reverted unmerged in favour of replay, not refuted. The 39% re-download
  figure that motivated it was measured on the 181-cell index and must be
  re-measured through `HfFileSystemFile._fetch_range` byte accounting — the
  training log cannot substitute (HTTP requests count *ranges*, and 16 loader
  processes interleave in one log with no worker id).

## Mechanism notes worth keeping

- **Stalls arrive on the loader rotation, not at random.** At
  `num_loading_workers: 8`, every >60 s stall from step 80 landed at exactly
  `step % 8 == 6`. Each worker's cell-boundary fetch blocks its own turn and
  nothing amortises it. This is why adding workers cannot help: each new
  worker adds its own serialised fetch to the same rotation.
- **Worker count is nonetheless a large lever**, and the concurrency table
  above says nothing about it (all rows are 8-worker runs). Measured: 2
  workers → 19.95 s/step vs 8 workers → 9.02 s/step, because a ~100 s fetch
  amortises over `num_workers` steps of the rotation.
- **The partition ceiling is gone.** `num_loading_workers <= floor(cells/dp)`
  bound the pilot recipes at 165 train cells; at 5,488 cells it no longer
  binds, and the deferred "shard by row group instead of by cell" loader
  change is not worth building.
- **`VAL_PARTITIONS = 8`** now reserves 0.15% of cells rather than 4.4%. Fine
  with `limit_val_batches: 0`; raise it before any run whose val loss is
  load-bearing.
- **Unmatched rows stay** (ADR 0005/0011), and the economics now back it:
  36.3% of training tokens come from rows never fetched for their own sake, at
  zero marginal bytes, spending compute we have in surplus. The real threat is
  **composition variance**, not the matched/unmatched ratio — HSC was 0.1% of
  objects in one run and 16.4% in another because its matches live in a small
  part of the footprint. Per-family loss weights normalise *within* a batch,
  so a stretch of spectrum-only batches trains no image head at all. Block
  ordering would make this worse. `object_id_log` already records what is
  needed to measure it.

## Caveats

Column shares come from one row group of one partition per catalogue; bytes
are compressed on-disk sizes, and pyarrow also fetches footers and coalesces
ranges, so real HTTP bytes are somewhat higher. Runs compared here hit a
shared hub at different times — the *absence* of benefit is the finding, not
the ordering between arms.

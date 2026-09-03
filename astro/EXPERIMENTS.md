# Data-loader throughput experiments — 2026-09-02

## Context

`astropt3-70m-jetformer-crossmatch.yaml` (DP=2, 8 loading workers/rank,
streaming `mmu_desi_edr_sv3 x mmu_ssl_legacysurvey_north` live via
`nanotron_loader.py`'s direct-lsdb path) was observed running with the GPU
mostly idle. This log covers the investigation into why, what was tried,
what was measured on real runs, and what's still open. All "real run"
numbers below come from `scripts/bench_report.py` reading the
`ASTROPT3_TELEMETRY_DIR` output of short verification runs on spare GPUs
(2,3), using a scratch checkpoint path and (mostly) `WANDB_MODE=disabled` so
they don't collide with production training. Two runs were logged to wandb
(`astropt3-loader-verify` project) for visual inspection.

## 1. Where the stall was coming from

`cProfile` against the real `InfiniteStream`/`PackedMicroBatches` path
(`profile_stream.py`, ad hoc) showed two distinct causes:

- **Legacy-only stream**: cold start (catalog open + first partition fetch)
  ≈ 40s, of which 34.6s was pure `_ssl._SSLSocket.read` — genuine network
  wait. Steady-state decode/pack of an already-fetched partition was
  ~30ms/micro-batch — negligible.
- **Crossmatch stream**: steady-state was 0.47s/micro-batch, ~89% of it
  inside `_as_mapping`'s `DataFrame.to_dict()` fallback. Each DESI
  spectrum arrives as a 7781-row × 5-column per-record `DataFrame`;
  `to_dict(orient="list")` boxes every one of those ~39k cells
  individually via pandas' `maybe_box_native`. This was a real, fixable
  CPU bug, not network cost.

## 2. Fix: fast-path the spectrum DataFrame, then migrate to `map_rows`

- First fix: `_decode_spectrum` reads columns directly via `.to_numpy()`
  when handed a `DataFrame`, skipping `to_dict()` entirely.
  **0.47s → 0.02s/micro-batch.**
- Investigated `nested_pandas.NestedFrame.map_rows` (the library's own
  purpose-built row-wise accessor for nested/struct columns — delivers
  nested sub-columns as numpy arrays straight from Arrow storage, never
  materializing a per-row `DataFrame` at all). Benchmarked in isolation:
  **19x faster** than the already-fixed `.iterrows()` path for spectrum
  extraction alone (0.241s → 0.013s over 1017 rows), byte-identical output.
- Migrated `_open_records`'s full decode loop to `map_rows`
  (`_row_from_map_rows` reassembles the dotted-key output back into the
  nested-mapping shape `decode_legacy_row`/`decode_crossmatch_row` already
  expect, so their public contracts didn't change). Offline test fixtures
  were upgraded from plain `pandas.DataFrame` to real
  `nested_pandas.NestedFrame`s (`legacy_fixture.nested_frame`, via
  `join_nested`) so the CPU suite exercises the same code path as
  production — the old fake fixtures didn't support `map_rows` at all,
  which the test suite alone wouldn't have caught.
- **Net: crossmatch decode 0.47s → 0.01s/micro-batch (~47x)**, legacy-only
  2.6x faster. Decode dropped out of the profile entirely; packing
  (`torch.arcsinh`, `spectral_normalize`) became the dominant CPU cost at
  ~10-20ms/micro-batch.

Commits: `637ba68` (rows/rows_per_s telemetry, used throughout this
investigation), `e881e9b` (crossmatch decoding baseline), `c330446`
(`map_rows` migration).

## 3. Fetch overlap: does prefetching help?

`lsdb.streams.catalog_streams.InfiniteStream` only prefetches in the
background when given a real `dask.distributed.Client` — with the
`client=None` default it was using, `submit_next_partitions` calls
`.compute()` synchronously, so partition N+1's fetch fully blocks the call
that returns partition N. Zero overlap with anything.

Added a lightweight per-worker `dask.distributed.Client` backed by
`LocalCluster(processes=False, ...)` — an in-process scheduler + thread
pool, not a distributed cluster (~30ms startup, ~1.5MiB RSS, confirmed via
direct measurement). Verified the mechanism works correctly in isolation
(non-blocking submit, genuine background progress on a `Future` checked
mid-flight).

**Result on a real run: no measurable improvement.** `stall_share` stayed
flat at ~85% before and after. Root cause: a worker only stays
`DataLoader`'s `prefetch_factor` micro-batches ahead of training-step
consumption before it blocks — a depth-1 partition lookahead barely dents
a 10-50s fetch, because the wall-clock window it gets to run in isn't "however
long draining a partition takes," it's bounded by how fast the main loop
is actually consuming.

Commit: `9800176` (kept — correct mechanism, real value once paired with
a wider window; see next section).

## 4. `partitions_per_chunk` tuning (pre-outer-crossmatch)

Widening the window via `InfiniteStream(partitions_per_chunk=N)` — fetch N
partitions per draw instead of 1, giving the background prefetch more time
to finish before the worker needs it. Real-run sweep (same config, DP=2,
8 workers/rank):

| chunk | stall_share | p95 step | p99 step | rows/s | MFU | RAM (box-wide) |
|---|---|---|---|---|---|---|
| 1 | ~85% | ~13s | ~51s | ~1,000 | 2.6% | not measured |
| 4 | 65.4% | 0.46s | 24.0s | 2,044 | 5.5% | ~161GB |
| **8** | **45.5%** | **0.45s** | **6.4s** | **3,324** | **8.6%** | ~237GB |
| 16 | 61.7% | 0.45s | 6.8s | 2,312 | 6.0% | ~595GB |

`chunk=8` was a real optimum, not a plateau — 16 regressed on every metric
(worse stall, worse throughput) while RAM more than doubled, consistent
with 16 workers × 2 chunks-in-flight × 16 partitions pushing into genuine
memory/scheduling pressure.

Commit: `74851c3` (`_PARTITIONS_PER_CHUNK = 8`).

## 5. Wire-efficiency investigation

With the network confirmed as the dominant remaining cost (measured ~815
MiB/s aggregate at chunk=8, ≈65% of the box's 10Gbps NIC — not clearly
link-limited, but not far off either), the next question was whether the
bytes being pulled were actually useful.

- **Byte breakdown at chunk=8**: 66.8% of wire bytes were Legacy image
  data.
- **Measured match rate** (live sample, 18,355 DESI rows across 4
  partitions): only **10.0%** have a Legacy image match within 1
  arcsec; **90.0%** are spectrum-only.
- **Read `lsdb`'s crossmatch source** (`crossmatch_catalog_data.py`,
  `kdtree_match.py`): the KDTree match algorithm reads the *full* right
  (Legacy) partition — including the large `image` struct column — for
  every candidate row in an overlapping pixel pair, *before* running the
  distance filter. The ~90% of rows that don't match within radius get
  their image bytes fetched and then silently dropped by
  `AbstractCrossmatchAlgorithm._create_crossmatch_df`.
- **Net finding**: roughly **60% of total wire bytes** (66.8% × ~90%)
  were being spent downloading Legacy images that get discarded.
- **Tried and rejected**: dotted sub-column projection
  (`columns=["spectrum.flux", "spectrum.lambda", "spectrum.mask"]`
  instead of the whole `spectrum` struct) to drop the unused `ivar`/
  `lsf_sigma` fields. Confirmed by direct measurement this does **not**
  reduce wire bytes — HATS/parquet reads the whole nested column's
  row-group regardless of which sub-fields are kept afterward. In-memory
  schema pruning only, no wire savings. Negative result, not pursued
  further.
- **Not pursued**: `MFU_busy` (compute efficiency while *not* stalled) sat
  at ~16-18% across every run measured, against a 989 TFLOP/s peak — a
  separate, wire-independent lever (kernel/model-shape efficiency, not
  data pipeline). Flash attention, fused RMSNorm, fused rotary embeddings
  and QKV packing are already on in this config; no GPU-level profiling
  (torch profiler / nsys) was done this session to look further.

## 6. `OuterKdTreeCrossmatch`: recovering the wasted 60% for free

Since the unmatched Legacy image bytes are already being read into memory
before being discarded, the fix is to stop discarding them rather than to
fetch less — zero extra network cost either way.

`outer_crossmatch.py`'s `OuterKdTreeCrossmatch` subclasses `KdTreeCrossmatch`
and extends `how="left"` row assembly to also emit unmatched right-side
(Legacy) rows as image-only records (NA spectrum/DESI columns), instead of
silently dropping them. Two real bugs were caught by testing against live
data rather than shipping on inspection alone:

- **Margin-cache double-counting**: the right catalog's margin cache
  (candidate rows borrowed from a neighboring partition, included only to
  catch boundary-crossing matches) had to be excluded via spatial-index
  filtering (`healpix_to_spatial_index` bounds check against the row's own
  native pixel) — otherwise a margin row gets emitted here *and* again
  when its home partition is processed.
- **`pandas.NA` vs `None`**: a null `object_id` (pyarrow string) scalar
  comes through as `pandas.NA`, not Python `None`, both via raw pandas
  access and via `map_rows`. The original `decode_crossmatch_row` used
  `is None`, which would have silently decoded these rows' `object_id` as
  the literal string `"<NA>"` rather than falling back to
  `object_id_legacy` or raising. Fixed by switching to `pd.isna()` (a
  strict superset of `is None`, so plain-`None` test fixtures are
  unaffected).

**Verified on real data**: 6,703 rows split exactly into 1,403 matched +
4,333 spectrum-only + 967 image-only — zero loss, zero duplication, zero
rows with neither modality. `ObjectSequencer.build()` succeeds on all
sampled image-only objects.

Scope is deliberately narrow: this only recovers Legacy rows *within pixel
pairs the crossmatch already visits* (near some DESI pointing). A Legacy
partition with no DESI coverage nearby is never fetched by this join at
all — that's the majority of unmatched Legacy imagery, and recovering it
would need a genuinely separate, full-wire-cost stream (not built).

Commit: `62cce63`.

## 7. Re-tuning `partitions_per_chunk` after the outer join

`OuterKdTreeCrossmatch` recovers real extra data, but each recovered image
row is real extra weight (~277KB/row) buffered per chunk. This pushed
`chunk=8` into the same RAM-pressure regression `chunk=16` hit in section
4. Re-tuned with two equally-mature real-run readings (~552-575s each):

| | chunk=8 + outer-join | chunk=4 + outer-join |
|---|---|---|
| stall_share | 49.8% | **32.8%** |
| rows/s | 2,324 | **2,997** |
| MFU | 8.90% | **12.13%** |
| E_values/MiB | 66,218 | **89,458** |
| p99 step | 7.75s | **3.50s** |
| slow steps (>60s) | 1 | **0** |
| RAM (box-wide) | 529GB | **349GB** |

`chunk=4` won on *every* metric, not just RAM — confirms this wasn't a
tradeoff, `chunk=8` was genuinely past the regression point once the outer
join's extra data is counted. `chunk=2` was not tested (stopped once the
win was decisive); worth checking if further tuning is ever revisited.

Commit: `696e359` (`_PARTITIONS_PER_CHUNK = 4`).

## 8. Current state

Final settings as of this log: `map_rows`-based decode, per-worker
lightweight `dask.distributed` prefetch client, `partitions_per_chunk=4`,
`OuterKdTreeCrossmatch` for the crossmatch stream. Confirmed with a
wandb-logged run of this exact configuration
([`astropt3-loader-verify/runs/jsoi53bq`](https://wandb.ai/smith42/astropt3-loader-verify/runs/jsoi53bq),
separate project, scratch checkpoint path — not production), read at a
comparable maturity (540s) to the section-7 numbers:

| | section 7 reading | wandb-confirmed reading |
|---|---|---|
| stall_share | 32.8% | 35.4% |
| rows/s | 2,997 | 2,881 |
| MFU | 12.13% | 11.64% |
| E_values/MiB | 89,458 | 87,992 |

Consistent within normal run-to-run variance — the section-7 numbers hold.

**End-to-end from the original ~85% stall_share baseline (section 1) to
here: stall_share 85% → 35%, rows/s ~1,000 → ~2,900 (~2.9x), MFU 2.6% →
11.6% (~4.5x).**

## Open items / not pursued

- **`chunk=2`**: not tested after the section-7 re-tune; the win at 4 was
  decisive enough to stop there.
- **Compute-side MFU (`MFU_busy` ~16-18%)**: a real, wire-independent
  lever, not investigated this session. Needs GPU-level profiling
  (torch profiler / nsys), not data-pipeline telemetry.
- **Legacy imagery outside DESI's footprint**: the majority of unmatched
  Legacy images (partitions with no nearby DESI coverage at all) are still
  never fetched by this stream. Recovering them needs a genuinely separate
  stream at full wire cost — a real design decision (more corpus coverage
  vs. more bandwidth spent), not attempted here.
- **lsdb docstring accuracy**: `InfiniteStream`/`CatalogStream`'s docstring
  claims background prefetching happens generically ("derived from
  `client` object"), which overstates the `client=None` case (fully
  synchronous). Worth a small upstream issue/PR; not filed.

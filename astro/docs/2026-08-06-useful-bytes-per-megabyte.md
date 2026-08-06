# Useful bytes per megabyte downloaded

**Status:** for review. Measurements are real; recommendations are not yet
decisions. Written 2026-08-06 after the ADR 0013 three-spoke GPU runs showed
the corpus is transfer-bound rather than compute-bound.

> **AMENDED 2026-08-06 (later the same day).** Everything below the next
> section was measured on the **three-spoke** index (149,491 anchors, 181
> cells). The five-spoke index finished building at 04:47 and changed the
> corpus by 14.6x, which invalidates several conclusions — most importantly
> the headline `spectrum.ivar` recommendation. Read "Corrections after the
> five-spoke corpus" first; it says which claims survive.

---

# Corrections after the five-spoke corpus (2026-08-06, later)

The SDSS and galaxies-with-hats spokes landed overnight. The merged index is
`astroPTv3_index/north-v2-merged-5spoke/match_index.parquet`.

| | three-spoke (this doc) | five-spoke (now) |
| --- | ---: | ---: |
| anchors | 149,491 | **2,182,875** |
| cells | 181 | **5,488** |
| edges | 265,481 | 2,494,712 |

Spoke coverage, as a share of anchors: `galaxies_train` **94.1%**, `sdss`
8.0%, `desi` **6.3%**, `provabgs` 5.1%, `hsc` 0.7%. Spokes per anchor:
`{1: 1,913,654, 2: 230,786, 3: 34,286, 4: 4,117, 5: 32}`.

## 1. The `spectrum.ivar` projection is demoted, not wrong

The 40% measurement (60.7 -> 36.4 MB, counted through
`HfFileSystemFile._fetch_range`) still stands **per spectrum read**. What
changed is how much of the corpus it touches.

On the three-spoke index, 137,906 of 149,491 anchors (**92.2%**) carried a
DESI match, so nearly every anchor forced a read into a DESI partition. On
the five-spoke index DESI is **6.3%** of anchors. The saving is now roughly
40% of the spectrum bytes of 6% of rows, against a byte budget the document
itself measures as 92.3% `image.flux`.

**It is still free and still correct — it is no longer "the recommendation".**
The image crop (option 2) is now the only lever of consequence.

## 2. `galaxies_train` is why throughput improved, and it is a column-projection story

`_SOURCE_COLUMNS` projects per spoke. `desi`/`sdss` read the `spectrum`
array and `hsc` reads `image`, but the galaxies branch reads only
`dr8_id, ra, dec, _healpix_29` plus the 34 `gwh_*` float fractions — **all
scalars, no arrays**. So the 94% of anchors that are galaxies-only cost
essentially their anchor image and nothing more, where previously nearly
every anchor dragged in a single-row-group DESI partition of up to 520 MB.

Corroborated by the fetch stream of a live five-spoke run:
`mmu_ssl_legacysurvey_north` 587 references, `galaxies-with-hats` 414,
`mmu_sdss_sdss` 48, `mmu_desi_edr_sv3` **6**, `mmu_desi_provabgs` **3**.
DESI has essentially left the byte budget.

A second mechanism follows from the cell count: `_spectrum_owners` assigns
each DESI spectrum partition to exactly one owning cell by crc32 over the
path. Spreading that over 5,488 cells instead of 181 means **~97% of cells
now own no unmatched DESI partitions at all**, where before every cell owned
~1/181 of DESI and paid for it at its boundary.

## 3. Appendix C's 39% re-download figure needs re-measuring

The `434 refs / 266 distinct` count and every LRU/block-size table in
Appendix C were computed against the 181-cell index. At 5,488 cells the
spatial-locality picture is different and the numbers should not be quoted.

Two attempts to re-measure it today from a live run's HTTP log **both
failed**, and the failures are worth recording so nobody repeats them:

- counting HTTP requests per partition measures **range requests**, not
  downloads — one 774 MB partition is ~14 row-group GETs, which inflates
  "redundancy" to a meaningless 26x;
- counting *contiguous runs* of requests per partition is defeated by
  concurrency: 16 loader processes share one log, so one worker's request
  splits another's run. The log carries `DP=` tags but no worker id.

The original 39% was measured properly with byte accounting through
`HfFileSystemFile._fetch_range`. **That instrument, not the training log, is
the one to use.**

## 4. Block-shuffled cell order: built, inconclusive, reverted

Implemented as a block permutation in `shuffled()` and run at 8 workers. One
design constraint the appendix understates: `shuffled()` runs **before**
`owned_by_rank`, which deals cells round-robin to `dp * num_loading_workers`
consumers. A block of size `B` therefore contributes `B/consumers` cells to
each consumer, and at `B == consumers` each consumer gets exactly one cell
per block — the locality is stripped straight back out. Appendix C's
suggested blocks of **8-16 are smaller than the 16 consumers of a dp=2 x 8
worker run** and would have bought nothing. A usable block is ~256.

Measured over the first 36 steps against a matched baseline: identical
median (0.69 vs 0.68 s), identical slow-step count (6/35), mean worse but
entirely from two startup stalls. **No signal either way** — reverted
unmerged in favour of the replay work below, not because it was refuted.

## 5. "What does not work" is too kind to the prefetch, and wrong about workers

Two corrections to that table.

**The prefetch is worse than "no better than baseline" — at 8 workers it is
fatal.** Restored from the stash and run on the five-spoke corpus, it reached
step 7 and then one rank's loader produced nothing for 20 minutes; NCCL's
collective watchdog killed the job. Re-run at 2 workers with the same code it
ran fine past that point, which distinguishes bandwidth contention from a
deadlock: the prefetch adds a *second* concurrent large read per loader
process (~32 in flight instead of 16), and a cell read stretches past the
timeout. It does not create bandwidth; at a worker count that already
saturates the pipe there is no idle bandwidth to prefetch into.

**Worker count is a large lever, and the table's phrasing obscures it.** All
three rows of that comparison were run at `num_loading_workers: 8`; the
"within noise" finding is about baseline vs prefetch vs readahead, *not*
about worker count. The run table in the handoff shows 1 worker at 24 s/step
against ~2.9 s/step at 8. Confirmed today from the wrong direction: dropping
to 2 workers on the five-spoke corpus gave **19.95 s/step against the
8-worker baseline's 9.02 s/step**, because a worker's cell fetch amortises
over `num_workers` steps of the loader rotation — at 2 workers a ~100 s fetch
is spread over 2 steps instead of 8.

## 6. Replay (idea D) is built and measured — it works

Implemented as `ar_replicas` in `nanotron_loader.py`. The ADR 0008 span order
is already a pure function of `(object_id, epoch)`, so re-emitting a record
under a suffixed id **is** the reseed: no new RNG, nothing extra to
checkpoint, and resume stays exact because the stream position is still a
record index.

The appendix's three open questions are answered:

- **exactly-once audit:** replica 0 keeps the original `object_id`, replicas
  get `#n`. Every logged line stays unique, so the no-replay audit still
  catches accidental duplication while permitting deliberate replay. Verified
  live: every worker logs exactly 50% replicas at `ar_replicas: 2` and zero
  duplicate lines.
- **resume:** `count` still increments once per record, so replicas are
  re-derived on resume rather than saved.
- **overfitting budget:** at fixed `train_steps` this is a non-issue, and the
  arithmetic is the opposite of what was feared. `seq_len` fills with the same
  number of *objects* regardless of replicas, so unique records per step falls
  proportionally. Measured on a live run at 952 objects/step:

  | `ar_replicas` | records/step | epochs over 20k steps | total exposures/object |
  | ---: | ---: | ---: | ---: |
  | 1 | 952 | 8.7 | **8.7** |
  | 2 | 476 | 4.4 | **8.7** |
  | 4 | 238 | 2.2 | **8.7** |

  Total repetition is **unchanged**; only the arrangement differs. Raising
  replicas moves *toward* the data-constrained-scaling comfort zone (fewer
  epochs), not away from it. The real risk is within-batch correlation —
  replicas land in the same or adjacent packed rows, so effective batch size
  falls with replicas even though wall-clock cost does not. Document masking
  means the copies cannot attend to each other, so it is purely a
  gradient-diversity effect. A shuffle buffer would decorrelate them.

Throughput, five-spoke, 8 workers, dp=2, over the identical first 117 steps:

| arm | median | mean | slow steps | % of wall in stalls |
| --- | ---: | ---: | ---: | ---: |
| `ar_replicas: 1` | 0.68 s | **7.80 s** | 21/116 | 93% |
| `ar_replicas: 2` | 0.70 s | **2.51 s** | 11/116 | 69% |

**~3x faster per step**, with slow steps roughly halved — half as many
distinct records per step means half as many cell boundaries crossed. It also
broke up the rigid stall periodicity described in the next item.

## 7. New observation: stalls arrive on the loader rotation, not at random

On the `ar_replicas: 1` five-spoke baseline, every stall over 60 s from step
80 onward landed at exactly `step % 8 == 6` with gaps of exactly 8, at
`num_loading_workers: 8`. The DataLoader round-robins its workers, so each
worker's cell-boundary fetch blocks its own turn in the rotation and nothing
amortises it: a worker spends ~100 s fetching to serve ~8 steps x 0.68 s of
work.

This is the mechanism behind the concurrency negative result. Adding workers
cannot help, because each added worker adds its own serialised fetch to the
same rotation.

## 8. Corpus-scale side effects

- **The partition ceiling is gone.** `num_loading_workers <= floor(cells/dp)`
  bound the pilot recipes at 165 train cells (dp=64 -> 2 workers). At 5,488
  cells it no longer binds, and the deferred "shard by row group instead of
  by cell" loader change is not worth building.
- **`VAL_PARTITIONS = 8`** now reserves 0.15% of cells rather than 4.4%.
  Fine for throughput runs with `limit_val_batches: 0`; wants raising before
  any run whose val loss is load-bearing.
- **Appendix D's composition variance is not resolved by the new corpus.**
  HSC is now 0.7% of anchors and its matches are still confined to a small
  part of the footprint, so the 0.1% -> 16.4% swing risk stands.

## What is still true

Unaffected by the corpus change: the image-crop measurement (92.3% of an
anchor row group is `image.flux`, 60% discarded by the 96x96 crop), the
periphery-is-not-empty-sky result in Appendix B, the Parquet mechanism
survey in Appendix A (one row group per DESI partition, no page index, PLAIN
float encoding), and the argument in Appendix D for keeping unmatched rows.

---

## Selected for follow-up (owner, 2026-08-06)

Of everything below, the project owner picked **two** to carry forward. They
are written up in full in Appendix B; this is the pointer.

### >> AR span-order replay while stalled (Appendix B, idea D)

The GPU idles for minutes at every cell boundary while ~1.18 GB of partner
data arrives, and the previous cell's decoded objects are still in memory.
Re-emitting them under a different autoregressive span order — the ADR 0008
shuffle, already seeded on `crc32(object_id) ^ epoch` and therefore already a
supported source of variation — converts dead GPU time into gradient steps at
**zero additional bytes**.

Why it is attractive: it is the only idea that exploits the stall instead of
fighting it, and three separate attempts to fight it (workers, cell prefetch,
fsspec/pyarrow readahead) measured as no better than baseline.

Open questions before building:
- the exactly-once / no-replay audit (`object_id_log`) must distinguish a
  replayed object from a duplicated one, or the Phase 4 gate breaks;
- resume semantics: a replayed step is not a new stream position, so the
  saved state must not advance on replay;
- how much replay before it becomes overfitting rather than free signal —
  needs an A/B against wall-clock-matched training, not step-matched.

### >> Block-shuffled cell order + small partner cache (Appendix C)

39% of partner-partition fetches are re-downloads (434 refs, 266 distinct)
because the per-epoch cell shuffle destroys HATS spatial locality. Shuffling
*blocks* of 8-16 contiguous cells instead of individual cells, with a 2-8
partition LRU, cuts partner fetches 424 -> ~280-292 while still shuffling
11-23 blocks per epoch. Unlike prefetching, this removes bytes rather than
rearranging when they arrive.

Open questions before building:
- bumps `SOURCE_GRAPH_ASSEMBLY` (record order changes; stale states rejected,
  weights still load);
- cache must be byte-bounded (a projected DESI partition is ~245 MB), not
  slot-bounded;
- **interacts with composition stability** — see Appendix D: spatial blocks
  make the modality mix *more* clumped, and it is already swinging wildly.

### >> Per-band tokenisation (Appendix B, idea B)

A token today is 8x8x3 = 192 floats spanning g/r/z. Splitting it into three
8x8x1 = 64-float tokens yields **3x the tokens from identical bytes**, and
makes the model learn cross-band structure autoregressively instead of
receiving it pre-fused in the channel dimension.

Why it is attractive: pure bytes/token win with no new data, and it is a
small, testable change — a modality-registry and patchify change, not a
pipeline redesign.

Open questions before building:
- sequence length per object triples for images; check p95/p99 object length
  against `sequence_length: 4096` and the packing utilisation;
- the jetformer flow and GMM head dimensions are tied to `input_size`, so the
  flow gets narrower — may change the noise curriculum's useful range;
- band order becomes an autoregressive ordering decision (fixed g,r,z, or
  shuffled with the ADR 0008 span rule?);
- checkpoints are incompatible; this needs its own A/B against the current
  fused-band tokenisation at matched tokens *and* matched bytes.

---

## Why this matters now

The 70M three-spoke runs sit at ~0.63 s/step median but ~3 s/step mean, with
~16% of steps taking minutes. Three separate attempts to fix that with
*concurrency* — 8 loader workers, a cell-boundary prefetch thread, fsspec
background block cache plus pyarrow `pre_buffer` — produced **no measured
improvement** in a like-for-like window (details in "What does not work"
below). That is the signature of a bandwidth limit, not a latency limit.

If we are bandwidth-limited, the only lever that matters is **how many
loss-bearing token bytes we get per megabyte transferred**. This document
measures that ratio and lists the ways to raise it.

## What a megabyte buys today

Measured on real partitions at the pinned revisions (single-partition samples,
so treat as indicative, not corpus-wide).

### Anchor images — LegacySurvey North

`Npix=9835.parquet`: 2384 rows, 12 row groups, row group 0 = **56.6 MB / 200
rows = 283 KB per object**.

| column | MB in row group | share |
| --- | ---: | ---: |
| `image.flux` | 52.31 | **92.3%** |
| everything else (`ebv`, `flux_*`, `psfdepth_*`, ids, coords) | <0.1 | ~0% |

The image is the corpus. Every other anchor column is free.

`image.flux` is `3 x 152 x 152` float32 = 277 KB raw, and compresses to ~262
KB — float noise does not compress. But `packing.IMAGE_CROP = 96` takes the
central 96x96 before patchifying:

```
downloaded    3 x 152 x 152 = 69,312 floats = 277 KB
tokenised     3 x  96 x  96 = 27,648 floats = 111 KB   (144 tokens x 192)
useful fraction                                39.9%
```

**60% of every image byte we pay for is discarded before it reaches a token.**

### DESI spectra

First partition: 34 rows, **3.9 MB / 34 rows = 114 KB per spectrum**.

| column | MB | share | used by the model? |
| --- | ---: | ---: | --- |
| `spectrum.flux` | 1.59 | 41.2% | yes |
| `spectrum.ivar` | 1.58 | **41.0%** | **no** |
| `spectrum.lambda` | 0.11 | 2.9% | yes |
| `spectrum.mask` | 0.00 | ~0% | yes |

`packing.py:129-131` reads `flux`, `lambda`, `mask`. `grep -rn ivar
src/astropt3/` returns nothing outside `synthetic.py` fixtures. We download
`ivar` on every spectrum and never look at it.

## Options, best ratio of payoff to effort first

### 1. Project out `spectrum.ivar` — 41% of every spectrum byte, no modelling change

> **DEMOTED — see correction 1.** Measured on a corpus where 92.2% of anchors
> had a DESI match. On the five-spoke corpus DESI is 6.3% of anchors, so this
> now saves 40% of the spectrum bytes of 6% of rows. Still free, no longer the
> headline.

`_SOURCE_COLUMNS["desi"]` asks for the whole `spectrum` struct, so pyarrow
fetches all four child columns. Parquet stores nested fields as separate
column chunks, so requesting `spectrum.flux`, `spectrum.lambda`,
`spectrum.mask` by path skips `ivar`'s bytes entirely.

- **Saving:** ~41% of spectrum transfer; ~47 KB per matched pair.
- **Risk:** low. Verify pyarrow accepts dotted nested paths in this version;
  if not, read the struct and drop the child before decode (no saving) or
  use `read_row_group(columns=[...])` with explicit leaf paths.
- **Watch:** ADR 0007's spectral normalization does not use `ivar`, but
  confirm no future quality cut wants it before deleting it from the schema
  contract. Projection is reversible; a schema change is not.
- **Effort:** hours.

### 2. Use the pixels we already pay for — up to 2.5x on image bytes

We download 152x152 and tokenise the central 96x96. Two ways to close that:

**(a) Tokenise the full 152x152.** 19x19 patches at patch 8 = 361 tokens per
image, and the config already carries `max_positions: 361`. Useful fraction
goes 39.9% -> 100%, a **2.5x** improvement in bytes per token, at 2.5x the
tokens per object. Sequence budget per object rises, so fewer objects per
packed row — this trades corpus breadth for byte efficiency and is a
**modelling decision, not an optimisation**. It also changes what the model
sees (more sky per object, more background pixels).

**(b) Keep 96x96 but raise the patch size** so the token count stays similar
while covering more area. 152 is not divisible by 16; 8 is the only clean
patch for the full frame.

- **Saving:** up to 2.5x on 92% of the corpus bytes — by far the largest
  single lever.
- **Risk:** medium-high. Changes tokenisation, so checkpoints are
  incompatible and every downstream number (packing utilisation, object
  lengths, probe R^2) moves. Needs its own A/B.
- **Effort:** small code change, large evidence burden.

### 3. Skip partner row groups that contain no wanted ids

`_scan_partners` reads **whole** partner partitions to find a median of 748
matched ids per cell. Parquet row groups carry min/max statistics, and the
match index already knows every wanted id at build time.

- Store the partner **row-group index** alongside each edge in the match
  index (a build-time change to `build_match_index.py`), then read only those
  row groups.
- Or push down a filter on `object_id` if the partitions are sorted by it —
  needs checking, HATS sorts by `_healpix_29`.
- **MEASURED, AND IT IS DEAD.** Every DESI partition is a *single* row group
  (up to 520 MB / 6825 rows), and no page index is written. There is no
  sub-file granularity to skip to. See the appendix. Do not build this.

### 4. HSC images have the same crop waste

`HSC_IMAGE_SHAPE = (5, 160, 160)`, cropped to 96x96 — useful fraction
`96^2/160^2 = 36%`, across 5 bands. Same trade as (2), and HSC's loss is
currently ~10x the legacy image loss, which is worth understanding on its own
before changing its tokenisation.

### 5. Reuse bytes across epochs

Each epoch re-downloads the entire corpus. 181 cells at ~283 KB/object is a
fixed cost paid again every pass. A transient local cache would make epochs
2+ nearly free.

**This is in tension with ADR 0006**, which chose live streaming and deleted
the local reshard, and with ADR 0011's "no local cache". It is not a decision
to take quietly — but at multi-epoch scale it dominates everything else in
this document. Worth an explicit ADR amendment discussion, framed as
*transient cache* (evictable, reconstructible, no schema of its own) rather
than a reshard.

### 6. Drop unused anchor scalars

`_ANCHOR_COLUMNS` already projects, and the non-image columns are <0.1% of
the row group. **Do not bother** — this is the kind of optimisation that
looks productive and buys nothing.

## What does not work — measured today, do not repeat

> **AMENDED — see correction 5.** The prefetch is not merely neutral: at
> `num_loading_workers: 8` on the five-spoke corpus it hangs a rank for 20
> minutes and NCCL kills the run. And all three rows below are 8-worker runs,
> so "within noise" says nothing about worker count, which is a large lever.

Three concurrency changes, compared over an identical window (steps 11-142,
startup excluded, same corpus position):

| variant | mean s/step | median | slow steps |
| --- | ---: | ---: | ---: |
| baseline | 2.88 | 0.64 | 21 (16%) |
| + cell-boundary prefetch thread | 3.20 | 0.63 | 22 (17%) |
| + fsspec `cache_type=background`, pyarrow `pre_buffer` | 3.73 | 0.63 | 22 (17%) |

All three are within noise of each other and of baseline, and the *count* of
slow steps is unchanged. Earlier readings that suggested 12s -> 6s -> 3s were
a sampling artifact: runs of 538, 142 and 65 steps compared against each
other, where the longer run had simply reached more stalls.

The changes are correct (an order-equivalence test pins the prefetch path to
the inline path) but unjustified. **Recommendation: revert all three** unless
a controlled test shows a win — they add a background thread, a byte-capped
buffer, an overflow fallback and two constants for no measured gain.

Note the runs were at different times against a shared hub, so the ordering
between them is not causal. The absence of benefit is the finding.

## Suggested order

**Owner selection 2026-08-06:** Appendix B ideas B (per-band tokens) and D
(AR span-order replay) are the two carried forward. The list below is the
byte-efficiency work that stands independently of them.

1. **Measure** partner row-group selectivity (option 3) — cheap, decides
   whether a medium-effort change is worth anything.
2. **Ship** `spectrum.ivar` projection (option 1) — real saving, low risk,
   no modelling implications.
3. **Decide** on the image crop (option 2) — the biggest lever, but it is a
   modelling change and belongs in an ADR with an A/B, not in a perf pass.
4. **Revert** the three concurrency changes.
5. **Raise** the epoch-cache question (option 5) against ADR 0006 before
   multi-epoch training makes it urgent.

## Caveats

- Column shares come from one row group of one partition per catalogue. Rerun
  across a sample before quoting them as corpus figures.
- Bytes here are *compressed on-disk* sizes, which is what crosses the
  network, but pyarrow also fetches footers and coalesces ranges, so actual
  HTTP bytes will be somewhat higher.
- The ADR 0013 test plan already requires "actual transferred bytes and
  loss-target tokens/byte" as acceptance evidence per spoke. The instrument
  used for the 2026-08-05 spoke smoke (`HfFileSystemFile._fetch_range` byte
  counting) is the right tool to confirm any of this end to end.

---

# Appendix: HATS / Parquet mechanisms — what is actually available to us

Added 2026-08-06 after reviewing the Parquet and HATS/LSDB literature and
then checking what the *published MMU files* actually support. The literature
offers four relevant mechanisms; three of them are unavailable in our data as
published, and the fourth gives a measured 40%.

## The mechanisms, and whether we have them

| mechanism | what it buys | available to us? |
| --- | --- | --- |
| Nested column projection | skip child columns of a struct | **yes — 40% measured** |
| Row-group skipping via statistics | skip whole row groups | **no — 1 row group per file** |
| Page index (column + offset index) | skip pages inside a column chunk | **no — not written** |
| `BYTE_STREAM_SPLIT` encoding | better float compression | **no — PLAIN, needs republication** |

### 1. Nested column projection — the one that works

Parquet stores each leaf of a struct as its own column chunk, so
`spectrum.ivar` can be skipped by requesting sibling paths explicitly.
Measured on `mmu_desi_edr_sv3` partition `(6, 16041)`, counting real payload
bytes through `HfFileSystemFile._fetch_range`:

```
current   (columns=[... 'spectrum' ...])                        60.7 MB   3.2 s
projected (columns=[... 'spectrum.flux','spectrum.lambda',
                        'spectrum.mask' ...])                   36.4 MB   2.7 s
                                                        saving  40%
```

`_SOURCE_COLUMNS["desi"]` and `["sdss"]` both request the whole `spectrum`
struct today. Changing them to leaf paths is a one-line edit per source with
no modelling implications. **This is the recommendation.**

### 2. Row-group skipping — not possible in this data

The literature's standard advice is to use row-group min/max statistics to
skip row groups. Our files defeat it:

```
cell (5,4010) partner (6,16042): 1 row group, 6825 rows, 520 MB
cell (5,4010) partner (6,16043): 1 row group, 5427 rows, 414 MB
cell (5,3839) partner (6,15359): 1 row group, 6731 rows, 513 MB
```

**Every DESI partition is a single row group**, up to 520 MB. There is no
sub-file granularity to exploit: you read the column chunk or you do not.
One anchor cell can require 1.18 GB of partner data across four such files —
that *is* the multi-minute boundary stall, and no amount of prefetching or
concurrency removes it.

The good news is that the ingredients for skipping do exist if the files were
ever rewritten: `sorting_columns` is set (ascending `_healpix_29`) and
`_healpix_29` statistics are present per row group. Matched partners are
within 1 arcsec of their anchor, so their healpix values are nearly
identical — if partitions were written with, say, 64 MB row groups, a cell's
wanted ids would concentrate in a handful of them and skipping would be very
effective. This is an **upstream ask**, not something we can do at read time.

### 3. Page index — not written

`column_index_offset` and `offset_index_offset` are `None` on every column
chunk we checked. Parquet's page index would allow skipping individual data
pages within that single 520 MB column chunk, which is exactly the
granularity we lack. pyarrow can write it (`write_page_index=True`), but the
published files do not carry it.

### 4. `BYTE_STREAM_SPLIT` — the largest upstream ask

`image.flux` and `spectrum.flux` are written `PLAIN` + ZSTD. For float32
arrays, `BYTE_STREAM_SPLIT` regroups bytes by significance before
compression and is the standard encoding for scientific float data —
introduced in Parquet 1.12 precisely for this case, and typically a large
win where PLAIN + a general compressor does poorly. Our numbers show
`image.flux` compressing 277 KB -> 262 KB, i.e. **ZSTD is achieving almost
nothing on PLAIN float noise**, which is the exact regime
`BYTE_STREAM_SPLIT` targets.

We cannot re-encode someone else's published catalogue. But this is a
concrete, well-founded request to the MMU publishers, and it applies to any
derived dataset we ever write ourselves.

### 5. HATS index tables — solve a problem we do not have

LSDB's supplemental index tables give fast object-id -> partition lookup
without a full scan. Our match index already records the partner partition
for every edge, so we never scan to *find* a partition. Index tables would
not reduce bytes read within a partition. Not useful here.

## Conclusion

Within the published layout there is exactly one lever — **nested column
projection, measured at 40% off every spectrum read** — and it should be
taken. Everything else in the Parquet toolbox (row-group skipping, page
index, float encoding) requires the data to be written differently, which
makes it an upstream conversation with the MMU publishers rather than an
optimisation we can apply.

That also settles option 3 in the main document: partner row-group
selectivity is not worth building, because the row groups do not exist.

## Sources

- [Page Index — Apache Parquet](https://parquet.apache.org/docs/file-format/pageindex/)
- [Bloom Filter — Apache Parquet](https://parquet.apache.org/docs/file-format/bloomfilter/)
- [Encodings — Apache Parquet](https://parquet.apache.org/docs/file-format/data-pages/encodings/)
- [Querying Parquet with Millisecond Latency — Apache Arrow](https://arrow.apache.org/blog/2022/12/26/querying-parquet-with-millisecond-latency/)
- [Column Indexes and Bloom Filters — CERN Databases blog](https://db-blog.web.cern.ch/node/194)
- [HATS Catalog Structure and Performance — LSDB](https://docs.lsdb.io/en/latest/data-access/hats.html)
- [Using LSDB to enable large-scale catalog distribution, cross-matching, and analytics (arXiv:2501.02103)](https://arxiv.org/pdf/2501.02103)
- [PARQUET-1622: Add BYTE_STREAM_SPLIT encoding](https://issues.apache.org/jira/browse/PARQUET-1622)
- [Query Engines: Gatekeepers of the Parquet File Format — DuckDB](https://duckdb.org/2025/01/22/parquet-encodings)

---

# Appendix B: saturating useful bits per bit — ideas, with two measurements
that change the answer

## Measurement 1: the discarded periphery is NOT empty sky

I assumed the outer ring of each 152x152 image was mostly background, making
the 96x96 crop nearly free. It is not. Per-patch standard deviation in
arcsinh-normalised units, 40 real North objects, 8x8 patches:

```
96x96 crop  median 0.6246  p10 0.5247  p90 0.8389   n=5760
full 152    median 0.6189  p10 0.3222  p90 0.7651   n=14440
median ratio crop/full = 1.01x
only 12.7% of full-frame patches fall below the crop's own 10th percentile
```

The median peripheral patch carries **the same** pixel variance as the median
central patch. We are not discarding background; we are discarding data of
comparable richness to what we keep.

Caveat: per-patch standardisation later removes each patch's mean and std, so
std is a proxy for "has structure", not a direct measure of learnable signal.
A high-variance pure-noise patch is still irreducible. A sharper proxy
(spatial autocorrelation within a patch, or variance against the per-pixel
noise estimate) would settle it — but the crude version already refutes "the
periphery is empty".

**Consequence: option 2 in the main document is stronger than written.** Full
frame tokenisation is not 2.5x more tokens of diminishing quality; it is 2.5x
more tokens of roughly equal quality, for zero additional bytes.

## Measurement 2: bytes per loss-bearing token, by modality

Combining measured transfer with the token counts each modality produces:

| modality | bytes/object | tokens | **KB per token** |
| --- | ---: | ---: | ---: |
| anchor scalars (`ebv`, photometry) | ~0 (same row) | 2 | **~0** |
| PROVABGS scalars | small (scalar columns only) | 5 | **~0** |
| images, full 152 frame | 262 KB | 361 | **0.73** |
| DESI spectra, ivar projected | ~36 KB | 31 | **1.15** |
| images, 96x96 crop (today) | 262 KB | 144 | **1.82** |
| DESI spectra, as fetched today | ~60 KB | 31 | **1.93** |
| HSC images, 96x96 crop (today) | ~512 KB | 144 | **3.56** |

This reframes modality selection as an economic decision. Scalars are free
tokens — we already pay for the row. HSC is the most expensive token in the
corpus by a factor of five, which is worth weighing against it appearing in
only ~5-17% of batches and carrying ~10x the legacy image loss.

## Ideas, strongest first

### A. Foveated tokenisation — 100% pixel coverage at ~1.4x tokens

Full-frame at patch 8 is 361 tokens (2.5x). But the target galaxy is centred
by construction, so resolution need not be uniform: keep patch 8 over the
central 96x96 (144 tokens) and tile the surrounding annulus at patch 16
(~54 tokens). Total ~198 tokens for **all** the pixels — 1.4x the tokens for
2.5x the coverage, i.e. **1.8x better bytes/token than today** without the
full 2.5x sequence-length cost.

Requires per-token patch-size metadata, which the modality registry does not
currently carry. Astronomically natural: fine detail on the source, context
resolution for neighbours.

### B. Per-band tokens — 3x tokens, same bytes  **[SELECTED 2026-08-06]**

A token today is 8x8x3 = 192 floats. Splitting bands gives 8x8x1 = 64-float
tokens, 3x as many, from identical bytes. More loss terms per downloaded
byte, and the model learns cross-band structure autoregressively rather than
having it pre-fused. Costs sequence length; interacts with the jetformer flow
dimension. Cheap to try as an A/B.

### C. Multi-view per download — amortise the byte cost across objects

One downloaded image currently yields one training object. Emitting several
deterministic views (centre crop, full-frame downsample, corner crops) turns
one 262 KB download into 2-4 objects. Not new information, but more gradient
per byte, and standard practice in SSL multi-crop.
Risk: correlated samples within a batch, and for an exact-likelihood
objective, near-duplicate content invites memorisation. Needs an
epoch-seeded view choice so a given object is not repeated identically.

### D. Replay while stalled — free steps out of the boundary stall  **[BUILT 2026-08-06 — see correction 6]**

The GPU is idle for minutes at each cell boundary while 1.18 GB arrives. The
previous cell's decoded objects are still in memory. Replaying them with a
different span order (ADR 0008's shuffle is already epoch-seeded, so this is
a supported variation) converts dead GPU time into gradient steps at **zero**
additional bytes.
This is the one idea that exploits the stall rather than fighting it — and
the three failed concurrency attempts suggest fighting it is not working.
Care needed: replayed objects must not corrupt the exactly-once/no-replay
audit, so they would need to be logged distinctly or excluded from the epoch
accounting.

### E. Schedule by match density

Spokes per anchor across the current index: 1 spoke x 37,502 anchors,
2 x 107,988, 3 x 4,001. An anchor with three spokes yields far more tokens
for the same anchor image bytes. Preferentially scheduling dense cells early
(or weighting them) raises tokens per byte — but it biases the corpus, so it
is a curriculum decision requiring evidence, not a free win.

### F. Make the cell the unit of work

A cell costs ~1.18 GB whatever we do with it. Today we stream through it once
and move on. Treating the cell as a unit — decode once, then take many steps
against everything in it before evicting — maximises value per expensive
fetch. This is idea D generalised, and it argues for a data-ordering redesign
rather than more prefetching.

## Ideas considered and rejected

- **Filtering empty-sky patches.** Refuted by measurement 1: patches are not
  noise-only, and the periphery is as rich as the centre.
- **Row-group / page skipping.** Refuted by Appendix A: one row group per
  file, no page index.
- **HATS index tables.** Solve partition *lookup*, which the match index
  already gives us.
- **Harder column projection on anchors.** Non-image columns are <0.1%.

---

# Appendix C: we re-download 39% of partner bytes, and it is the cell order's fault

> **NUMBERS SUPERSEDED — see correction 3.** Every count and table here is
> against the 181-cell three-spoke index; the corpus is now 5,488 cells and
> these need re-measuring with `_fetch_range` byte accounting. Also see
> correction 4: the suggested 8-16 cell blocks are SMALLER than the 16
> consumers of a dp=2 x 8-worker run and would buy nothing.

Measured on the current three-spoke index. This is not about using fetched
bytes better — it is about not fetching the same bytes twice.

## The waste

```
partner-partition references   434
distinct partitions            266
duplication                    1.63x  =>  39% of partner fetches are re-downloads
```

| source | refs | distinct | duplication |
| --- | ---: | ---: | ---: |
| desi | 191 | 161 | 1.19x |
| hsc | 72 | 69 | 1.04x |
| provabgs | 171 | 36 | **4.75x** |

PROVABGS partitions are order 4 while anchors are order 5-6, so one PROVABGS
partition serves many anchor cells — up to **11** re-fetches of the same file.
Its columns are small scalars so the byte cost is modest; DESI's 1.19x on
80-520 MB partitions is where the bytes actually are.

## The cause is the epoch shuffle, not the fetching

A per-worker LRU cache of decoded partner partitions, simulated against the
real index:

```
                          LRU2  LRU4  LRU8  LRU16      (floor = 266)
epoch shuffle (today)      430   427   424    404
healpix sorted             291   278   272    272
greedy partition overlap   290   278   267    266
```

Under today's order a cache is **useless** — 16 slots recover 7%. Under
healpix order a **2-slot** cache gets within 10% of the theoretical floor.

Spatially adjacent anchor cells share partner partitions; that spatial
coherence is the whole point of HATS, and `shuffled(in_split, seed, epoch)`
destroys it every epoch. We pay ~1.6x for partner data to obtain shuffling.

## The fix: shuffle blocks, not cells

Keep short runs of spatially contiguous cells intact and shuffle the runs:

```
block size   LRU2  LRU4  LRU8      (shuffle today = 424, floor = 266)
     1        428   420   414
     2        361   357   349
     4        328   317   309
     8        309   298   292
    16        297   286   280
    32        293   282   276
   181        291   278   272   (fully sorted, no shuffle)
```

**Blocks of 8-16 cells with a 2-8 partition LRU recovers 83-91% of the
available saving while still shuffling 11-23 independent blocks per epoch.**
Partner fetches drop 424 -> ~280-292, about **a third fewer partner-partition
downloads**, for a cache that is trivially small in slot count.

## What it costs

- **Record order changes**, so `SOURCE_GRAPH_ASSEMBLY` must be bumped and
  stale stream states will be rejected on resume (weights still load). This
  is the documented, intended mechanism for order changes.
- **Cache memory is in bytes, not slots.** A projected DESI partition is
  ~36 KB/row x ~6800 rows = ~245 MB, so LRU2 is ~0.5 GB per worker and 8
  workers is ~4 GB. Size the cache in bytes with eviction, like
  `UNMATCHED_BUFFER_BYTES`, not in partition count. Note LRU2 already
  captures most of the benefit under block order, so the cache stays small.
- **Shuffling quality.** Within a block the order is spatial, so objects in
  nearby packed rows are spatially correlated. Packing already puts ~340
  objects from a single cell into one row, so the marginal correlation from
  8-cell blocks is small — but it is a training-dynamics change and wants an
  A/B, not an assumption.
- Interacts with `num_loading_workers`: cells are dealt round-robin to
  workers, so a block should ideally land on one worker. Dealing *blocks*
  rather than cells to workers would preserve that.

## Why this is worth more than it looks

Partner fetches are the multi-minute cell-boundary stalls — one cell can pull
1.18 GB. Cutting a third of those fetches attacks exactly the thing that
three rounds of concurrency work failed to move, and unlike prefetching it
reduces the bytes rather than rearranging when they arrive.

---

# Appendix D: are unmatched rows in pulled healpixes worth training on?

**We already do, and they are the majority of trained objects.** The question
is whether to keep them, not whether to add them.

## Realized composition, from the no-replay audit logs

> Three-spoke figures. On the five-spoke corpus HSC is 0.7% of anchors; the
> variance concern below stands. See correction 8.

| run | anchor records | DESI-only | HSC-only |
| --- | ---: | ---: | ---: |
| w8, 100 steps | 30,858 (42.8%) | 41,097 (57.0%) | 97 (0.1%) |
| 1k, 538 steps | 125,808 (38.2%) | 149,788 (45.4%) | 54,064 (16.4%) |

By **tokens** rather than objects (1k run; estimate reconciles with actual
`consumed_tokens` to 0.6%, 35.1M vs 35.3M):

| | tokens | share |
| --- | ---: | ---: |
| anchor (image + mostly spectrum) | 22.3M | 63.7% |
| DESI-only (spectrum) | 4.9M | 14.1% |
| HSC-only (image) | 7.8M | 22.2% |

**36.3% of all training tokens come from rows we never needed to fetch for
their own sake** — the partitions were pulled for their matched rows, and
these came along at zero marginal byte cost.

## The argument for keeping them is stronger than "they are free"

We are bandwidth-bound with the GPU idle roughly 95% of the time. A token
derived from already-fetched bytes consumes the resource we have in surplus
(compute) and none of the resource we are short of (bandwidth). Dropping
unmatched rows would not speed up training at all — the stall is the fetch,
not the step — it would simply train on less data for the same wall clock.

So: **keep them.** ADR 0005 and ADR 0011 already decided this, and the
economics now measured support it.

## The real problem is not the ratio, it is the variance

HSC was **0.1%** of objects in one run and **16.4%** in another. A 160x swing,
from nothing but which cells the run happened to visit — HSC's matches live in
25 of 181 cells, so a short run either hits them or does not.

That is a far bigger threat to training than the matched/unmatched ratio:

- per-family loss weights (1:1:0.1) normalise *within a batch*, so a run of
  spectrum-only batches trains no image head at all for that stretch;
- the earlier observation that `hsc_images_loss` was zero in 95 of 100 steps
  is this same effect, and it means short runs cannot evidence a spoke;
- **Appendix C's block ordering makes this worse**, since spatially
  contiguous blocks are compositionally correlated by construction.

Worth measuring before either change lands: composition per 100-step window
across a long run, to see how far the mix swings and whether any modality
starves for long stretches. That is cheap — the `object_id_log` already
records it, which is how this appendix was written.

## What is genuinely discarded today

Not much. Every row of a fetched partner partition is either matched (emitted
as a pair by its own anchor cell) or globally unmatched (emitted by the single
owning cell). The only true loss is a fetched partition whose referencing
cells are all in the *other* split — `_partition_owner` returns no owner and
its unmatched rows are dropped rather than triggering another scan. Worth
counting, but likely small.

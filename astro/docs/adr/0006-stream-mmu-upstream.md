# ADR 0006: Stream MMU upstream instead of pre-resharding local parquet

- **Status:** Implemented (accepted 2026-07-17)
- **Date:** 2026-07-17
- **References:**
  - `astro/PLAN.md` "Data pipeline" — the local-shard + `HF_DATASETS_OFFLINE=1`
    design this ADR **supersedes**
  - `astro/src/astropt3/data/streaming.py` — the implementation
  - `astro/scripts/prepare_pilot_data.py` — the lsdb crossmatch → parquet
    reshard, deleted (its `row_to_record` moved into `streaming.py`)
  - `astro/src/astropt3/data/mmu.py` — `PILOT_FEATURES`, `MMUIterableDataset`,
    `assign_split` being deleted
  - `astro/src/astropt3/data/nanotron_loader.py` — `PackedMicroBatches`, rewired
    to stream
  - MMU's own streaming-crossmatch workflow:
    <https://huggingface.co/blog/hugging-science/multimodal-universe-hats>

## Question

The training corpus is built by lsdb-crossmatching two Multimodal Universe
(MMU) HATS catalogs on a login node and **rewriting** the result into a
bespoke `PILOT_FEATURES` parquet schema, which the compute nodes then stream
back offline. That custom schema is a second, hand-maintained source of truth
that drifts from what MMU actually publishes, and onboarding any new dataset
means editing schema code in three places (`mmu.py`, `synthetic.py`, the
crossmatch). Can the loader instead consume MMU **in its native shape**, so we
stop maintaining a parallel schema and adding a dataset becomes configuration,
not schema code?

## Context

- MMU publishes each survey as a HATS collection on the HF hub and provides
  crossmatch utilities; there is **no single pre-joined images×spectra
  release** — you crossmatch yourself. The endorsed way is lazy and streamed:
  `lsdb.open_catalog("hf://datasets/UniverseTBD/mmu_*")` +
  `.crossmatch(other, n_neighbors=1)` builds a task graph, and `CatalogStream`
  yields **one HEALPix partition at a time with background prefetch**, loading
  only overlapping tiles (~4 GB RAM for the whole join). **Internet is required
  at read time.**
- Today's pipeline (`prepare_pilot_data.py`) materializes that join into
  `PILOT_FEATURES` shards under `{root}/{train,val}/`; `MMUIterableDataset`
  streams them back with `HF_DATASETS_OFFLINE=1`; `synthetic.py` mirrors the
  same record shape so everything runs offline in CPU tests.
- The pilot trains exactly **one** corpus (LegacySurvey × DESI). There is no
  second corpus to mix yet.
- Checkpoint-resume today is exact ("never replays a trained record"), built
  entirely on HF `datasets`' stateful streaming iterator — the machinery this
  ADR deletes.
- Evaluation demands a **fixed, deterministic** val set: `eval/val_loss` reuses
  fixed val batches and the probe sweep re-evaluates it on every checkpoint.
- `synthetic.py` and the frozen contracts (special-token ids in
  `tokenization.py`, the `ObjectSequencer`/`PackedCollator` packing contract)
  are load-bearing and must not move.

## Decision drivers

- One schema, MMU's — no hand-maintained parallel `PILOT_FEATURES`.
- Adding a dataset should be config, not schema code.
- Keep CPU tests **offline and lsdb-free** (a hard repo constraint) even though
  training now needs the network.
- Don't build a multi-source mixing engine for a repo that trains one corpus.
- The frozen tokenization/packing contracts and `synthetic.py` stay untouched.

## Decision

**Stream the MMU catalogs directly at train time via lsdb/HATS + `CatalogStream`;
delete the local reshard and its custom schema entirely.** The compute nodes now
have internet — this knowingly supersedes PLAN's offline-node/local-shard data
design.

1. **Corpus = three interleaved streams**, all streamed live:
   - **images-only** — ~14M LegacySurvey images as single-modality image
     sequences,
   - **spectra-only** — ~1.1M DESI spectra as single-modality spectrum
     sequences,
   - **crossmatch (inner)** — ~0.7M matched pairs as multimodal image+spectrum
     sequences. The pairs are **precomputed offline into a published
     match-index** (see Performance); this stream reads native image + spectrum
     partitions and joins them on the index in memory — no live spatial join.

   The join flips `how="left"` → `how="inner"`: image-only coverage now comes
   from the standalone images stream, so the join carries pairs only. A matched
   object's image appears both standalone (stream 1) and paired (stream 3); this
   redundancy is inherent to combining MMU datasets and is accepted.

2. **Interleave by fixed provisional per-record weights** — **0.60 / 0.15 /
   0.25** (images-only / spectra-only / pairs). Images stay dominant (bulk,
   cleanest signal); pairs are up-weighted ~5× above their ~5% natural share
   because cross-modal learning is the point; spectra sit between so the
   spectrum head sees standalone signal without ~13× over-memorizing the small
   corpus. **These weights are provisional** — tuning them is the mixing issue
   below.

3. **Three sources hardcoded**, no config registry. General, weighted,
   config-driven multi-source mixing is deferred to a **GitHub issue**, to be
   built the day there is a second corpus to mix.

4. **Resume is partition-granularity.** Per-source cursor
   `(epoch, partition_index)` over the deterministic, per-epoch-shuffled
   partition order; on resume, reopen the native catalogs and reload the
   match-index (cheap; no spatial join to rebuild), skip whole partitions up to
   `partition_index` (no download), and restart the in-flight partition from its
   beginning. This **replays ≤ 1 partition of already-trained
   records per kill** (a partition is thousands of objects). The Phase 4
   kill/resume gate and the `object_id_log` audit are rewritten from "no replay"
   to **"no replay beyond the in-flight partition."** The weighted-sampler RNG
   state is part of the saved cursor so the source draw sequence resumes
   identically.

5. **Val = reserved partitions.** A fixed set of the first **K** HEALPix
   partitions (chosen to overlap the spectra catalog's coverage, so val contains
   matched pairs and the redshift linear-probe has `Z` labels) is reserved for
   validation and excluded from train. Val is streamed each eval — only K
   partitions, not the whole catalog — keeping it fixed and spatially disjoint
   (whole partitions ⇒ no leakage). Replaces `assign_split`'s hash-by-tile
   split.

6. **Float to latest upstream revision** (no pinning). This knowingly softens
   the reproducibility goal: if MMU pushes a new revision mid-project the corpus
   changes silently. Mitigation left open (see Open issues): logging each
   catalog's resolved revision at run start costs nothing and restores
   after-the-fact traceability without config upkeep.

7. **Schema demolition.** Delete `PILOT_FEATURES`, `normalize_record`,
   `write_shard`, `decode_record`, `assign_split`, `MMUIterableDataset`,
   `prepare_pilot_data.py`, `check_pilot_data.py`. The one thing kept is the
   lsdb-row → record-dict logic (`row_to_record`), which **moves into the
   streaming loader as the sole decode adapter** over MMU-native fields.
   `ObjectSequencer`, `PackedCollator`, `tokenization.py`, `band_registry.py`,
   `transforms.py` are untouched. `synthetic.py` is unchanged and remains the
   offline test source via the `data_root == "synthetic"` sentinel. "Kill the
   custom schema" is satisfied: there is no longer a second hand-curated Arrow
   schema distinct from what MMU publishes — the loader consumes native rows and
   adapts minimally.

8. **lsdb is offline-only; train-time streaming is `pyarrow` + `hats`.** lsdb
   runs only in the prep env, to compute the match-index once; no train-time
   path needs its dask/KD-tree machinery. `pyarrow`/`hats` are lazy-imported on
   the streaming path only (behind the `synthetic` sentinel), so the CPU suite
   stays offline; they — not lsdb — move into the nanotron training env's
   dependencies. lsdb stays in the `[data]`/prep extra.

9. **Adapter test = checked-in cassette.** A handful of real crossmatch rows are
   captured once (online, login node) into a small repo fixture and replayed
   offline through `row_to_record` in a CPU test — covering the non-trivial
   decode (struct coercion, `_desi` suffix handling, null handling) without
   lsdb/network, and doubling as documentation of the real upstream row shape.

10. **Sharding/shuffle mechanism.** Partition-level DP/worker split: assign the
    per-epoch-shuffled partition index across `(world_size × num_workers)` by
    modulo — deterministic, disjoint, no cross-rank coordination. Per-epoch
    seeded partition-order shuffle (identical across ranks, so the modulo split
    stays disjoint) plus a per-worker buffer shuffle break the spatial
    clustering of HEALPix-ordered partitions. Because the stream is endless,
    uneven partition sizes never starve a rank at the optimizer all-reduce — each
    rank always has another partition to draw.

## Performance

Streaming must sustain **≥2× training-consumption per rank** so the GPU never
starves; beyond that floor, throughput work stops ("as fast as possible" is not
the target). The 75 obj/s of the old serial `prepare_pilot_data.py` prep is
**not** that number — training runs `DP × num_workers` (~256 at pilot scale)
loader processes, so the naive live path may already clear the floor by worker
parallelism, or 256 concurrent engines may exhaust RAM. **A profiling spike
settles it before anything is built** (see Open issues): one live worker over
~5 real partitions, coarse timers on `partition read` vs `row decode` vs
`tokenize`, so the optimization is aimed at the measured bottleneck rather than
a guessed one.

Three refinements shape the streaming path — chosen so they also survive scale
(a 62 TB Legacy South imaging corpus grows the *images-only* stream; the
crossmatch stays bounded by the smaller spectroscopic side, so imaging volume
never inflates it):

- **Precompute the crossmatch into a published match-index, not joined
  records.** `lsdb.crossmatch` runs offline, once; the artifact published to HF
  is only `(image_healpix, image_id, spectrum_healpix, spectrum_id,
  dist_arcsec)` — ~0.7M rows, tens of MB, **no pixels duplicated** (joined
  records would be ~280 GB of images already hosted in the LegacySurvey
  catalog, and a second bespoke schema). The index is bounded by matched pairs,
  so it survives arbitrarily large imaging corpora unchanged. This keeps the
  "one native schema" win: the published artifact is pointers into MMU's data,
  not a copy of it.
- **lsdb offline-only; train-time streaming is `pyarrow` + `hats`.** Once the
  join is an id-dict lookup, no train-time path needs lsdb's dask/KD-tree
  engine: `hats` enumerates partitions + HEALPix order + `hf://` resolution,
  `pyarrow`/`fsspec` stream partition files, and matches resolve through the
  in-memory index. Removes the 256-dask-scheduler feasibility risk and drops a
  distributed-join engine from a path that only reads files sequentially.
- **Optimize the shared decode, images-first.** The 60% images-only stream does
  no join and shares its decode/tokenize path with the other two, so a decode
  win lifts all three; it is also the majority of every batch and the binding
  throughput constraint. The decode (currently per-row `iterrows`/struct
  coercion) becomes columnar per-partition — bulk pyarrow→numpy, sliced into
  per-object records at the frozen `ObjectSequencer` boundary — contingent on
  the flux column being fixed-shape, which the spike confirms. `ObjectSequencer`
  / `PackedCollator` / `tokenization.py` are untouched.

Prefetch is the existing DataLoader worker queue (`prefetch_factor` /
`num_workers`); no custom prefetcher is built. Network-stall resilience and
cross-epoch local partition caching are **out of scope** for the sub-epoch
pilot — they live under the "network is a hard dependency" negative below, not
the speed work.

## Options considered

### Option A — Native-schema local shards (keep the reshard, mirror MMU)

Keep `prepare_pilot_data.py` but make its output schema the faithful union of
MMU's two native schemas instead of a curated subset. **Rejected:** still a
second on-disk schema to maintain and still a login-node prep step; it trims the
drift but not the maintenance surface, and it keeps the whole offline-shard
machinery this ADR is trying to delete.

### Option B — Stream upstream at train time (chosen)

Drop local shards; stream the crossmatch live with lsdb/`CatalogStream`.
**Chosen:** removes the parallel schema and the prep step entirely, consumes MMU
exactly as published, and is MMU's own recommended workflow. Costs the offline
guarantee and exact resume (see Consequences), both accepted.

### Option C — Multi-source mixing engine now

Build the general N-dataset, weighted, config-driven mixing loader immediately.
**Rejected as premature:** the pilot trains one corpus; sampling weights,
interleave, and per-source modality routing are speculative machinery. Deferred
to a GitHub issue; the interim is three hardcoded sources with fixed weights.

### Option D — Status quo (local reshard, offline nodes)

Keep PLAN's design. **Rejected:** it is exactly the hand-maintained parallel
schema and three-place dataset-onboarding this ADR removes.

## Consequences

### Positive

- One schema — MMU's. No `PILOT_FEATURES`, no `normalize_record`/`write_shard`,
  no login-node reshard step; a large, drift-prone slice of the data layer is
  deleted outright.
- Adding a dataset (once mixing lands) is pointing at an `hf://` catalog, not
  editing schema code in three files.
- Single-modality *and* paired records both fall out of the same live stream
  set; the corpus mirrors how MMU is actually published.
- CPU tests stay offline and lsdb-free (lazy import behind the synthetic
  sentinel); the frozen tokenization/packing contracts and `synthetic.py` are
  untouched.

### Negative / tradeoffs

- **Network is a hard training dependency.** HF hub downtime or rate-limiting
  stalls training with no local fallback — the deliberate price of
  streaming-only.
- **Resume replays ≤ 1 partition per kill** (thousands of objects); the exact
  no-replay guarantee and its Phase 4 gate are relaxed accordingly.
- **Reproducibility softened** by floating revisions — the corpus can change
  under a long run if MMU pushes upstream.
- **`pyarrow`/`hats` become training-env dependencies** (lsdb stays offline).
  That `hats` alone — without lsdb — exposes deterministic partition
  enumeration, HEALPix order, and `hf://` resolution is an **implementation
  risk** assumed here, not yet verified in code (see Open issues).
- Interim fixed weights (0.60/0.15/0.25) are a guess; the small corpora
  (spectra, pairs) still carry over-memorization risk until the mixing issue
  tunes them.

## Open issues

- **Multi-source weighted mixing** (GitHub issue): general, config-driven source
  list with tunable sampling weights, interleave, and per-source modality
  routing — supersedes the hardcoded three sources and the provisional weights.
- **Profiling spike (do first):** one live `pyarrow`/`hats` worker over ~5 real
  partitions, timing `partition read` vs `row decode` vs `tokenize`, and
  checking whether ~256 concurrent workers are RAM-feasible. Confirms whether
  worker parallelism alone clears the floor and which path (if any) needs the
  columnar decode — nothing else is built until this runs.
- **`hats`-without-lsdb partition API:** verify `hats` alone exposes
  deterministic partition enumeration, HEALPix order, and `hf://` resolution
  (the pieces of the HATS abstraction the train-time stream needs). If it does
  not, fall back to lsdb-at-train-time or reimplement partition discovery — the
  latter being the worse option.
- **Match-index build:** compute the crossmatch once with lsdb and publish
  `(image_healpix, image_id, spectrum_healpix, spectrum_id, dist_arcsec)` to HF;
  confirm matches for an image in partition `P` fall in the same/adjacent
  spectrum partition so the in-memory join stays partition-local.
- **Revision logging:** cheap run-start capture of each catalog's resolved HF
  revision into a `provenance.json`, restoring after-the-fact traceability under
  the float-to-latest decision.
- **Val coverage:** confirm the reserved K partitions actually contain enough
  matched pairs for a stable redshift probe.

## References

- `astro/src/astropt3/data/nanotron_loader.py` — `PackedMicroBatches` (rewired
  to three weighted `pyarrow`/`hats` partition streams + match-index join),
  per-source partition cursors, resume.
- `astro/src/astropt3/data/mmu.py` — `row_to_record` (moved in), streaming
  crossmatch dataset; deletions listed above.
- `astro/src/astropt3/data/synthetic.py` — unchanged offline record source.
- MMU streaming-crossmatch blog (above) — the endorsed lsdb/`CatalogStream`
  workflow.
- [ADR 0003](0003-checkpoint-samples-in-eval-sidecar.md) — house format
  precedent.

## Implementation notes (2026-07-20)

Implemented in `astro/src/astropt3/data/streaming.py`. Three decisions above
changed once the code met the real API; the rest landed as written.

1. **Weights apply per record, not per partition draw** (§2). Partitions hold
   wildly unequal row counts — measured on the real catalogs: ~5100 rows in a
   DESI partition, ~2447 in a LegacySurvey one, ~290 in a crossmatch one — so
   drawing whole partitions by weight would realize a corpus mix nothing like
   0.60/0.15/0.25. Each source buffers its current partition and only fetches
   the next when that buffer drains.

2. **Resume is exact; the ≤1-partition replay budget was not needed** (§4).
   Because a source buffers its partition whole, the row offset into it is
   checkpointable, so resume re-fetches the in-flight partition but re-trains
   none of it. The Phase 4 no-replay gate and the `object_id_log` audit stand
   unchanged. The saved state is `(draw, per-source (epoch, cursor,
   row_off))` — all ints. There is **no sampler RNG state in the checkpoint**:
   the source draw order is a fixed repeating length-20 pattern derived from
   the weights, not a random draw.

3. **No cassette fixture** (§9). The CPU suite may now use the network, so a
   `network`-marked test decodes real hub rows end-to-end instead. A cassette
   would have been a second artifact to capture, refresh, and let drift. The
   cursor logic is covered offline by injecting fake fetchers behind the
   `Source` API (`tests/fake_mmu.py`), which is faster and deterministic.

Resolved open issues:

- **`CatalogStream` resume API** — it exposes no cursor and no
  skip-to-partition, but `_delayed_partitions` is a plain list indexed by
  partition index and `submit_next_partitions` takes explicit indices, so the
  partition order is ours to own. `CatalogStream` is constructed only for its
  culled per-partition dask graphs; its own iterator is never used.
- **Val coverage** — the pairs source is an inner join, so every reserved val
  partition of it carries matched pairs with `Z`. K is
  `streaming.VAL_PARTITIONS`.

Still open:

- **Prefetch** requires a dask client (`open_sources(client=...)`); without
  one, partition fetches are synchronous and block the training loop.
- **Sharding ceiling**: the pairs source has ~200 partitions, so
  `world_size × num_workers > 200` raises rather than starving a rank.
- **Revision logging** — unchanged from above, still worth doing. Note that a
  suspected mid-project drift was investigated and was a measurement error
  (`images.npartitions` = 5488 vs the LEFT-crossmatch's 5596 partitions);
  upstream had not moved.

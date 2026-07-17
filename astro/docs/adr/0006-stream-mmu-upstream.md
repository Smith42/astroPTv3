# ADR 0006: Stream MMU upstream instead of pre-resharding local parquet

- **Status:** Accepted
- **Date:** 2026-07-17
- **References:**
  - `astro/PLAN.md` "Data pipeline" — the local-shard + `HF_DATASETS_OFFLINE=1`
    design this ADR **supersedes**
  - `astro/scripts/prepare_pilot_data.py` — the lsdb crossmatch → parquet
    reshard being deleted (its `row_to_record` is the one thing kept)
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
     sequences.

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
   partition order; on resume, rebuild the lazy crossmatch (cheap), skip whole
   partitions up to `partition_index` (no download), and restart the in-flight
   partition from its beginning. This **replays ≤ 1 partition of already-trained
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

8. **lsdb/hats are lazy-imported** on the streaming path only (behind the
   `synthetic` sentinel), so the CPU suite stays offline and lsdb-free. They
   move from the `[data]` extra into the nanotron training env's dependencies.

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
- **lsdb/hats become training-env dependencies**, and `CatalogStream`'s
  resume-relevant API (deterministic partition iteration + skip-to-partition) is
  an **implementation risk** assumed here, not yet verified in code.
- Interim fixed weights (0.60/0.15/0.25) are a guess; the small corpora
  (spectra, pairs) still carry over-memorization risk until the mixing issue
  tunes them.

## Open issues

- **Multi-source weighted mixing** (GitHub issue): general, config-driven source
  list with tunable sampling weights, interleave, and per-source modality
  routing — supersedes the hardcoded three sources and the provisional weights.
- **`CatalogStream` resume API:** verify it exposes deterministic partition
  order and a skip-to-partition path; if not, the partition cursor needs a
  different anchor (e.g. re-deriving `get_ordered_healpix_pixels()` and driving
  `.partitions[i]` directly).
- **Revision logging:** cheap run-start capture of each catalog's resolved HF
  revision into a `provenance.json`, restoring after-the-fact traceability under
  the float-to-latest decision.
- **Val coverage:** confirm the reserved K partitions actually contain enough
  matched pairs for a stable redshift probe.

## References

- `astro/src/astropt3/data/nanotron_loader.py` — `PackedMicroBatches` (rewired
  to three weighted `CatalogStream`s), per-source partition cursors, resume.
- `astro/src/astropt3/data/mmu.py` — `row_to_record` (moved in), streaming
  crossmatch dataset; deletions listed above.
- `astro/src/astropt3/data/synthetic.py` — unchanged offline record source.
- MMU streaming-crossmatch blog (above) — the endorsed lsdb/`CatalogStream`
  workflow.
- [ADR 0003](0003-checkpoint-samples-in-eval-sidecar.md) — house format
  precedent.

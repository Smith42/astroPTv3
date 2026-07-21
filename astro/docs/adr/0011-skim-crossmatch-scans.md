# ADR 0011: Skim single-modality records from the crossmatch scans

- **Status:** **Adopted** (2026-07-21) — the A/B on DeltaAI passed (matched
  445-iteration fresh-start window vs `astropt3-70m-jetformer-mmu`: stalls
  ≥5 s 11.0% vs 12.1%, mean step 2897 vs 3178 ms, plus the structural halving
  of hub bytes), and the skim is now the ONLY assembly whenever a
  `match_index` is present: the `skim_images` flag and the 3-source
  interleave were removed. Pre-skim stream states (e.g. the
  `astropt3-70m-jetformer-mmu` baseline's checkpoints) can no longer resume
  their stream position — the loader rejects them with a clear error; weights
  remain loadable. Amended 2026-07-21 to **relax the skimmed train stream to
  coarse determinism** (partition-level, not micro-batch) — see §Determinism
  model.
- **Date:** 2026-07-21
- **References:**
  - [ADR 0006](0006-stream-mmu-upstream.md) — the streamed three-source
    corpus this modifies
  - `astro/src/astropt3/data/streaming.py` — `_paired_examples`,
    `interleaved`, `open_stream`
  - `astro/docs/2026-07-21-streaming-throughput-audit.md` — measured wire
    costs, NIC ceiling
  - `astro/docs/2026-07-21-streaming-shakeout-handoff.md` — why the loader
    is the way it is

## Question

The trainer's 1 GbE NIC is the measured binding constraint (RX bursts 985
Mbit/s = 98.5% of line rate, averages 70–90%; 6 loading workers — 8 pins
the link flat and starves SSH). Under `all_exhausted`, the pairs source
re-scans the matched image partitions ~7–8× per epoch, and every scan
downloads ~1/d image rows per pair served (d = match density) while
**discarding the unmatched ones**. Can those unmatched rows serve the
images-only (and spectra-only) draws instead — eliminating the standalone
catalog downloads — with no extra bandwidth and **no local
materialization**?

## Context

- The corpus is three live streams interleaved per record at fixed weights
  **0.60 / 0.15 / 0.25** (images / spectra / pairs), `all_exhausted`
  (ADR 0006). Pairs are upweighted ~5× over their natural share because
  cross-modal learning is the point.
- **Conservation law.** At 25% pair share and ~0.7M pairs vs ~5.4M pair
  draws per epoch, each pair must be *served* ~7–8× per epoch. Without
  local storage, each serve costs a fresh scan of its partition. The
  re-scans are forced; the only choice is whether their unmatched rows
  (downloaded anyway) are used or dropped. Today: dropped.
- **No row-level filtering over HTTP.** Parquet's fetch granularity is the
  column chunk; at any plausible d every ~200-row row group contains
  matches, so no row group can be skipped. The throwaway cannot be
  filtered away — only used.
- **Column projection is dead** (audit, measured live): `image.flux` is
  50 MB of every 56 MB row group; noisy float32, incompressible; the
  remaining columns are scalars. ~250 KB per image on the wire.
- **Collator packing is a minor lever.** Object lengths ~148–190 tokens
  into 4096-token rows, greedy first-fit ⇒ ~2–4% pad. Demoted.
- **No local materialization** (user constraint, 2026-07-21): no disk
  cache of matched partitions, no cache-as-you-go pairs shard. This rules
  out the only levers that touch the dominant re-scan term.
- Local materialization is also distinct from ADR 0006's rejected ~280 GB
  **hosted** pairs shard — it was on the table here on different merits
  and is excluded by constraint, not by analysis.

## The byte model (parametric — read before quoting any gain)

Let d = match density (matched rows ÷ rows scanned in matched partitions),
~250 KB/image-row, ~100 KB/spectrum-row on the wire, weights 0.60/0.15/0.25.

- **Today, per training record:**
  `0.60×250 + 0.15×100 + 0.25×(250/d + spectrum-side)` KB.
  The pairs re-scan term dominates at small d (625 KB/record at d = 10%).
- **Skimming eliminates the first term** (the standalone images download)
  when discard supply ≥ demand: `(1−d)/d ≥ 0.60/0.25 = 2.4 ⟺ d ≤ 29%`.
  Plus part of the spectra term, symmetrically.
- **Realistic saving ≈ the images-source share of epoch bytes ≈ 20–40%
  for d ∈ [5%, 25%] — i.e. ~1.2–1.6× records/sec at a saturated link.**
  The ~3–4× figure floated during design was an artifact of an
  inconsistent partition count (below) and is **retracted**. Skimming does
  NOT reduce the dominant pairs re-scan term; nothing without local
  storage does.

### Unresolved numeric inconsistency (must be measured)

Repo notes variously imply ~200 matched partitions (ADR 0006 — but that
was the lsdb inner-crossmatch's **own output** partition count, not
matched image cells), 165 train cells (`streaming.aligned` docstring),
~0.7M pairs, ~170–774 MB and ~680–3,100 rows per image partition. These
cannot all hold (they imply d > 100%). **d and the matched-cell count are
unknown within ~2× and the saving scales with d.** One pass over the
match index settles it; it is part of the validation gate.

## Decision (in progress)

Pursue the **demultiplexed scan**: one generator per worker walks the
matched partitions once and emits records at the **nominal internal ratio
12 images : 5 pairs : 3 spectra** — where "images" (resp. "spectra") are the
*unmatched* rows skimmed from the scan, and pairs are the matched rows merged
in memory (existing `_paired_examples` logic). The draw is a **weighted
sampler**, not a fixed repeating pattern: the skimmed train stream is only
**coarsely deterministic** (§Determinism model), which is what lets a sampler
replace the pattern.

1. **Single generator, no cross-process sharing.** Folding the draw into one
   generator sidesteps the rejected alternative — two logical
   `IterableDataset`s sharing one physical scan across
   independently-`__iter__`'d DataLoader workers — which is the exact class
   that failed three times in the shakeout. Under coarse determinism the draw
   is a per-emission weighted sampler seeded on `(seed, epoch, cell)`; resume
   does **not** checkpoint sampler RNG or a skim buffer — it re-opens at the
   **partition boundary** and reseeds, so the resume cursor stays the existing
   `(epoch, partition)` ints (§Determinism model).
2. **Both modality skims.** Unmatched spectra in the matched spectrum
   partitions (already downloaded for the join) feed spectrum-only draws.
3. **Spectra shortfall rule: redraw-if-empty.** Skimmed-spectra supply is
   per-cell and non-stationary; when the sampler picks the spectrum slot and
   the skim buffer is empty, it simply **redraws** — a one-liner under coarse
   determinism, where the old exact-replay design instead needed the skip to
   be a pure function of scan position. Effective spectra share still floats
   below 0.15 unless a reduced-weight standalone spectra source tops it up;
   the choice rides on the supply measurement.
4. **Drop-in for `interleaved()`**, validated **offline over local parquet
   fixtures** through the same datasets machinery — the path
   `interleaved()` was factored for. No training-machine contact before
   the gate passes.
5. **Adoption gate (the prototype):** assert the realized mix vs
   0.60/0.15/0.25 under redraw-if-empty (a **statistical** check over N
   draws, since the train stream is no longer bit-exact), **coarse-resume
   invariants** (re-open at a partition boundary replays no completed
   partition and skips none), **exact val replay** (val stays fully
   deterministic — same `(seed, epoch)` ⇒ identical stream), byte accounting
   (counting file wrapper), and a per-cell spectra supply-vs-demand report.
   Measure d and matched-cell counts from the match index at the same time.
   Building this gate was explicitly descoped on 2026-07-21; it remains the
   precondition for adoption.

## Determinism model (relaxed 2026-07-21)

ADR 0006 keeps the corpus **exactly** replayable: a repeating draw pattern
(never a sampler) so no RNG state is checkpointed, and a micro-batch-exact
resume cursor. Skimming **already forfeits that** for the streams it feeds —
image-only draws come from discards that repeat ~7–8× per epoch and skew to
the DESI footprint (see Negatives). Paying the full exact-determinism price —
a fixed 12:5:3 pattern, skip-if-empty as a pure function of scan position,
`pattern_phase` + skim-buffer state in every checkpoint — to protect a
no-replay guarantee the feature has *already* broken is incoherent. So the
skimmed **train** stream is relaxed to **coarse determinism**.

Only three guarantees are kept, because only these are load-bearing:

- **Val stays bit-exact.** Eval compares checkpoints; identical val batches
  per checkpoint are non-negotiable. Val is a separate `split`, tiny, and not
  skimmed — it keeps ADR 0006's exact path untouched.
- **Cross-rank shuffle agreement stays.** `shuffled(files, seed, epoch)` is
  identical on every rank so the modulo split is disjoint. This is sharding
  *correctness*, not draw-order RNG, and the relaxation does not touch it.
- **Coarse (partition-level) resume stays.** A kill/resume re-opens at a
  partition boundary: no partition is skipped or re-trained (coverage
  integrity), only the in-progress partition is partially re-drawn.

Given up deliberately: **bit-exact train reproducibility** and
**micro-batch-exact resume** of the skimmed stream. At 14M records × epochs
the ≤1 partition of re-drawn records per kill is statistically nil, and
dropout/init RNG already makes a run non-bit-exact unless painstakingly
seeded — so little real reproducibility is lost, while the sampler,
redraw-if-empty, and the vanishing skim/phase checkpoint state delete most of
the mechanism and most of the coupled-source risk (the shakeout failures were
in stateful exact resume + cross-process sharing, not in the draw itself).

Scope: this relaxation applies only to the skim source in this ADR; ADR
0006's live three-source interleave is unchanged until this design is
adopted.

## Options considered

### Option A — Yield discards from the pairs source (dilution)

**Rejected** (prior review, reconfirmed): at weight 0.25 a source yielding
~90–95% image-only rows delivers ~1–2% pairs; restoring 25% needs a
weight > 1. Per-record weights cannot express upweighting a sub-stream.

### Option B — Local materialization (cache-as-you-go pairs shard; pre-downloaded matched partitions)

**Rejected by constraint** (no local disk materialization), not by
analysis: it is the only option that also kills the dominant re-scan term
(~3–5× class). If disk policy changes, reopen — it dominates Option C.

### Option C — Demultiplexed scan (chosen, in progress)

As above. ~1.2–1.6× at fixed link; mix weights unchanged; every scanned
byte trains.

### Option D — Valley-filling only

fsspec readahead / next-row-group prefetch to lift average link
utilization from 70–90% toward ~95%+: ~1.1–1.4×. **Deferred, not
rejected** — complementary to C and much smaller in code.

### Option E — Lower the pairs weight

Cuts re-scans linearly with zero loader code. **Rejected** as a science
decision: the 0.25 upweight is the cross-modal bet (ADR 0006 §2).

### Option F — Row-level match filtering over HTTP

**Impossible**: parquet column-chunk granularity; every row group has
matches. (Also why the match index, which knows the matched ids exactly,
cannot reduce first-pass bytes.)

### Option G — Re-publish the catalog pre-cropped to 96×96

~60% wire cut (audit). **Out of scope**: a data-engineering job on the
hosted catalogs, not a loader change; cross-referenced so it isn't lost.

## Consequences

### Positive

- Standalone images-catalog download eliminated whenever d ≤ 29%
  (plausibly ~20–40% of epoch bytes); spectra-side download reduced.
- Every byte the forced re-scans download produces a training record.
- Record mix, interleave weights, and the union-schema contract unchanged;
  the change is contained in source construction.
- Stacks with Option D (~1.1–1.4×) for a combined ~1.3–2× without new
  infrastructure.

### Negative / tradeoffs

- **The gain is ~1.2–1.6×, not ~3–4×.** The dominant pairs re-scan term
  is untouched; on a fixed link this moves step time modestly, it does
  not make the NIC sufficient. Expectations set accordingly.
- **No-replay weakens for skimmed streams**: image-only draws come from
  discards that repeat ~7–8× per epoch, and coverage skews toward matched
  partitions (the DESI footprint). Tunable via a skim ratio < 1 (skim
  some discards, keep a reduced standalone images source) at the cost of
  giving back bytes.
- **Spectra share floats** under redraw-if-empty unless a top-up source is
  kept, complicating the outer interleave weights.
- **Coupled source = shakeout failure class.** Mitigated (single
  generator, no cross-process state, offline test path) but not erased:
  the DP/worker sharding interactions (`aligned`, `_iter_pytorch`,
  `n_shards` collapse) must be re-verified for the new source.
- Resume cursor stays the existing `(epoch, partition)` ints — no
  `pattern_phase`, no skim-buffer state — because the skimmed train stream is
  only coarsely deterministic (§Determinism model). Checkpoints across this
  change are still data-order-incompatible.

## Open issues

- **Measure d, matched image cells, rows/partition** from the match index
  (resolves the inconsistency and pins the saving).
- **Build the adoption gate** (prototype + offline assertions in §Decision
  5) — descoped 2026-07-21, still the precondition.
- Skim ratio: full skim vs partial (coverage retention) — needs the d
  measurement and a pretraining-quality judgment call on repeated
  discards.
- Spectra top-up vs floating share.
- Option D (readahead/prefetch) sizing and stacking.
- Re-verify DP=2/worker-split behavior and kill/resume at a **partition
  boundary** (offline) before any live run — coarse resume must replay no
  completed partition and skip none (§Determinism model).

## References

- `astro/src/astropt3/data/streaming.py` — source construction,
  `_paired_examples`, `interleaved`, `open_stream`.
- `astro/src/astropt3/data/packing.py` — the collator (pad-waste sizing).
- [ADR 0006](0006-stream-mmu-upstream.md) — the three-source corpus,
  weights rationale, match-index.
- `astro/docs/2026-07-21-streaming-throughput-audit.md` — ~250 KB/image,
  incompressible flux, 1 GbE ceiling, pre-crop republish idea (Option G).

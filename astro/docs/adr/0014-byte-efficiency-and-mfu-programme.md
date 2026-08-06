# ADR 0014 — Byte-efficiency and MFU programme for five-spoke training (crop retained)

- **Status:** Accepted (2026-08-06). Rewritten the same date to fold the
  measurement study into the gated byte-efficiency + MFU programme below;
  the programme is the committed plan and its gates govern execution, not
  adoption. Supersedes and absorbs the labbook working document *"Useful
  bytes per megabyte downloaded"* (2026-08-06, amended same day for the
  five-spoke corpus), which was deleted when the first version of this ADR
  was extracted — recover it from git history if a derivation needs
  checking.
- **References:** [ADR 0006](0006-stream-mmu-upstream.md), [ADR
  0008](0008-scalar-modalities.md), [ADR 0011](0011-skim-crossmatch-scans.md),
  [ADR 0013](0013-legacy-centred-mmu-expansion.md), PR #31.
- **Owner decision recorded here (dated 2026-08-06):** retain the 96×96
  central image crop; full-frame and foveated tokenisation are out of scope
  for this programme (§1).

## Context

Training on the ADR 0006/0011/0013 live stream is **transfer-bound, not
compute-bound**. On the five-spoke North corpus the GPU idles at every cell
boundary while ~1.18 GB of partner data arrives: median 0.68 s/step against a
mean of 7.80 s/step, with 93% of wall clock inside stalls. Three independent
attempts to fix that with *concurrency* all failed (§11). The only lever that
matters is therefore **loss-bearing signal per megabyte transferred**.

### What a megabyte buys (measured, single row group per catalogue)

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
- The discarded periphery is **not** empty sky: on 40 real North objects the
  median peripheral patch carries the same arcsinh-normalised variance as the
  median central patch (ratio 1.01×), and only 12.7% of full-frame patches
  fall below the crop's own 10th percentile. Full-frame tokenisation would be
  2.5× more tokens of roughly equal quality at zero extra bytes.

## Question

The five-spoke corpus is transfer-bound: ~93% of wall clock on the
`ar_replicas: 1` baseline is stalls waiting for bytes, and three separate
concurrency attempts (workers beyond 8, cell prefetch, fsspec/pyarrow
readahead) produced no improvement — the prefetch is actively harmful at 8
workers. Given that the binding constraint is loss-bearing signal per
megabyte transferred, and given the owner constraint that anchor images
remain cropped to 96×96, what is the ordered set of changes we commit to,
what metrics and gates govern each one, and what do we explicitly refuse to
build?

## Decision summary

Adopt a gated programme, in this order:

1. **Immediately, in parallel:** fetch-boundary instrumentation, MFU
   accounting, and a frozen benchmark; the sequence-assembly fingerprint
   fix; nested spectrum-leaf projection for DESI and SDSS; validation-set
   repair; the replay (`ar_replicas`) quality validation; open the
   transient-cache ADR conversation against ADR 0006/0011.
2. **Next:** per-band tokenisation A/B **on the cropped image** — now the
   only image-representation experiment in scope.
3. **Last:** transient cross-epoch cache prototype under stable ownership;
   re-measure partner-fetch locality on the 5,488-cell index before any
   ordering/caching-adjacent work beyond that.

Progress is judged on two headline axes — **useful bits per byte
transferred** and **model FLOPs utilisation (MFU)** — with scientific
quality per resource as the deciding gate. Full-frame tokenisation,
foveated tokenisation, and HSC full-frame are **rejected for this
programme** by owner decision (§1). Everything on the stop list (§11) is
refused with reasons.

---

## 1. Owner decision: retain the 96×96 image crop

`packing.IMAGE_CROP = 96` stands, for Legacy North anchors and HSC alike.
Full-frame tokenisation of the 152×152 frame (0.73 KB/token vs 1.82 cropped,
per §Context), foveated tokenisation (patch-8 centre + patch-16 annulus ≈
198 tokens for all pixels), and any patch-size change that alters image
tokenisation are out of scope.

**Owner rationale as stated:** the discarded periphery is mostly empty sky.

**Evidence note (recorded for honesty, per governance):** the periphery
measurement in §Context measured the opposite — the median peripheral patch
carries the same arcsinh-normalised variance as the median central patch.
Patch variance is a proxy for structure, not learnable signal — a
high-variance pure-noise patch is still irreducible — so the rationale may
yet be vindicated by a sharper measurement (spatial autocorrelation within
a patch, or variance against the per-pixel noise estimate). As measured
today, the "black sky" claim is not supported.

**Grounds on which the decision stands regardless:**

- **Sequence budget.** Full frame is 361 tokens/image vs 144 — fewer
  objects per packed row, fewer unique objects per step, and pressure on
  `sequence_length: 4096` p95/p99 object lengths once per-band tokenisation
  (§8) is layered on.
- **Checkpoint compatibility.** Any crop change invalidates every
  checkpoint and downstream number; it is a modelling change demanding its
  own ADR and A/B, a large evidence burden the programme does not need to
  pay.
- **Distribution shift.** Full frame changes what the model sees (more
  neighbours and background per object), entangling a byte-efficiency
  change with a modelling change.

**Consequence accepted with this decision:** the image byte-efficiency
ceiling is fixed. 92.3% of anchor row-group bytes are `image.flux`, of
which 60.1% are fetched and discarded before tokenisation; images stay at
~1.82 KB per fused token. No lever in this programme recovers those bytes.
All remaining gains come from factorisation (replay §7, per-band §8),
projection (§6), and cross-epoch reuse (§9).

**Reopening clause:** this decision may be revisited through an evidenced
amendment if (a) the sharper periphery measurement shows the discarded
region is predominantly irreducible noise (strengthening the decision) or
predominantly learnable structure (weakening it), or (b) the cache (§9)
lands and the corpus stops being transfer-bound, at which point the
byte-efficiency argument for full frame lapses anyway.

## 2. Metrics: useful bits per byte, and MFU

Every change in this programme is judged on four metrics, not on token
counts:

- **Wire efficiency** `E_values` = valid loss-target scalar values per MiB
  actually fetched. Counts observed target dimensions; replay and per-band
  do **not** increase it.
- **Factorisation efficiency** `E_AR` = loss-bearing AR tokens per MiB
  fetched, decomposed into primary tokens, replay tokens, distinct span
  permutations, and identical duplicates.
- **Model FLOPs utilisation (MFU)** = achieved model FLOPs per second
  divided by the accelerator's peak (bf16, dense — no sparsity credit).
  Defined and decomposed in §2a below.
- **Scientific efficiency** `E_science` = Δ validation quality (family and
  modality losses, fixed redshift/morphology probes) per TiB downloaded
  **and per GPU-hour**, at matched bytes and matched wall clock.

No change is accepted for raising `E_AR` or MFU alone — both are gameable
by repetition (identical replicas raise MFU and `E_AR` while adding zero
information; so would padding). They are **necessary-but-not-sufficient**
gates; `E_science` is the deciding metric wherever they conflict.

### 2a. MFU: definition, decomposition, and role

**Definition.** Per step: model FLOPs = FLOPs/token × non-padding tokens in
the step (padding earns no MFU credit — a padded-out packed row is wasted
compute exactly as a stall is wasted time). FLOPs/token uses the standard
decoder estimate (≈ 6·N_params + attention term at `seq_len` 4096) plus the
modality encoder/decoder and jetformer flow/GMM terms, which are **not**
negligible for a 70M backbone with 47 modality heads — the accounting must
include them or small-model MFU will be flattered. Divide by wall time and
peak FLOPs for the reserved GPUs.

**Decomposition (mandatory in every report):**

```text
MFU = MFU_busy × (1 − stall_share) × utilisation_packing
```

- `MFU_busy` — MFU over compute-busy time only (excludes loader waits).
  Moved by representation and model changes: per-band's narrower 64-dim
  heads, kernel efficiency, micro-batch shape.
- `stall_share` — fraction of wall clock waiting on the loader. Moved by
  byte/loader levers: replay, projection, cache, workers.
- `utilisation_packing` — non-padding fraction of packed rows. Moved by
  packing policy and object-length distributions.

The decomposition is the point: an undecomposed MFU comparison between a
fused and a per-band arm confounds a data-pipeline effect with a
model-shape effect and supports the wrong conclusion.

**Where the baseline stands (measured wall-clock ratios, instrumented MFU
pending §3):** step medians of ~0.68 s against a 7.80 s mean put the
`ar_replicas: 1` baseline at roughly **9% of its own stall-free MFU**, and
replay-2 at ~28% (2.51 s mean). The §3 instrumentation replaces these with
measured values, and the benchmark freezes the measured baseline as B0's
reference MFU.

**Why MFU belongs alongside the byte metrics.** The byte metrics price the
resource we are short of (wire bytes); MFU prices the resource we have in
surplus but are wasting (reserved FLOPs). They are duals under the
fixed-step arithmetic below: at fixed tokens/step, cutting stall time
raises MFU exactly as it cuts bytes per unit time. A change that raises
`E_AR` without raising MFU (or vice versa) is a flag that something else
moved — packing, model shape, or duplication — and the decomposition says
which.

**Fixed-step arithmetic (applies throughout):** at fixed `train_steps`,
tokens per step are pinned by `seq_len × batch`, so inflating tokens per
object (replay, per-band) proportionally deflates unique objects per step,
epochs per run, and total bytes transferred — and, by shrinking
`stall_share`, raises MFU. Total exposures per object are unchanged
(measured for replay: 8.7 exposures at replicas 1, 2, and 4). The single
governing hypothesis of this programme is therefore: **does quality at
fixed step count survive the reduced unique-object diversity?** Every A/B
in §7–§8 is an instance of that question; MFU and the byte metrics measure
the benefit side, `E_science` measures the risk side.

## 3. Instrumentation and benchmark (Phase 0)

Byte accounting through `HfFileSystemFile._fetch_range` is the only
accepted instrument — HTTP-request counting and contiguous-log-run counting
are both recorded failures (range requests are not downloads; 16 workers
interleave one log). Extend the instrument with structured fields: DP rank,
loader worker, source, partition path, anchor cell, byte range and payload
bytes, fetch wait duration, and projected columns where known.

Per step (or short window), record: transferred bytes by source;
non-padding and loss-bearing tokens by family/modality; valid target
dimensions; distinct base object ids; replica count and distinct span-order
count; cells consumed and boundaries crossed; packing utilisation; **step
wall time split into compute-busy and loader-wait; achieved model FLOPs and
the three MFU factors of §2a**; rolling 100-step source composition (the
`object_id_log` already carries this); RSS per process.

MFU accounting notes: FLOPs/token is computed once per (model config,
tokenisation policy) pair and pinned in the benchmark record — per-band
changes it (more tokens, narrower heads), so arms must not silently share a
constant. GPU-idle attribution uses the trainer's own step timing plus the
loader-wait instrumentation; no profiler in the measurement path (profiler
runs are separate, unfrozen).

Freeze a benchmark: pinned five-spoke index and source revisions, fixed
dp=2 × 8-worker layout, fixed cell order and starting state, one ordinary
window and one HSC-enriched window, several repetitions. Report mean, p50,
p95, p99, stall count, stall share, **and decomposed MFU** — the 0.68 s
median / 7.80 s mean baseline shows the median alone is blind to the
problem, and undecomposed MFU would be equally blind to its cause.

## 4. Validation repair (Phase 0b — gates everything downstream)

Every acceptance gate below routes through validation quality, and the
current validation set cannot carry that weight: `VAL_PARTITIONS = 8` is
0.15% of 5,488 cells, and realised composition swings 0.1% → 16.4% for HSC
between runs (§Mechanism notes). Therefore, before the benchmark freezes:

- raise `VAL_PARTITIONS` to a defensible fraction of cells (order 1–2%,
  sized so each family's val loss is stable across reruns); this changes
  the split and **bumps the source assembly** — do it once, now, not
  mid-programme;
- land the composition-per-100-step-window measurement from the audit logs;
- require composition-matched or composition-stratified comparison windows
  for every A/B; decide HSC-touching questions on HSC-enriched cells, since
  0.7% coverage means random short windows cannot evidence that spoke.

## 5. Sequence-assembly fingerprint (immediate, independent of experiments)

`source_assembly` protects record *order* only. A checkpoint saved at
`ar_replicas: 1` and resumed at `ar_replicas: 2` today passes the assembly
check while silently changing the emitted sequence stream. Extend the
resume-state tag to a fingerprint over: source-graph assembly and
revisions; modality-config hash; image-crop policy; band-tokenisation
policy (§8); `ar_replicas` and replica-separation policy (§7); span-order
algorithm version; sequence length. Mismatch rejects the stream state
(weights still load), exactly like an assembly bump. This protects every
A/B in the programme from contaminated resumes.

## 6. Nested spectrum-leaf projection (ship now)

Change `_SOURCE_COLUMNS["desi"]` and `["sdss"]` from the whole `spectrum`
struct to the leaf paths `spectrum.flux`, `spectrum.lambda`,
`spectrum.mask`, and change `_spectrum_part()` to a whitelist of those
three children rather than copying every struct child present.

Measured saving: 40% of spectrum bytes per read (60.7 → 36.4 MB, counted at
`_fetch_range`). Demoted, correctly, from headline to hygiene: DESI is 6.3%
of five-spoke anchors (was 92.2% at three spokes), so the corpus-level
effect is small — but it is free, reversible, and scientifically neutral
(`ivar` is read nowhere outside synthetic fixtures; ADR 0007 normalisation
does not use it).

Gates: decoded records byte-identical before/after; the saving re-verified
against a **520 MB single-row-group partition**, not only the 60.7 MB one
(footer and range-coalescing overheads need not scale linearly); SDSS
saving measured independently; record order and resume unchanged.

## 7. Replay validation (`ar_replicas`) — the headline lever

With the crop retained, replay is the largest measured lever in scope. It
is already built: `ar_replicas` in `nanotron_loader.py` re-emits each
downloaded record under a different ADR 0008 span order — the shuffle is
already a pure function of `(object_id, epoch)`, so the suffixed id *is*
the reseed, with no new RNG and nothing extra to checkpoint. Replica 0
keeps the original `object_id`, replicas get `#n`, so the no-replay audit
still catches accidental duplication. Measured over identical first 117
steps, five-spoke, 8 workers, dp=2: `ar_replicas: 1` → 7.80 s/step mean,
21/116 slow; `ar_replicas: 2` → 2.51 s/step mean, 11/116 slow — **~3.1×
faster per step with slow steps halved**, an implied stall-share drop from
~91% to ~72% and a ~3× MFU improvement, to be confirmed by the §3
instrument. It is promoted to production default only through the
following gates.

**7a. Distinct-order enforcement.** Suffixing the object id reseeds the ADR
0008 shuffle but does not guarantee a different permutation. Before
emitting a replica: skip replay for one-span records (no alternative order
exists); cap replicas at the number of distinct useful permutations;
enforce that emitted replicas carry distinct span orders. Note the corpus
asymmetry: 94% of anchors are galaxies-only with ~37 spans (image + 34
`gwh_*` + ebv + photometry) — a huge permutation space where replay is most
defensible — while unmatched two-span records support at most one extra
ordering. This gate is precisely what keeps replay's MFU gain honest:
identical duplicates would raise MFU and `E_AR` while training on nothing
new.

**7b. Decorrelation.** Replicas currently land adjacent in the same packed
row; document masking prevents cross-attention but not gradient repetition
within the batch. Add a deterministic bounded buffer that separates
replicas from their base record — at most one replica of a base object per
packed row, preferably per micro-batch — reconstructible from the saved
stream position (or carried in checkpoint state) so resume stays exact.

**7c. Quality A/B.** Arms: replicas 1 (B0); replicas 2 adjacent (B1 —
continuity with the measured result); replicas 2 decorrelated (B1-D).
Replicas 4 is refused until B1-D passes. Compare at matched bytes and
matched wall clock on the repaired validation set. Accept if quality per
TiB **and per GPU-hour** is preserved or improved, the audit stays
exactly-once (replica 0 keeps the original id; every logged line unique),
resume stays exact, and most of the measured MFU/stall gain survives
decorrelation.

## 8. Per-band tokenisation — on the cropped image

With full frame out of scope, per-band is the only image-representation
experiment: 144 × 192-float fused tokens → 432 × 64-float per-band tokens
from identical bytes. State it correctly: a finer autoregressive
factorisation with narrower heads — `E_AR` rises 3×, `E_values` does not
move (27,648 target values either way), and cross-band structure is learned
autoregressively instead of pre-fused.

MFU note: per-band moves **two factors in opposite directions**. It
shrinks `stall_share` (3× tokens per object → fewer unique objects per
step → fewer boundary fetches) but may lower `MFU_busy` (64-dim heads and a
narrower jetformer flow do less arithmetic per token against the same
kernel overheads; more tokens per object also lengthens attention). The
decomposed report decides whether the net is a win; an undecomposed MFU
number would hide the trade.

Design for the first A/B: one modality per survey image; **fixed** band
order (g,r,z; HSC g,r,i,z,y if it ever reaches this stage), centre-out
spatial order within band; no band shuffling (one variable at a time — band
order as an ADR 0008 ordering decision is a follow-up). Config carries
`channel_tokenization: fused | per_band` and `band_order`; 432 positions
for the Legacy crop; the fingerprint (§5) covers it; checkpoints
incompatible — new configs, never edits to historical ones.

Gates: p95/p99 object length and packing utilisation against `seq_len`
4096 (a cropped per-band image object is ~437 tokens before scalars — fine
alone, check the packed distribution); jetformer flow/GMM behaviour at
`input_size` 64, including the noise curriculum's useful range; validation
loss **per pixel** at matched bytes, probes at matched wall clock;
decomposed MFU with per-arm FLOPs/token accounting (§3). Accept only on
`E_science`, never on token count or headline MFU. If accepted, combine
with the §7 winner (arm P1) and gate the combination the same way.

## 9. Transient cross-epoch cache (conversation now, prototype last)

Each epoch re-downloads the corpus; at ~262 KB per anchor row group,
2,182,875 anchors ≈ 0.57 TB/epoch (indicative, from single-partition
sampling), and 8.7 exposures per object over 20k steps ≈ 5 TB of anchor
traffic without reuse. This dominates every single-epoch optimisation for
any multi-epoch future — and it is the only lever in the programme that can
push MFU toward compute-bound territory rather than trimming the stall
share, since epoch-2+ reads come from local disk. It is in tension with
ADR 0006 (live streaming, reshard deleted) and ADR 0011 (no local cache).
Open the amendment conversation **now** (longest lead time); build
**last**.

Framing: a *transient* cache — evictable, reconstructible, no schema of its
own — never a reshard or second corpus. Keyed by pinned revision, path,
byte-range/projection, and decode version; checksum-validated;
byte-bounded; safe under concurrent readers. Partial-by-construction: at
dp=2 a full per-node share is ~285 GB (likely infeasible); at dp=32 it is
~18 GB (trivial); hottest-cells-first under a byte budget covers both.

Prerequisite: **stable cell ownership.** Today's shuffle-then-deal changes
which rank sees a cell every epoch, defeating node-local reuse. Evaluate
stable cell→rank ownership with shuffling only within each rank's set
(assembly bump; composition and balance measured — the composition-variance
concern in §Mechanism notes interacts here). Go/no-go: ≥80% epoch-2 byte
reduction, bounded disk/RSS, no cross-split or duplicate emission,
acceptable rank balance, and a measured epoch-2 MFU report (expected to be
the largest single MFU movement in the programme).

## 10. Worker and scheduling policy

Baseline: 8 loader workers, exactly one large read in flight per worker, no
cell-prefetch thread, no fsspec background cache. Worker count is a large
lever (2 workers: 19.95 s/step vs 9.02 at 8, because a ~100 s fetch
amortises over `num_workers` steps of the loader rotation) — never reduce
it as a congestion workaround; the prefetch is fatal at 8 workers (NCCL
watchdog kill) and merely useless at 2. After the representation
experiments, a bounded sweep (4/8/12) and, separately, a controlled
ready-first delivery experiment are permitted — noting ready-first changes
record order (assembly bump) and biases short-window composition toward
fast cells. Adaptive replay *during* a measured boundary wait (replaying
only while the single next-cell fetch is incomplete) is the designated
successor to fixed replay, after §7 passes — it preserves unique-record
rate whenever data is ready and converts residual stall time to MFU
directly.

Before any locality/block-ordering work: re-measure partner-fetch
redundancy on the 5,488-cell index with the §3 instrument (the 39% /
434-vs-266 figures are three-spoke and superseded; the 8–16-cell block
suggestion is smaller than the 16 consumers of a dp=2 × 8 run and buys
nothing — mechanically, `shuffled()` runs *before* `owned_by_rank` deals
cells round-robin to `dp × num_loading_workers` consumers, so any block
smaller than the consumer count has its locality stripped on arrival). Only
build if the five-spoke trace shows a material byte opportunity at
consumer-aware block sizes — the null result of the reverted block-shuffle
run and DESI's exit from the byte budget both argue it will not.

## 11. Stop list (refused, with reasons)

- **Full-frame and foveated tokenisation** — owner decision, §1.
- **Cell-boundary prefetch thread** — not merely neutral: at 8 workers on
  the five-spoke corpus it hung a rank for 20 minutes and NCCL's watchdog
  killed the job. It adds a *second* concurrent large read per loader
  process (~32 in flight instead of 16) and creates no bandwidth.
- **fsspec `cache_type=background` + pyarrow `pre_buffer`** — within noise
  of baseline (3.73 vs 2.88 s/step mean, identical slow-step count).
- **Row-group / page skipping on published files** — every DESI partition
  is a *single* row group (up to 520 MB) and no page index is written;
  there is no sub-file granularity to skip to. Remains an upstream ask
  (64 MB row groups, page index, `BYTE_STREAM_SPLIT`) to the MMU
  publishers.
- **Request-count or log-run "redundancy" metrics** — recorded measurement
  failures (HTTP requests count *ranges*, and 16 loader processes
  interleave in one log with no worker id).
- **Headline (undecomposed) MFU as an acceptance criterion** — gameable by
  duplication and padding, and confounds pipeline effects with model-shape
  effects; only the §2a decomposition is admissible.
- **Dropping unused anchor scalars** — <0.1% of the row group; keep them.
- **Filtering "empty-sky" patches** — refuted by the §Context periphery
  measurement (and §1's evidence note applies equally here).
- **HATS index tables** — solve partition lookup, which the match index
  already provides.
- **Standalone unmatched-source scans; strict N-way joins** — ADR 0013
  fetched-only policy stands; unmatched rows are kept (36.3% of tokens at
  zero marginal byte cost; dropping them saves no wall clock — and they
  raise MFU for free, since the stall is the fetch, not the step).
- **8–16-cell block shuffles from the 181-cell simulation** — superseded
  numbers, the consumer-count error above, and an inconclusive live run
  (block 256 over 36 steps: identical 0.69 vs 0.68 s median, identical
  slow-step count; reverted unmerged in favour of replay — no signal
  either way, and no five-spoke re-measurement yet).
- **`ar_replicas: 4`** — until B1-D passes distinctness and correlation
  gates.

## Experiment matrix (gated, in order)

| Arm | Image | Bands | Replay | Gate to run |
| --- | --- | --- | ---: | --- |
| B0 | 96 crop | fused | 1 | frozen baseline (reference MFU) |
| B1 | 96 crop | fused | 2 adjacent | reproduce measured result |
| B1-D | 96 crop | fused | 2 decorrelated | after §7a/7b land |
| P0 | 96 crop | per-band | 1 | after B-arms decided |
| P1 | 96 crop | per-band | §7 winner | only if B1-D and P0 both pass |

Each arm reports the §3 per-step record plus `E_values`, `E_AR`
(decomposed), **decomposed MFU (MFU_busy, stall_share,
utilisation_packing, with per-arm FLOPs/token)**, and `E_science` at
matched bytes and matched wall clock.

## Mechanism notes worth keeping

- **Stalls arrive on the loader rotation, not at random.** At
  `num_loading_workers: 8`, every >60 s stall from step 80 landed at exactly
  `step % 8 == 6`. Each worker's cell-boundary fetch blocks its own turn and
  nothing amortises it. This is why adding workers cannot help: each new
  worker adds its own serialised fetch to the same rotation.
- **The partition ceiling is gone.** `num_loading_workers <= floor(cells/dp)`
  bound the pilot recipes at 165 train cells; at 5,488 cells it no longer
  binds, and the deferred "shard by row group instead of by cell" loader
  change is not worth building.
- **Unmatched rows stay** (ADR 0005/0011), and the economics back it: 36.3%
  of training tokens come from rows never fetched for their own sake, at
  zero marginal bytes, spending compute we have in surplus. The real threat
  is **composition variance**, not the matched/unmatched ratio — HSC was
  0.1% of objects in one run and 16.4% in another because its matches live
  in a small part of the footprint. Per-family loss weights normalise
  *within* a batch, so a stretch of spectrum-only batches trains no image
  head at all. Block ordering would make this worse. `object_id_log`
  already records what is needed to measure it (§4 lands that measurement).

## Caveats

Column shares come from one row group of one partition per catalogue; bytes
are compressed on-disk sizes, and pyarrow also fetches footers and coalesces
ranges, so real HTTP bytes are somewhat higher. Runs compared here hit a
shared hub at different times — the *absence* of benefit is the finding, not
the ordering between arms.

## Consequences

- The image byte-efficiency ceiling is accepted: ~1.82 KB per fused image
  token and a 60% pixel discard stand for the life of this decision (§1
  reopening clause notwithstanding).
- The programme now carries a compute-side ledger as well as a wire-side
  one. Expected movements, to be confirmed by instrumented MFU: baseline
  ~9% of stall-free MFU; replay-2 ~3× that; the cache the only path toward
  compute-bound MFU; per-band an open trade between stall share and
  MFU_busy.
- The programme's expected byte wins are: replay ~2× fewer unique-record
  bytes per fixed-step run (measured ~3.1× wall-clock mean); per-band up to
  a further token-inflation factor **if and only if** quality per pixel
  holds; projection a few corpus percent; the cache, if adopted, ~an
  epoch's bytes for every epoch after the first — the dominant term for
  multi-epoch runs.
- Validation cost: `VAL_PARTITIONS` rises and every arm pays for
  composition-controlled windows; this is the price of decisions that mean
  anything on a corpus whose modality mix swings 160× between short runs.
- Two assembly bumps are scheduled (validation split §4, stable ownership
  §9); both reject stale stream states while keeping weights loadable, per
  the documented mechanism.
- Old checkpoints are unaffected by everything except per-band adoption,
  which creates a new model family by design.

# MMU modality expansion plan

**Status:** plan (2026-08-04), post-ADR-0006-closure
**Objective:** maximise useful training signal per wire byte. The run is
NIC-bound (1 Gbit, ~250 KB/image), so useful-signal-per-byte and
useful-samples-per-second are the same optimisation problem.

This supersedes the *ordering* in
[`mmu_crossmatch_research.md`](mmu_crossmatch_research.md), which ranks the
same catalogs by science value and reaches a different sequence. Where the two
disagree, this document governs sequencing; the research doc remains the
reference for governance rules and per-catalog schema detail.

Derived from the 2026-08-04 byte-economics review of PR #28 (external
analysis session, archived with the project records), whose live probes
against the published HATS catalogs supply the footprint and yield numbers
below.

---

## The shape: hub-and-spokes, not a pairwise survey walk

One image scan is **the hub** — it's the expensive byte stream, and
crossmatch-only already made it the only pixel stream. Each partner catalog is
an independent **ids-only match index against the same image rows**:

```
match_index_desi:      (img_cell, img_id, desi_cell, desi_id, dist)
match_index_sdss:      (img_cell, img_id, sdss_cell, sdss_id, dist)
match_index_galhats:   (img_id -> dr8_id)        # pure id join, no positions
match_index_provabgs:  (desi_id -> provabgs_id)  # pure id join
```

At demux time each image row picks up whatever attachments its ids appear in:
some records image-only, some image+DESI, some image+SDSS, a rare few with
both. `ObjectSequencer` already packs whatever modalities a record carries —
"both modalities are optional per record" is an existing invariant.

**Why not a multi-way join.** Measured, not assumed: in a cone yielding 1,031
image×DESI pairs, DESI×SDSS overlap was **7 objects** (4,431 DESI and 6 SDSS
spectra present; the 7 matches are real — redshifts agree to ~1e-4 — the yield
is just tiny). Each added catalog multiplies the footprint penalty:

| join | ceiling |
|---|---|
| images ∩ DESI | ~1 M (DESI's 1 % footprint binds) |
| images ∩ DESI ∩ SDSS | ~10³–10⁴ |
| + JWST/HSC deep fields | essentially empty |

**Why the economics favour attachments.** An image costs ~250 KB on the wire;
attachments are 1–30 KB. Attaching *both* DESI and SDSS adds ~25 % bytes while
roughly doubling that record's training signal. Per MB, an image with more
attachments is strictly better — so the multi-index design monotonically raises
useful-signal-per-byte on a fixed NIC.

---

## Order (by return per engineering hour and per wire byte)

### Phase 1 — attachments (the per-byte win)

Config generalizes `match_index: path` → `match_indexes: [...]` with a
per-index join type.

| # | Attachment | Rows | Bytes/record | Why |
|---|---|---|---|---|
| 1a | `galaxies-with-hats` | 8.65 M | ~1 KB | GZ-DESI vote fractions, NSA Sérsic/mass/absmags, OSSY AGN/BH, ALFALFA HI, JHU SFR posteriors, photo-z + spec-z. Already cross-identified to Legacy imaging by `dr8_id`/`brickid`+`objid` — a **trivial id join, not positional**. Probably attaches more supervision per wire byte than everything else combined. |
| 1b | `mmu_sdss_sdss` | 806 k | ~30 KB | Second spectrum partner. Same struct as DESI, so the tokeniser works as-is (different λ grid/length, which the continuous-λ patch design already handles). 23.8 % sky over LS-North's 13.7 % ⇒ far more of the row groups the scan streams anyway contain a match. ~2× pairs, spread beyond DESI's rosettes. |
| 1c | `mmu_desi_provabgs` | 223 k | <1 KB | Bayesian M*/SFR for DESI targets already paired. Id join. |

**Skip `mmu_gz10`** — subsumed by `galaxies-with-hats`' vote fractions.

**Caveat on 1a:** its own images are JPEG thumbnails, not linear flux. Use it
as a *metadata* attachment only; do not stream its pixels.

**Caveat on 1b:** SDSS `object_id` comes back as a padded byte-string
(`b'   1176734223403345920'`) vs DESI's clean strings. Normalize in the index
build or joins silently miss. Also re-verify partition-locality — DESI's
"one image partition → exactly one spectrum partition" was verified, but SDSS's
HATS partitioning is coarser. Worst case 2–3 spectrum partitions open per image
partition; still row-group-bounded.

### Phase 2 — second imaging regime (the diversity win)

- **JWST first** (~250 k rows). The packer already handles 96×96 cubes; needs
  NIRCam band-registry entries. A JWST×LS *image-pair* index comes almost free
  once the source streams — and image×image pairs are real multi-view signal
  (same galaxy at 0.03″ vs 1″, different depth/PSF/bands) that no augmentation
  fakes.
- **HSC last** (475 k, 0.2 % sky). Its image struct genuinely differs — adds
  `ivar`/`mask` planes, 5 bands. Real "different image family" engineering for
  modest yield. Gate it on whether JWST's image pairs show value in probes.

- **`mmu_legacysurvey_dr10_south_21`** sits alongside these: it roughly doubles
  the imaging corpus and raises pair yield (much DESI/SDSS coverage is in the
  South), but it's *more image bytes* — at a fixed NIC it reallocates
  throughput rather than adding it. Value is diversity + pairs, not obj/s.
  It is also the only proposed fix for the dp=64 worker ceiling that isn't a
  loader change.

### The strategic fork — stellar branch

`mmu_gaia_gaia` (122 M rows, 48 % sky) is the biggest sample-count lever in
MMU and absurdly cheap per byte (~1 KB/object). But BP/RP coefficients are
**not** λ-sampled flux, so the DESI/SDSS tokeniser doesn't apply — it needs a
new modality head. `mmu_apogee_dr17` (720 k, λ-sampled) is closer to drop-in
and pairs with Gaia + LS imaging.

**Advice from the plan: reserve the schema slot (one enum field on the index),
build nothing.** Don't paint into a corner; don't take the branch yet.

### Skip for now

PLAsTiCC (simulated LSST light curves — synthetic data in a frontier
pretraining corpus), the light-curve sets (TESS/YSE/CfA/CSP/PS1/SNLS/DES/Swift,
`mmu_btsbot`) — a whole new time-series modality for mostly tiny datasets,
Chandra (129 k X-ray), MaNGA (10.7 k IFU), VIPERS (91 k).

---

## Two design decisions this forces

1. **Governance generalizes from a ratio to a composition budget.** The old
   `0.60:0.25` images:pairs skim isn't expressible once "pairs" isn't one
   class. Govern on a composition histogram — image-only / image+1 / image+2+ /
   spectrum-only — or rare compositions starve and common ones flood.
   *Note:* crossmatch-only has since removed weighting entirely and the mix is
   emergent (~41 % image-bearing measured). So this is now a decision to
   *reintroduce* governance, not to modify it.

2. **Dedup policy for multi-attached objects.** An object with DESI *and* SDSS
   spectra: one record with both (richer, cheaper per byte, needs the sequencer
   to accept two spectrum segments) or two records (no packer change, image
   tokens duplicated)? Plan's advice: **two records now**, combined records
   when the sequencer grows multi-instance modality support.

**Watch as attachments multiply:** a record with image + 2 spectra + 40 scalars
hits the 4096-token budget differently from image-only records. The composition
governor and the packer must agree on token-length expectations per composition
class, or micro-batch token efficiency quietly degrades.

---

## Mechanics in this repo

- **Val split:** define val by **image partitions**, attachments follow the
  image. That keeps spatial disjointness where it matters and avoids leakage
  when an attachment catalog's partitioning differs from the image catalog's.
  Deliberately untouched by the ADR 0006 merge.
- **`SOURCE_ASSEMBLY` must be bumped on any change to record order** — a saved
  stream position is an index into it. Currently `crossmatch_only_v3`.
- **Memory is per-cell.** `UNMATCHED_BUFFER_BYTES` (256 MiB) was sized from
  measured footers (74.6 KiB/DESI spectrum row; cells pin 3.7 MiB–1.13 GiB
  unbounded) against a 68.7 GiB cgroup. Every new partner changes that
  arithmetic — redo it with `scripts/probe_stream_rss.py --cells N`.
- **Partition ceiling:** `num_loading_workers ≤ floor(165 / dp)` today. More
  attachments don't lift it (same image hub); more *image* cells do.
- **No lsdb at train time.** `[data]`-extra, index-build only.
- **Resume state stays ints:** cursor per source + per-index row offsets, no RNG.

## Contracts that raise rather than guess — preserve this

| contract | file | what a new partner needs |
|---|---|---|
| Image bands | `data/band_registry.py` | NIRCam entries for JWST; source-aware entries for HSC. Unknown bands raise. |
| Image shape | `packing.py`, `tokenization.py` | Fixed `(3,152,152)` → 96×96 crop. 4/5-band is a new modality config, not a flag. |
| Spectra | `data/spectral.py` | Unknown grids raise deliberately. SDSS is same-struct/different-grid; APOGEE needs its own transform. Never interpolate silently. |
| Scalars | `data/scalar_registry.py` | One-token GMM, fixed transforms, `loss_weight` 0.1. |

## Governance rules (from the research doc — these still hold)

1. Join on sky coordinates, never `object_id` across surveys — except the
   genuine id-joins (`galaxies-with-hats`, PROVABGS) where a catalog is already
   cross-identified. Store both ids, both cells, separation, radius, epoch
   treatment, revision.
2. Spatial splits apply to the connected component, not one source.
3. Record selection functions and licenses per source; preserve revision hashes
   (revisions float upstream).
4. Normalise by physical provenance, not corpus statistics — fixed invertible
   transforms. AION-1 fits empirical scalar CDFs; incompatible with checkpoint
   portability here.
5. Paired rows are not population data. Reciprocal matching favours bright,
   isolated, well-centred sources.

---

## ADR numbering collision — resolve before writing

The plan calls these ADR 0012 (attachments) and ADR 0013 (second imaging
regime). **0012 is already taken** in the repo by "Gate MMU streaming
throughput by measured byte economics" (Status: Proposed, 2026-07-22), and
0013 exists only in commit subjects. Either renumber the new ones to 0013/0014
or fold/supersede the existing 0012 first.

## Current state of the deferred work

`mmu-corpus-expansion` branch (pushed, no PR) carries a reciprocal match-index
builder (`scripts/_match_index.py`), three pairing CLIs (Legacy South × DESI,
HSC × SDSS, Gaia × APOGEE), seven `*_scout.parquet` density samples, the
research doc, and `tests/test_match_index.py`.

Note this is aimed at the *research doc's* ordering, not the plan's: there is
**no `galaxies-with-hats` or PROVABGS builder**, and no SDSS-against-LS-North
index (the existing one is HSC × SDSS). Phase 1a/1b/1c need new builders.

Two known defects in what's there: the reverse crossmatch pass isn't gated by
`--limit-partitions` (so a "scout" run still pays a full spectra-anchored
scan — fatal on Legacy South), and the index schema emits ids and cells only,
without the separation/radius/epoch/revision the governance rules require.

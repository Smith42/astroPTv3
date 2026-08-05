# Repository activity — 2026-08-04

**Date boundary.** The system clock was `2026-08-05T14:17:07+01:00`, so “yesterday” is the local calendar day **2026-08-04 (Europe/London offset +01:00)**. This note uses commits on all local refs with commit timestamps in `[2026-08-04T00:00:00+01:00, 2026-08-05T00:00:00+01:00)`. The 2026-08-05 commit `3a44f7a` and current uncommitted work are excluded.

## Executive summary

Two connected changes dominated the day:

1. The existing MMU loader was made safer and more honest: unmatched-spectrum memory was bounded, all 165 train cells were dealt across DP ranks without duplication or truncation, invalid worker counts became hard errors, the one-pass crossmatch-only corpus was documented, and a 600-step live run validated the result.
2. **ADR 0013 introduced the “hub and spoke” design**—formally a Legacy-centred rooted star—and its shared model/config foundation landed. At day-end the ADR remained **Proposed**: metadata, token allocation, registry-driven packing, family-balanced loss, logging, and the HF/Nanotron parity surface existed, but anchor selection, source indexes/adapters, stream topology, and spoke-specific evidence did not.

## Commits

| Area | Commits | Consequence |
| --- | --- | --- |
| Crossmatch-only correctness | `545d5ac`, `1f78dd4`, `374f86b`, `bd4da5f` | Bounded a confirmed OOM source; enforced DP/worker capacity; corrected the corpus count to 173 cells/165 train; replaced truncation plus `split_dataset_by_node` with `files[rank::dp]`. |
| Crossmatch-only decision/evidence | `103cf07`, `e46d628`, merge `56cc755` | Closed ADR 0006, amended ADR 0011 to crossmatch-only, recorded live validation, and merged PR #30 to `main`. |
| Expansion exploration and correction | `9d1dfff`, `cbce31b`, `c2e6569` | Added exploratory reciprocal builders/scouts, replaced their pairwise sequencing with a byte-economics hub-and-spokes plan, then deliberately deleted the unreproducible/wrong-sequence builders while retaining the research history. |
| Hub-and-spoke decision/foundation | `c423371`, `d12c94f` | Added ADR 0013 and its acceptance test plan, then implemented shared modality metadata, expandable token blocks, source/family-driven packing, and family loss in both model surfaces. |
| Hygiene/topology | `80b9195`, `1102edf`, `b9ab8b3`, merge `4cfc618` | Ignored 71 GB-class rebuildable caches/scouts, removed an accidentally cross-branch test, deleted seven obsolete experiment configs (1,016 lines), and merged `origin/main` into the expansion branch. |

## ADR 0013: Legacy-centred hub and spokes

### Decision and rationale

ADR 0013 answers how to add galaxies-with-hats, SDSS, PROVABGS, and HSC without repeatedly paying for image scans, erasing provenance, leaking across spatial splits/resume, or allowing many scalar heads to dominate loss. It chooses a **Legacy-centred rooted star**: one Legacy North or South anchor; DESI, SDSS, and HSC reciprocal positional spokes only to that anchor; galaxies-with-hats→Legacy and PROVABGS→DESI only through genuine lineage identifiers; no attachment-to-attachment spatial indexes and no complete N-way match requirement ([`astro/docs/adr/0013-legacy-centred-mmu-expansion.md`](../adr/0013-legacy-centred-mmu-expansion.md) at `d12c94f`, lines 17–45).

The immediate rationale was byte economics. The superseding plan records that a strict pairwise/N-way walk collapses overlap (only 7 DESI×SDSS objects in a cone with 1,031 image×DESI pairs), while 1–30 KB attachments are cheap beside an already-fetched ~250 KB image (`cbce31b`; [`astro/docs/mmu_modality_expansion_plan.md`](../mmu_modality_expansion_plan.md)). The rooted star therefore pays image bytes once, keeps source identity explicit, and grows positional indexes linearly rather than pairwise (ADR 0013 at `d12c94f`, lines 186–200).

Other fixed contracts were consequential:

- Select the anchor with deterministic stratified scouts using non-padding target tokens per transferred byte; stop at a bootstrap 95% relative half-width ≤10% or 32 cells, and report “inconclusive” at the cap rather than silently escalating (ADR 0013 at `d12c94f`, lines 47–67).
- Preserve existing token ids 0–16; allocate complete three-id blocks in 17–63, then explicitly grow the vocabulary above 63. Old checkpoints are not silently upgraded (lines 69–79).
- Keep surveys/products as distinct modalities and provenance domains; DESI/SDSS spectra and Legacy/HSC images do not share heads or identities (lines 69–74).
- Average losses inside `image`, `spectrum`, and `scalar` families, then combine present family means at `1:1:0.1`; adding scalar fields cannot increase the scalar family’s total objective share (lines 98–110).
- Emit unmatched partner rows only from row groups already fetched for anchor work, with deterministic same-split ownership; never launch standalone scans merely to increase unmatched coverage (lines 122–139).
- Treat unknown transforms, unsafe memory/locality/worker capacity, split leakage, duplicates, silent overflow, and unrecorded order changes as non-waivable. Every order-changing rollout bumps `SOURCE_ASSEMBLY`; stale stream positions fail while weights remain loadable (lines 141–157).

### Status at end of 2026-08-04

The ADR was **Proposed**, not accepted. Its day-end progress statement says the common metadata, config-carried token allocation, registry-driven packing, backward-compatible family objective, HF/Nanotron parity surface, and family logging were implemented, while anchor scouting/selection, stream topology, and source spokes remained pending; no expansion-related `SOURCE_ASSEMBLY` bump had occurred (ADR 0013 at `d12c94f`, lines 3–15).

`d12c94f` concretely changed behavior as follows:

- Config modalities gained `family`, `source`, `record_keys`, and explicit `token_ids`; required vocabulary size is derived and undersized configs raise ([`astro/src/astropt3/configuration_astropt3.py`](../../src/astropt3/configuration_astropt3.py), lines 90–110 and 116–170).
- New modalities consume stable non-overlapping blocks starting at 17, append above 63 after the reservation, and reject duplicate names/collisions ([`astro/src/astropt3/tokenization.py`](../../src/astropt3/tokenization.py), lines 39–116).
- `ObjectSequencer` dispatches from registry family/record keys rather than hardcoded image/spectrum names and serializes each configured token block ([`astro/src/astropt3/data/packing.py`](../../src/astropt3/data/packing.py), lines 178–257).
- The model reports modality and family losses; `loss_aggregation: family` applies the ADR formula, while the default `legacy_modality_mean` preserves old checkpoint behavior ([`astro/src/astropt3/modeling_astropt3.py`](../../src/astropt3/modeling_astropt3.py), lines 209–346).

### Verification and remaining gates

CPU tests pin old ids and exercise reservation exhaustion, vocabulary growth, and collision failure ([`astro/tests/test_tokenization.py`](../../tests/test_tokenization.py) at `d12c94f`, lines 20–78). Model tests check the exact `1:1:0.1` family formula and the legacy compatibility formula ([`astro/tests/test_model.py`](../../tests/test_model.py) at `d12c94f`, lines 82–117); save/load and eval tests also assert serialized modality metadata and family-loss reporting (`astro/tests/test_saveload.py`, `astro/tests/test_eval.py`). GPU tests were extended to compare HF and Nanotron family losses (`astro/tests/test_nanotron_gpu.py` in `d12c94f`), but `d12c94f` does not record a completed full verification run in its commit message.

The standardized plan still showed the spoke gates unchecked: reciprocal/provenance index invariants, transform roundtrips and rejection, split/ownership/resume/no-replay, bounded live byte/RSS audits, HF↔Nanotron parity, TP gradients, per-spoke GPU pilots, and a final combined pilot ([`astro/docs/mmu_modality_expansion_test_plan.md`](../mmu_modality_expansion_test_plan.md) at `d12c94f`, lines 9–92 and 125–184). Risks explicitly retained by the ADR include vocabulary/head and stream-order incompatibility, object overflow from retaining all valid scalars, scalar-head overhead, and PROVABGS circular/distillation targets that must not be reported as independent physical recovery (ADR 0013 at `d12c94f`, lines 81–96, 112–120, 186–200).

## Other consequential work: crossmatch-only stabilization

`545d5ac` traced the step-13,354 worker OOM to whole-cell unmatched-spectrum buffering, not repeated stream rebuilds: measured cells pinned 3.7–1,128.7 MiB per worker, up to 17.6 GiB across 16 workers. It introduced a 256 MiB unmatched buffer, emits overflow as read, and bumped order compatibility to `crossmatch_only_v2`. `bd4da5f` then bumped to `crossmatch_only_v3` and dealt shuffled cells round-robin across DP ranks, avoiding both `split_dataset_by_node`’s dp× read/discard fallback and truncation of 37/165 cells at dp=64 ([`astro/src/astropt3/data/streaming.py`](../../src/astropt3/data/streaming.py) at `e46d628`, lines 33–46, 174–185, 257–352, 393–454).

The loader now fails when a rank has fewer partitions than workers instead of accepting `datasets`’ warning and silently stopping surplus workers; configs are checked against `floor(165/dp)` ([`astro/src/astropt3/data/nanotron_loader.py`](../../src/astropt3/data/nanotron_loader.py) at `e46d628`, lines 282–320; [`astro/tests/test_nanotron_loader.py`](../../tests/test_nanotron_loader.py) at `e46d628`, lines 113–147). Unit coverage also verifies complete balanced DP coverage, unchanged/no-duplicate records under a zero-byte unmatched buffer, exact resume, and rank disjointness ([`astro/tests/test_streaming.py`](../../tests/test_streaming.py) at `e46d628`, lines 102–114 and 137–170). Commit messages record full CPU results of 130 passed (`545d5ac`) and 132 passed (`1f78dd4`, `bd4da5f`), with one network test also passing for `bd4da5f`.

ADR status changed accordingly: ADR 0006 became **Closed**, while ADR 0011 became **“Adopted, then collapsed to crossmatch-only”**. One mandatory index now defines a single scan that emits matched pairs, unmatched images, and globally unmatched owned spectra—no source weights, draw pattern, governor, standalone source, or fallback ([`astro/docs/adr/0011-skim-crossmatch-scans.md`](../adr/0011-skim-crossmatch-scans.md) at `e46d628`, lines 1–8 and 269–306; [`astro/docs/adr/0006-stream-mmu-upstream.md`](../adr/0006-stream-mmu-upstream.md), lines 1–14).

The live validation recorded by `e46d628` completed 600/600 steps on one A100 with six workers: 547 ms/step, 120K tokens/s/GPU, flat 19.4–23.1 GiB process-tree RSS late in the run, 399,193 objects with zero repeats and disjoint worker sets, and all 165 train shards visible (ADR 0011 at `e46d628`, lines 327–341). Follow-ups remain: dp=2×8 workers was only extrapolated, live resume was not exercised, and buffer overflow made the local mix lumpier—about 8% of micro-batches lacked image tokens versus ~1% before. Pipelining the next cell’s spectra against the current image scan was identified but deferred (lines 343–364). Row-group-level sharding also remains the likely way to lift the per-rank worker ceiling (lines 289–305).

## Bottom line

By midnight, the old crossmatch-only path had substantially stronger memory, sharding, configuration, and live evidence. The hub-and-spoke expansion had moved from exploratory pairwise builders to a governed rooted-star ADR plus reusable model/config plumbing, but **no new spoke was yet part of the corpus** and ADR 0013 correctly remained Proposed pending anchor, stream, source, GPU, resume, and learning evidence.

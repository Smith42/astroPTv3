# ADR 0013: Expand MMU through a Legacy-centred attachment graph

- **Status:** Proposed
- **Supersedes:** the sequencing and topology recommendations in
  `mmu_modality_expansion_plan.md` (deleted 2026-08-06; in git history)
- **References:** [ADR 0006](0006-stream-mmu-upstream.md),
  [ADR 0008](0008-scalar-modalities.md),
  [ADR 0011](0011-skim-crossmatch-scans.md), and PR #31
- **Implementation progress (2026-08-05):** the shared modality metadata,
  config-carried token allocation, registry-driven packing, family objective,
  and HF/Nanotron parity surface are implemented. Existing ids 0–16 remain
  frozen; the 47-modality source-graph config appends through id 144 and selects
  `loss_aggregation: family`. The [prototype anchor scout](../evidence/adr0013-anchor-scout-2026-08-04/README.md)
  capped inconclusive for both anchors; the project owner selected North first
  while retaining South as the expected second anchor. Every positional spoke is
  built with LSDB's default `crossmatch` rather than a hand-rolled reciprocal
  matcher (owner decision, 2026-08-05). The pointer-only schema-v2
  graph, source adapters, common spatial split, fetched-only unmatched policy,
  transforms, and all four ordered spokes now pass bounded CPU/live smoke
  evidence in [the source-spoke record](../evidence/adr0013-source-spokes-2026-08-05/README.md).
  Record order is tagged `legacy_north_source_graph_v1`; training-machine GPU,
  parity, resume/no-replay, and learning-evidence gates remain open, so this ADR
  remains Proposed.

## Question

How should AstroPTv3 add galaxies-with-hats, SDSS, PROVABGS, and HSC
without multiplying expensive image scans, hiding source provenance, breaking
spatial splits/resume, or letting many scalar targets dominate the objective?

## Decision

Adopt a **Legacy-centred rooted star per anchor**.

Choose one of Legacy Survey North or South for the first implementation. Expect
ultimately to include both nearly disjoint anchors unless measured marginal
coverage shows that the second adds little. DESI, SDSS, HSC, and
galaxies-with-hats match spatially only to their Legacy anchor, through LSDB's
default crossmatch (KdTree, one neighbour, 1 arcsec) — amended 2026-08-05 from
the reciprocal rule to the library default. The MMU collections supply the
10-arcsec margin that makes those joins exact at partition edges;
galaxies-with-hats publishes none, so its edge loss is measured, not assumed.
PROVABGS matches the anchor positionally on the same terms (owner decision,
2026-08-05: every spoke is an LSDB positional crossmatch, no identifier or
lineage joins anywhere). Its targets are still inferred from DESI spectra, so
they remain distillation/circular supervision; what changed is that the
PROVABGS row on a record is now selected by sky position rather than by being
the same DESI observation as that record's `spectra` span. Do not build
attachment-to-attachment spatial indexes and do not require complete N-way
matches.

A matched object record contains every available source-distinct modality.
DESI remains the baseline. Roll out one spoke at a time:

1. shared index/schema, registry, loss, scout, and validation support;
2. galaxies-with-hats;
3. SDSS;
4. PROVABGS;
5. HSC.

JWST is excluded initially. The project owner may reintroduce it through an
evidenced amendment to this ADR at any time; that amendment may reopen any
decision recorded here.

## Anchor selection

Compare Legacy North and South using adaptive, deterministic, stratified
scouts. Stratify comparable cells by footprint, attachment density, Galactic
latitude/crowding where relevant, and partition/row-group byte characteristics.

Measure non-padding autoregressive loss-target tokens per actual transferred
byte. Sample in batches until the bootstrap 95% interval has relative
half-width no greater than 10%, or 32 cells have been sampled for that anchor.
At the cap, report an inconclusive result rather than silently escalating to a
full scan.

Persist catalog revisions, stratum definitions, cell ids and order, seeds,
index-schema version, registry/transform hash, byte counts, raw per-cell
metrics, bootstrap method, source composition, object/packing lengths, RSS,
partition locality, and DP/worker capacity.

Unknown transforms, unsafe memory, invalid partition locality or worker
capacity, and split leakage disqualify a candidate. Otherwise the project owner
chooses the first anchor and records the evidence and rationale; no automatic
tie-break applies. After that prototype, the owner separately accepts or rejects
the second anchor from its marginal coverage and bounded-cost evidence.

## Modality identity and vocabulary

Every survey or product is a distinct modality with its own name, token block,
transform, positional contract, encoder/head, quality rules, and provenance.
DESI and SDSS spectra do not share a modality. Legacy and HSC images do not
share a modality.

Preserve every existing special-token id and meaning. Allocate unused blocks
within ids 17–63 first, then append new three-id modality blocks above 63 when
the existing reservation is exhausted. New configs and checkpoints explicitly
carry the enlarged vocabulary; old checkpoints are never silently upgraded.

## Numeric targets

Every galaxies-with-hats numeric science field whose value passes a fixed
source-specific row predicate becomes its own scalar modality. Each predicate
uses documented quality flags, finite uncertainty where available, valid
physical/domain ranges, and source guidance. A failed value omits only that
span.

Each accepted field has documented units, provenance, fixed transform and
inverse, missingness, valid range, quality predicate, and source revision.
Unknown or undocumented cases raise rather than guess.

PROVABGS inferred properties are ordinary loss-bearing scalar targets even
when the contributing DESI spectrum is present. This intentionally includes
pipeline distillation and circular supervision. Reports must not describe high
PROVABGS-target accuracy as independent physical recovery.

## Loss aggregation

Assign each modality to one fixed family: `image`, `spectrum`, or `scalar`.

For each batch:

1. mean losses across present modalities within each family;
2. weight the present family means `image:spectrum:scalar = 1:1:0.1`;
3. divide by the sum of weights for families present.

Adding another source or scalar cannot increase its family's total objective
share. Log every family and modality loss. This supersedes ADR 0008's
independent `0.1` weight per scalar for this corpus.

## Packing

Retain every valid scalar span; impose no speculative scalar cap. Whole objects
remain indivisible.

Report maximum, p95, and p99 object lengths, packed-row utilization,
objects per row, and composition-specific padding. If a valid object exceeds
the configured sequence length, raise. Never truncate fields, split the
object, or duplicate anchor observations silently.

## Fetched-only unmatched rows

Emit valid unmatched DESI, SDSS, and HSC rows only from partner row groups
already fetched to serve matched anchor work. Never add standalone partner
scans solely to increase unmatched coverage.

Use one common HEALPix train/validation split:

- matched connected components inherit their Legacy anchor's split;
- unmatched rows use their own coordinates;
- for each `(split, partner_partition)`, hash the stable partition path over
  the sorted referencing anchor cells in that split to select one owner;
- only that owner emits unmatched rows assigned to the active split;
- if no same-split owner exists, discard the row rather than scan again.

Ownership is stateless and part of the stream-order contract. Assert
exactly-once emission, train/validation disjointness, rank/worker disjointness,
and exact resume.

## Governance

Match indexes store both source ids and cells, separation, match radius, epoch
treatment, source revisions, and index-schema revision. Cross-survey string ids
are never join keys; every spoke joins the anchor by sky position.

Yield, characterized selection effects, provenance concerns, and bounded
throughput costs are advisory. The project owner may accept them only through
a dated evidence record containing the finding, rationale, consequence, and
owner.

Unknown transforms, unsafe memory, invalid locality or required worker
capacity, split leakage, duplicate emission, silent overflow, and undocumented
record-order changes are non-waivable.

Every order-changing rollout step bumps `SOURCE_ASSEMBLY`. Stale stream
positions are rejected; checkpoint weights remain loadable.

## Rollout and acceptance

Each spoke must pass:

1. index cardinality, uniqueness/reciprocity, provenance, deduplication, and
   connected-component split invariants;
2. schema/unit/band/grid, transform roundtrip, quality, missingness, and
   unknown-input checks;
3. repository CPU gates: pytest, parameter count, and decreasing-loss smoke;
4. stream ownership, rank/worker disjointness, resume, stale-state rejection,
   bounded-memory, and no-replay checks;
5. packing, objective, per-family/per-modality loss, and overflow checks;
6. a bounded live audit with pinned revisions and measured bytes, throughput,
   RSS, locality, composition, and yield;
7. a training-machine GPU pilot reporting matched-token learning curves,
   marginal references, old-family comparisons, scalar metrics, throughput,
   packing, and family shares.

The project owner accepts or rejects progression from the standardized evidence
and records the rationale.

After HSC, pin the complete corpus, indexes, registry, transforms, source
assembly, model config, seeds, and evaluation batches. Run the repository
verification gates and one combined-corpus GPU pilot. This ADR becomes
**Accepted** only when the project owner approves that final evidence package.
A production-scale pretraining run is not required.

## Consequences

- Image bytes are paid once per selected Legacy scan while attachments add
  training signal.
- Source provenance remains explicit; incompatible instruments are never
  normalized into a generic modality silently.
- Index count grows linearly with positional spokes rather than pairwise.
- Useful non-Legacy partner rows already present on the wire are retained
  without standalone scans.
- Objects outside the selected Legacy footprint or footprints are represented
  only when encountered through fetched-only partner reads.
- Vocabulary, modality heads, configs, stream ordering, and both HF/nanotron
  loss implementations change; old stream states are incompatible.
- Many scalar heads increase model and delimiter overhead, but their collective
  objective share remains bounded.

# MMU modality expansion test plan

**Governing decision:** [ADR 0013](adr/0013-legacy-centred-mmu-expansion.md)

A spoke may proceed only after every non-waivable check passes and the project
owner records a disposition for the standardized structural, live, and
learning evidence.

## A. Index and provenance

- [ ] Positional joins are only DESI/SDSS/HSC/galaxies-with-hats → selected Legacy.
- [ ] The only lineage join is PROVABGS→DESI.
- [ ] No attachment-to-attachment spatial index is accepted.
- [ ] Positional matches come from LSDB's default `crossmatch` (KdTree, one
      neighbour, 1 arcsec); many-to-one on the spoke side is expected, and the
      corpus-scale duplicate rate is measured before acceptance.
- [ ] Index rows store both ids/cells, separation, radius, epoch treatment,
      source revisions, and index-schema revision.
- [ ] Duplicate source ids and many-to-one outcomes are reported and handled
      by the declared policy.
- [ ] Cross-survey strings are never treated as lineage ids.
- [ ] The adaptive scout reproduces from its manifest.
- [ ] Scout stops at ±10% relative bootstrap half-width or 32 cells.
- [ ] A non-converged 32-cell scout reports `inconclusive`.

## B. Schema and transforms

For every source-distinct modality:

- [ ] Units, bands/grid, shape, mask, valid range, and provenance are pinned.
- [ ] Forward/inverse transform roundtrip passes within declared tolerance.
- [ ] Unknown source, band, grid, units, or field raises.
- [ ] Missing/NaN/failed-quality values omit only their span.
- [ ] galaxies-with-hats row predicates match published quality semantics.
- [ ] PROVABGS targets are labelled as inferred/distillation targets.
- [ ] SDSS padded ids normalize before matching.
- [ ] HSC five-band order and input dimension are asserted.

## C. Vocabulary, sequence, and packing

- [ ] Existing ids 0–16 retain their exact meanings.
- [ ] New modalities consume complete unused blocks in 17–63 first.
- [ ] Vocabulary growth above 63 appends ids without renumbering.
- [ ] Config/checkpoint serialization preserves the expanded token map.
- [ ] Every present source-distinct span has the correct delimiters,
      placeholders, values, positions, and head.
- [ ] Uniform span shuffle remains deterministic from object id and epoch.
- [ ] No scalar field is capped, sampled away, or truncated.
- [ ] Maximum/p95/p99 object lengths are reported.
- [ ] Packing utilization, objects/row, and composition padding are reported.
- [ ] An oversized whole object raises in HF and nanotron paths.

## D. Family loss

Test exact formulas for:

- [ ] image only;
- [ ] spectrum only;
- [ ] HSC/image-only fetched record;
- [ ] image + scalar;
- [ ] image + spectrum;
- [ ] image + spectrum + scalar;
- [ ] two image modalities;
- [ ] DESI + SDSS spectra;
- [ ] many scalar modalities.

For each case:

- [ ] Modalities are averaged within family.
- [ ] Family weights are 1:1:0.1.
- [ ] Denominator is the sum of weights for present families.
- [ ] Adding a scalar does not increase total scalar-family budget.
- [ ] Per-family and per-modality losses are non-zero when targets exist.
- [ ] HF and nanotron family losses agree within existing parity tolerance.
- [ ] Scalar starvation is visible in per-field logs.

## E. Streaming, split, and resume

- [ ] Matched components inherit the Legacy anchor HEALPix split.
- [ ] Unmatched DESI/SDSS/HSC rows use their own coordinates.
- [ ] Hashed owner is deterministic from split, stable partition path, and
      sorted referencing anchor cells.
- [ ] Each unmatched row is emitted at most once per epoch.
- [ ] Train and validation ids/positions are disjoint.
- [ ] DP ranks and loader workers emit disjoint object ids.
- [ ] A row with no same-split referencing owner is discarded without another
      scan.
- [ ] No standalone unmatched partner source is opened.
- [ ] Row-group reads remain bounded.
- [ ] RSS stays within the declared worker/process budget.
- [ ] Resume continues the exact object/micro-batch sequence at supported
      worker count.
- [ ] A stale `SOURCE_ASSEMBLY` state is rejected; weights remain loadable.
- [ ] Every order-changing spoke bumps `SOURCE_ASSEMBLY`.

## F. Per-spoke checks

### galaxies-with-hats

- [ ] Reciprocal 1-arcsec positional join; no cross-release string-id join.
- [ ] Every accepted numeric field has registry/test coverage.
- [ ] Failed row quality omits only the affected field.
- [ ] No unmatched galaxies-with-hats stream is emitted.

### SDSS

- [ ] SDSS→Legacy positional index passes reciprocity/cardinality checks.
- [ ] SDSS transform rejects unknown grids/units.
- [ ] DESI and SDSS use distinct modality ids, transforms, and heads.
- [ ] Fetched-only unmatched SDSS is exactly once and spatially split.

### PROVABGS

- [ ] PROVABGS→DESI lineage join only.
- [ ] Targets may coexist with DESI by explicit design.
- [ ] Evaluation labels circular/distillation metrics correctly.
- [ ] No unmatched PROVABGS stream is emitted.

### HSC

- [ ] HSC→Legacy positional index only.
- [ ] Five-band calibrated flux path is source-aware.
- [ ] HSC ivar/mask/shape/crop behavior is asserted.
- [ ] Legacy and HSC use distinct modality ids and heads.
- [ ] Fetched-only unmatched HSC is exactly once and spatially split.

## G. Required commands

Local/offline during development:

```bash
uv run pytest -m 'not network'
uv run python scripts/count_params.py
uv run python -m astropt3.train_smoke \
  --config configs/model/test-tiny.yaml --steps 50 --assert-decrease
```

Bounded network-marked checks:

```bash
uv run pytest -m network
```

Final repository gate:

```bash
uv run pytest
uv run python scripts/count_params.py
uv run python -m astropt3.train_smoke \
  --config configs/model/test-tiny.yaml --steps 50 --assert-decrease
```

Training machine only:

- [ ] HF↔nanotron forward/loss parity.
- [ ] TP replicated modality gradients.
- [ ] Kill/resume exact continuation.
- [ ] One bounded GPU pilot per spoke.
- [ ] One pinned combined-corpus GPU pilot after HSC.

## H. Standardized live/GPU report

For every spoke and the final combined pilot, record:

- [ ] source/index/config revisions and hashes;
- [ ] object counts and composition histogram;
- [ ] actual transferred bytes and loss-target tokens/byte;
- [ ] row-group/partition locality;
- [ ] throughput and stall distribution;
- [ ] process tree and per-worker RSS;
- [ ] object lengths and packing utilization;
- [ ] exactly-once/no-replay audit;
- [ ] family and per-modality learning curves at matched tokens;
- [ ] marginal/unconditional references for new targets;
- [ ] old-family comparison curves;
- [ ] field-appropriate scalar metrics and calibration;
- [ ] owner acceptance/rejection and rationale.

## I. Final acceptance

- [ ] Every spoke has an accepted evidence record.
- [ ] Complete corpus state is pinned.
- [ ] Repository verification gates pass.
- [ ] Combined GPU pilot completes without structural failure.
- [ ] Owner approves the standardized combined evidence.
- [ ] ADR 0013 status changes from Proposed to Accepted with evidence link.

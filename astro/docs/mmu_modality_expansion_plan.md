# MMU modality expansion implementation plan

**Governing decision:** [ADR 0013](adr/0013-legacy-centred-mmu-expansion.md)

**Order:** foundation → galaxies-with-hats → SDSS → PROVABGS → HSC

Do not restore the deleted family of one-off match-builder CLIs. Reuse one
shared builder and the existing registry, sequencer, collator, row-group
streamer, and source-assembly machinery.

## Phase 0 — shared foundation

### Index and scout

Update `scripts/build_match_index.py` to support declarative spoke
specifications:

- positional joins: DESI/SDSS/HSC → selected Legacy;
- lineage joins: galaxies-with-hats → Legacy, PROVABGS → DESI;
- the complete ADR index schema;
- adaptive stratified North/South scouting;
- pointer-only outputs, never local corpus materialization.

Reuse `partition_cells()`, `containing_partition()`, HATS partition discovery,
LSDB for offline positional matching, and PyArrow metadata/row groups.

### Streaming

Generalize `src/astropt3/data/streaming.py` from one Legacy-North×DESI index to
one selected anchor plus ordered spoke indexes.

Implement:

- source-specific decode adapters;
- matched record assembly by anchor id;
- fetched-only unmatched DESI/SDSS/HSC emission;
- own-coordinate HEALPix split for unmatched rows;
- hashed `(split, partner_partition)` owner selection;
- cumulative stream state and explicit source revisions;
- a `SOURCE_ASSEMBLY` bump.

Keep row-group-bounded reads, `owned_by_rank`, loader-worker sharding, no
train-time LSDB, and no shared mutable deduplication state.

### Modality registry and packing

Update:

- `src/astropt3/modalities.py`
- `src/astropt3/configuration_astropt3.py`
- `src/astropt3/tokenization.py`
- `src/astropt3/data/packing.py`
- `src/astropt3/data/nanotron_loader.py`

Add fixed modality family and source/record-key metadata. Replace hard-coded
`images`/`spectra` data dispatch with registry-driven source-distinct handling.
Reuse optional spans and deterministic uniform span shuffle.

Consume token ids 17–63 before enlarging the vocabulary. Preserve ids 0–16.

Retain the existing whole-object overflow error in `PackedCollator` and
nanotron micro-batch packing; add explicit uncapped-scalar regression coverage.

### Family loss

Implement the ADR formula in both weight-compatible model paths:

- `src/astropt3/modeling_astropt3.py`
- `nanotron/src/nanotron/models/astropt3.py`
- corresponding HF/nanotron config types and converters.

Log per-modality and per-family losses. Extend HF↔nanotron parity checks.

### Foundation tests

Add or extend:

- `tests/test_match_index.py`
- `tests/fake_mmu.py`
- `tests/test_streaming.py`
- `tests/test_packing.py`
- `tests/test_tokenization.py`
- `tests/test_model.py`
- `tests/test_scalar_modalities.py`
- `tests/test_nanotron_gpu.py`

Gate:

```bash
uv run pytest -m 'not network'
uv run python scripts/count_params.py
uv run python -m astropt3.train_smoke \
  --config configs/model/test-tiny.yaml --steps 50 --assert-decrease
```

## Phase 1 — galaxies-with-hats

1. Inventory finite numeric science fields and document each source-specific
   row predicate, units, provenance, transform/inverse, and missingness.
2. Build only the genuine Legacy lineage-id join; no positional fallback.
3. Add one source-prefixed scalar modality per accepted field in
   `data/scalar_registry.py`, modality configs, synthetic fixtures, and tests.
4. Add a new cumulative run config; never edit historical run configs.
5. Run structural, bounded-live, CPU, and training-machine GPU gates.
6. Record owner disposition before Phase 2.

Do not emit unmatched galaxies-with-hats rows: fetched-only unmatched behavior
in ADR 0013 is limited to DESI, SDSS, and HSC.

## Phase 2 — SDSS

1. Build SDSS→Legacy positional matches and normalize padded SDSS ids during
   index construction.
2. Add a source-distinct SDSS spectrum modality.
3. Extend `data/spectral.py` with explicit SDSS grid/unit validation,
   transform, inverse, masks, and unknown-grid errors.
4. Add fetched-only unmatched SDSS ownership/split behavior.
5. Extend synthetic fixtures, spectral, streaming, packing, model, and parity
   tests.
6. Run all spoke gates and record owner disposition.

## Phase 3 — PROVABGS

1. Build the genuine PROVABGS→DESI lineage-id join and mediate records through
   the accepted Legacy→DESI edge.
2. Inventory accepted numeric properties and add one source-prefixed scalar
   modality per field.
3. Mark every target's model-derived provenance and circular-supervision
   semantics in the registry and evaluation output.
4. Do not add a PROVABGS→Legacy positional join or unmatched PROVABGS stream.
5. Extend scalar, index, streaming, evaluation, and model tests.
6. Run all spoke gates and record owner disposition.

## Phase 4 — HSC

1. Build HSC→Legacy positional matches.
2. Add a source-distinct five-band HSC image modality.
3. Reuse verified `hsc-g/r/i/z/y` band-registry entries; validate the source
   product's units, shape, band order, ivar, mask, scale, and crop policy.
4. Generalize fixed image-shape/channel dispatch without weakening
   source-specific validation.
5. Add fetched-only unmatched HSC ownership/split behavior.
6. Extend physical normalization, streaming, packing, model, and parity tests.
7. Run all spoke gates and record owner disposition.

## Phase 5 — combined acceptance

Pin:

- source and index revisions;
- selected anchor and owner decision;
- accepted field registry and transforms;
- token map and model config;
- source assembly;
- seeds and fixed evaluation batches.

Run:

```bash
uv run pytest
uv run python scripts/count_params.py
uv run python -m astropt3.train_smoke \
  --config configs/model/test-tiny.yaml --steps 50 --assert-decrease
```

On the training machine, run nanotron parity/resume tests and one bounded
combined GPU pilot. Produce the ADR evidence package and obtain owner
acceptance before changing ADR 0013 from Proposed to Accepted.

## Documentation cleanup

Update, without rewriting historical decisions:

- `AGENTS.md`
- `README.md`
- `docs/architecture.md`
- `docs/training.md`
- `PLAN.md`
- `docs/mmu_crossmatch_research.md`
- ADR 0008 with a “superseded for expanded-corpus loss aggregation” note
- ADR 0011 with a “superseded for unmatched ownership” note

Historical configs and evidence remain immutable.

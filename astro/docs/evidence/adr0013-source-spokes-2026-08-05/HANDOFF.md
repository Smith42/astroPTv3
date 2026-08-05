# ADR 0013 source-graph pickup handoff

## Recovery point

Branch: `mmu-corpus-expansion`.

The North-first schema-v2 source graph is implemented and CPU/live-smoke verified for the existing DESI edge plus galaxies-with-hats, SDSS, PROVABGS, and HSC. South remains the expected second anchor; no South index was built in this change.

The tracked implementation includes:

- deterministic North/South anchor scout and pinned evidence;
- one pointer-only positional/lineage builder with reciprocal 1-arcsec matching and exact PROVABGS→DESI lineage;
- cumulative schema-v2 graph loading with pinned revision and join-shape validation;
- a multi-source stream with projected columns, common order-4 spatial split, deterministic fetched-partition owners, source-only emission for DESI/SDSS/HSC only, DP/worker sharding, and `legacy_north_source_graph_v1` resume tagging;
- SDSS padded-id and wavelength-padding normalization, source-specific log-grid validation, transform, and inverse;
- five-band HSC image decoding and rendering;
- 34 documented Galaxy Zoo fractions plus five PROVABGS inferred targets and SDSS redshift;
- matched HF/Nanotron 47-modality token maps (vocabulary size 145) and family loss;
- safe non-pickle probe-set caching.

## Local evidence

Tracked evidence is in [`evidence.json`](evidence.json) and [`README.md`](README.md). Raw pointer smoke artifacts remain untracked under `tmp/adr0013-source-index-v2-smoke/` and are pinned by SHA-256 in the evidence record. The unrelated root PDF was deliberately not staged.

Live first-target paths produced:

- galaxies-with-hats: 285 reciprocal matches; 10 finite morphology spans on the sampled matched row;
- SDSS: 51 reciprocal matches; canonicalized byte-literal id, 3,855 valid bins, distinct spectrum/redshift spans;
- PROVABGS: 4,848 exact lineage matches; five inferred/distillation spans; no unmatched emission;
- HSC: 3 reciprocal matches; fetched-only five-band source record, 144 patch tokens of width 320.

The temporary cumulative graph loads as 11 anchor cells and 5,199 edges across five partner sources. `open_stream` and the complete tiny model consumed a live target-bearing row from each new spoke. Actual `HfFileSystemFile._fetch_range` payload bytes and target-token ratios are recorded in `evidence.json`.

## Verified commands

From `astro/`:

```bash
uv run ruff check .
uv run pytest
uv run python scripts/count_params.py
uv run python -m astropt3.train_smoke \
  --config configs/model/test-tiny.yaml --steps 50 --assert-decrease
```

Results at handoff: Ruff clean; `146 passed, 1 skipped, 12 deselected`; every named size within tolerance; smoke loss `2.6695 → 0.2468`.

The new source-specific focused tests and live smoke passed after these gates. No GPU or training-machine command was run here.

## Next work — training machine only

1. Build and publish the full cumulative North index directory at pinned revisions. Keep one parquet per source/split and point `ASTROPT3_MATCH_INDEX` at that directory.
2. Run `configs/nanotron/astropt3-70m-jetformer-source-graph-check.yaml` in the Nanotron environment. It intentionally leaves `match_index: null` for the environment variable.
3. Run HF↔Nanotron forward/loss parity, TP=2 replicated-gradient, kill/resume, and worker-count/no-replay audits.
4. Measure full-corpus composition, maximum/p95/p99 object lengths, packing utilization, actual bytes/tokens, locality, RSS, and exactly-once train/validation/rank/worker disjointness.
5. Run one bounded GPU learning pilot per spoke and the combined graph; report family/per-modality curves and label PROVABGS results as distillation/circular supervision.
6. Obtain the project owner's final learning-evidence disposition. ADR 0013 stays Proposed until that approval.
7. Prototype retained South and accept/reject it from marginal coverage evidence; do not replace North or merge anchor ids.

## Known limits

- The selected-cell LSDB builder reports conservative 5–25 GB catalogue-compute warnings, although every bounded cell completed. Watch peak RSS before launching a full parallel build; keep builds serial if needed.
- Only the 34 documented Galaxy Zoo answer fractions are accepted. NSA/OSSY/ALFALFA/JHU-MPA/photo-z fields remain blocked until units, sentinels, flags, and fixed inverses are pinned.
- HSC publishes no image ivar/mask in the pinned row schema; none is fabricated.
- The CPU evidence characterizes first-target byte paths, not corpus-average yield or learning quality.

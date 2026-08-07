# ADR 0013 North source-spoke smoke evidence — 2026-08-05

This bounded CPU/network evidence verifies the pointer index and first target-bearing stream path for the complete North graph. It is not the required training-machine GPU acceptance.

## Result

| Spoke | Join | Matches | First-target bytes | New target tokens/GiB |
| --- | --- | ---: | ---: | ---: |
| galaxies-with-hats train | reciprocal 1 arcsec | 285 | 137,442,409 | 78.12 |
| SDSS | reciprocal 1 arcsec | 51 | 45,802,072 | 398.53 |
| PROVABGS | exact id via pinned DESI edges | 4,848 | 283,180,504 | 18.96 |
| HSC | reciprocal 1 arcsec | 3 | 62,727,089 | 2,464.94 |

The cumulative directory loaded as schema v2 with 11 anchor cells, 5,199 edges, five partner sources (existing DESI plus four additions), and `SOURCE_ASSEMBLY=legacy_north_source_graph_v1`. All smoke indexes had one-to-one source ids; positional separations were below one arcsecond. The SDSS smoke rebuilt padded byte-literal ids to canonical digit strings before indexing.

The byte gate instruments `HfFileSystemFile._fetch_range` and sums returned payload bytes from opening one pinned spoke index through the first emitted target-bearing record. It measures the actual parquet payload requested on that path, not catalogue size estimates; HTTP headers are excluded. These short first-hit ratios characterize the smoke path and are not corpus-average yield estimates.

## Source contracts exercised

- **galaxies-with-hats:** projected only IDs, coordinates, and the 34 accepted Galaxy Zoo fraction columns; a live matched row produced 10 finite scalar spans. See `research/galaxies_with_hats_scalar_inventory.md` (deleted 2026-08-06; in git history).
- **SDSS:** a live padded row trimmed non-positive wavelength padding to 3,855 bins, passed the `1e-4` dex grid check, and landed at normalized median absolute value 0.313. A fetched-only unmatched row produced distinct `sdss_spectra` and `sdss_Z` spans.
- **PROVABGS:** the lineage path produced the five accepted inferred/distillation targets (`LOG_MSTAR`, `Z_HP`, `Z_MW`, `TAGE_MW`, `AVG_SFR`) and no unmatched rows.
- **HSC:** the pinned row was five bands in `hsc-g/r/i/z/y` order with shape `5×160×160`; the source-distinct 96-pixel crop produced 144 tokens of width 320. The published row has no image ivar/mask fields, so none are fabricated.

## Reproduction

The positional artifacts were built with `scripts/build_match_index.py --cell` using the cells shown in `evidence.json`; PROVABGS used `--join-kind lineage` through the pinned DESI schema-v2 index. Raw smoke artifacts remain under `tmp/adr0013-source-index-v2-smoke/` and are identified by SHA-256 in `evidence.json` rather than committed as corpus data.

CPU tests cover reciprocal/cardinality logic, schema/lineage validation, source projections, fetched-only unmatched policy, spatial split/ownership, stream state tags, transforms/inverses, all 47 modalities, token-map parity with the Nanotron check config, and a complete tiny-model family-loss forward pass.

## Remaining non-waivable gates

Training-machine-only HF↔Nanotron parity, TP gradients, kill/resume under real workers, bounded GPU learning pilots, corpus-wide p95/p99 lengths and packing utilization, full-epoch exactly-once/no-replay audits, and owner learning-evidence disposition remain open. No GPU or training run was performed on this machine.

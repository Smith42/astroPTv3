# ADR 0013 prototype anchor scout — 2026-08-04

**Owner disposition (2026-08-05):** the project owner selected **Legacy North for the first prototype** and retained South as the expected second anchor. Both adaptive scouts reached the 32-cell cap without the required ±10% relative bootstrap half-width, so the statistical result remains **inconclusive**; North was selected from the directional byte-efficiency evidence rather than an automatic tie-break.

This is the prototype-selection scout requested before source adapters exist. It measures reciprocal positional-spoke matches per deterministic physical catalog byte. It does **not** claim the ADR's final non-padding target-token/actual-transferred-byte gate; that remains required once the new transforms and stream exist.

## Frozen inputs

| Source | Revision |
| --- | --- |
| Legacy North | `f634744d3c44dd4fde0dee3172d4887c5e3c31c0` |
| Legacy South | `982f321784d8daf2a5f983eb3ec560f31b90d667` |
| DESI | `9fd88ba48233cb9857701ce802a7eade2d4c4a88` |
| SDSS | `da175ca9bb931f4301ab950431922ba9f99089ba` |
| HSC | `69e1aa54a2604d002bce1e7d10e7ce1dbae85711` |
| PROVABGS | `a50ea5c2baacfea8d88c48baf78bd9c507c2f525` |
| galaxies-with-hats | `c0188b776c4ce6312a805a04cbc25c891a075933` |

- Plan: [`plan.json`](plan.json), SHA-256 `a7c80c7718a9a04e5fdfc461e11cb147a626acde4bbdd4ad208ef9d20559ad23`
- Raw per-cell evidence: [`evidence.json`](evidence.json), SHA-256 `324c1cb66efe8f14237c1298b1b2e8867075bd42d4ad104413de7a828aac9459`
- Registry/transform hash is recorded in the plan.
- Sample seed `130013`; bootstrap seed `130014`; 10,000 replicates.
- Eight within-anchor median strata: positional-spoke density × absolute Galactic latitude × anchor partition bytes. One-spoke/multi-spoke footprint is alternated within each stratum. Cells are ordered by SHA-256 and sampled in batches of four.
- Reciprocal 1-arcsec nearest-neighbour matching uses pinned HATS margins. Physical bytes are the anchor partition plus unique referenced partner partitions.

## Results

| Anchor | Cells | Status | Matches/GiB | Bootstrap 95% interval | Relative half-width | Raw matches | Sample bytes | Peak RSS |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| North | 32 | inconclusive | 378.09 | 221.09–531.44 | 41.0% | 8,684 | 22.72 GiB | 1.63 GiB |
| South | 32 | inconclusive | 90.30 | 44.86–139.78 | 52.6% | 3,467 | 36.48 GiB | 2.68 GiB |

Raw positional matches in the sampled cells:

| Anchor | DESI | SDSS | HSC |
| --- | ---: | ---: | ---: |
| North | 7,312 | 1,369 | 3 |
| South | 1,897 | 617 | 953 |

North's prototype point estimate is 4.19× South's and the reported intervals do not overlap. This is useful directional evidence for building North first, but it does not override the ADR's inconclusive-at-cap rule. South supplies materially different HSC-heavy coverage.

## Footprint-level marginal coverage

These are order-10 HATS footprint row counts, not confirmed matches:

| Spoke | In North footprint | In South footprint | North-only | South-only |
| --- | ---: | ---: | ---: | ---: |
| DESI | 573,844 | 518,706 | 573,844 | 518,706 |
| galaxies-with-hats | 2,339,165 | 4,773,354 | 2,324,169 | 4,758,358 |
| HSC | 174,100 | 366,404 | 106,911 | 299,215 |
| PROVABGS | 112,617 | 103,176 | 112,617 | 103,176 |
| SDSS | 230,468 | 434,040 | 222,907 | 426,479 |

The anchors are nearly disjoint for these spokes, so South is not redundant: it adds substantially more galaxies-with-hats, HSC, and SDSS footprint even though North is more byte-efficient in this prototype scout.

## Cross-release identity finding

- Legacy North publishes only an opaque integer `object_id`; the MMU conversion assigned its internal row index and omitted the source DR9 `RELEASE/BRICKID/OBJID`.
- Legacy South publishes DR10 `BRICKNAME-OBJID`; galaxies-with-hats publishes DR8 `BRICKID/OBJID`.
- Legacy documents the unique key as `RELEASE,BRICKID,OBJID`, with distinct release values for DR8, DR9, and DR10. A cross-release string-id lineage join is therefore invalid.
- On 2026-08-05 the project owner amended ADR 0013 to use a reciprocal 1-arcsec positional galaxies-with-hats→Legacy join. The prototype scout predates that amendment, so galaxies-with-hats and PROVABGS contributed to footprint stratification only and were excluded from its live metric.

## Owner decision

On 2026-08-05, the project owner approved **North first** because its prototype point estimate is 4.19× South's and the reported intervals do not overlap. South remains the expected second anchor because its marginal HSC/SDSS/galaxies footprint is large. This approves the North streaming/index prototype; it does not waive the final target-token/actual-byte gate.

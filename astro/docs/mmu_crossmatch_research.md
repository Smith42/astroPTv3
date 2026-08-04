# MMU HATS crossmatch research

**Status:** recommendation (2026-08-06)  
**Question:** Which additional Multimodal Universe (MMU) HATS catalogs should
AstroPTv3 crossmatch to add modalities and scalars without breaking its current
image/spectrum streaming contract?

## Executive recommendation

Keep the existing LegacySurvey-North × DESI EDR/SV3 corpus as the baseline.
The smallest useful next step is **not** a giant all-survey join: build a
pairwise *graph* and retain single-survey records, as the current loader already
does. The initial graph should be:

1. **Legacy Survey DR10 South × DESI EDR/SV3** — highest-value galaxy-scale
   expansion: wide four-band optical images, DESI spectra, and native
   photometry/redshift/extinction scalars. It substantially enlarges the
   overlapping imaging footprint, but is a separate phase because the HATS
   card is 123.2M rows / 61.1 TiB and its 160×160 four-band schema does not fit
   the current fixed `(3, 152, 152)` image tokeniser.
2. **HSC PDR3 Deep/UltraDeep × (Legacy South or SDSS)** — the first deliberate
   *heterogeneous-instrument* edge: deep five-band optical images versus wide
   imaging or spectra. This is scientifically valuable for depth/PSF/bandpass
   transfer and is bounded at 475k HATS rows, but requires an HSC image adapter
   and a second spectral grid/unit entry for SDSS.
3. **Gaia DR3 × APOGEE DR17**, after the galaxy path is stable — the best first
   stellar mix: Gaia BP/RP coefficient spectra + astrometry/photometry joined
   to APOGEE near-IR spectra and stellar parameters. Use quality flags and
   proper-motion-aware matching; do not treat the many pipeline-derived
   parameters as independent ground truth.
4. **TESS SPOC × Gaia**, only after a real irregular-time-series modality
   exists. It gives light curves plus astrometric/photometric scalars at useful
   scale, but is not a config-only extension.

Do **not** make PROVABGS or Galaxy Zoo 10 pretraining sources initially:
PROVABGS properties are SED-inference products from the DESI BGS observations,
and GZ10 is a small, confident-label RGB benchmark. Both are stronger held-out
or downstream evaluation resources than unconditional scalar supervision.
Do **not** spatially crossmatch PLAsTiCC: its card states that RA/Dec are random.

This ordering is consistent with AION-1's useful precedent, not a recipe to
copy: AION-1 uses pairwise associations among Legacy Survey, HSC, SDSS, DESI,
and Gaia rather than requiring one row with every modality. Its authors also
warn that quality/magnitude cuts, footprints, XP availability, and reciprocal
matches define a selection function. AstroPTv3 should retain its continuous,
autoregressive design and fixed physical transforms rather than adopt AION-1's
CDF-fitted scalar quantisation.

## Repo constraints that determine the choice

AstroPT3 currently streams native HATS/Parquet rows through one decoder,
using a precomputed image--DESI match index; no train-time `lsdb`, local
reshard, or cache is allowed. The adopted assembly scans matched LegacySurvey
cells and emits pairs plus unmatched image-only and spectrum-only rows. It
therefore needs stable `object_id`, `ra`, `dec`, and HATS partitions, all of
which the candidate HATS cards expose. A future edge should follow that shape:
an offline spatial match index containing source ids and HEALPix cells, then
native partition reads at training time.

The existing modality contracts are narrower than MMU:

| Current contract | Consequence for candidate catalogs |
| --- | --- |
| Images: `(3, 152, 152)` calibrated DES `g/r/z`; registry knows `des-*`, `hsc-*`, and JWST bands but the image encoder input is fixed. | Four-band Legacy South and five-band HSC require a new image modality/tokeniser configuration and adapter; neither is config-only today. RGB cards must not be treated as calibrated flux. |
| Spectra: DESI fixed 7,781-bin, 3600--9824 Å grid; `spectral.py` deliberately rejects unknown grids/units. | SDSS, APOGEE, Chandra, and MaNGA need a survey-specific normalisation/grid path before they can share or extend `spectra`. Do not silently interpolate or pass through. |
| Scalars: `Z`, `ebv`, and joint `g/r/z` photometry are one-token GMM modalities with fixed transforms and loss weight 0.1. | Add only measurements that have explicit units, quality semantics, and a non-leaky scientific role; keep instrument/model provenance and uncertainties. |
| Span order is deterministic from `(object_id, epoch)` and packing permits missing modalities. | Pair edges and unpaired source records are both valid; the phase should not require complete multi-way matches. |

Sources: [ADR 0006](adr/0006-stream-mmu-upstream.md),
[ADR 0008](adr/0008-scalar-modalities.md),
[ADR 0011](adr/0011-skim-crossmatch-scans.md), and
[`streaming.py`](../src/astropt3/data/streaming.py).

## MMU HATS inventory relevant to crossmatching

The official HATS collection describes MMU v1.5 as HATS/Parquet catalogs with
HEALPix partitions and `ra`/`dec`; its examples use LSDB nearest-neighbour
crossmatch at one arcsecond. HATS/LSDB culls to overlapping tiles and streams
one partition at a time, so feasibility is principally footprint, density,
proper motion, schema, and network cost—not a need to bulk-download either
catalog.

| Family / official HATS catalogs | Modalities and useful scalars | Crossmatch assessment |
| --- | --- | --- |
| **Wide optical imaging:** `mmu_ssl_legacysurvey_north`; `mmu_legacysurvey_dr10_south_21` | North is the current 3-band image source. South is 123,185,970 160×160 `g/r/i/z` flux, ivar, mask, PSF/scale images; optical and WISE fluxes, `EBV`, and shape descriptors. | South × DESI is the preferred next galaxy edge. Same coordinate keys/HATS mechanics, but 61.1 TiB makes a sampled match-index/throughput gate mandatory. Treat `i` and its fourth channel explicitly. |
| **Deep optical imaging:** `mmu_hsc_pdr3_dud_22.5` | 474,954 160×160 `hsc-g/r/i/z/y` flux/ivar/mask images, PSF/scale, calibrated photometry and shape tensors. | Good bounded transfer edge to Legacy or SDSS/DESI; small deep fields mean a small, non-representative overlap. Do not collapse HSC and Legacy same-named filters without survey provenance. |
| **Optical spectroscopy:** `mmu_desi_edr_sv3`; `mmu_sdss_sdss` | DESI: fixed 7,781 flux/ivar/LSF/lambda/mask plus `Z`, `ZERR`, `ZWARN`, `EBV`, and `g/r/z` fiber/total fluxes. SDSS: 806,176 variable-length 3650--10400 Å spectra, ivar/LSF/mask, `Z`, `ZWARNING`, velocity dispersion, and `u/g/r/i/z` synthetic fluxes. | DESI remains the anchor. SDSS connects galaxy and stellar populations, and is the recommended HSC spectral partner once its variable grids/units and quality flags have a dedicated adapter. |
| **Stellar spectral/scalar:** `mmu_gaia_gaia`; `mmu_apogee_dr17` | Gaia: 122,302,572 rows, 110 BP/RP coefficients + errors, G/BP/RP photometry, parallax/proper motions, radial velocity, and GSP-Phot parameters. APOGEE: 719,832 7,514-pixel H-band spectra plus J/H/K, radial velocity, quality flags, atmospheric parameters and 16 abundances. | High science diversity, moderate APOGEE size. Crossmatch must propagate Gaia positions to a common epoch or use a conservative, proper-motion-aware radius; the coordinate pair alone is unsafe for nearby/high-PM stars. |
| **Time-domain:** `mmu_tess_spoc`; `mmu_btsbot`; `mmu_foundation`; DES/PS1/Swift/SNLS/YSE/CfA/CSP transient cards; `mmu_plasticc` | TESS has 1,122,883 variable-length PDC-SAP time/flux/error curves and coordinates. BTSbot has ZTF science/reference/difference image triplets plus alert metadata; Foundation and related sets have multiband SN light curves and labels/scalars. PLAsTiCC has 7.0M simulated multiband curves. | TESS × Gaia is the future physical time-series edge. BTSbot is alert-level (not unique objects) and transient sets are small/selection-heavy. PLAsTiCC is **not spatially matchable**: its card explicitly says RA/Dec are random. |
| **Derived labels / benchmarks:** `mmu_desi_provabgs`; `mmu_gz10` | PROVABGS has 222,752 DESI-BGS SED posterior samples and inferred mass/SFR/age/metallicity/redshift. GZ10 has 17,736 RGB images, 10 morphology labels, redshift, and coordinates. | Retain for evaluation/calibration. PROVABGS labels are inferred from overlapping observations; GZ10's RGB composites are display products and its labels are intentionally clean/confident, so both cause target leakage/selection bias if treated as generic pretraining scalars. |
| **Specialised, later:** `mmu_manga`; `mmu_chandra_spectra`; VIPERS W1/W4; JWST CEERS/GDN/GDS/NGDEEP/PRIMER images and Grizli products | MaNGA bundles 10,735 IFU spaxel spectra, 96×96 four-band images, and derived maps; Chandra has 128,900 0.5--8 keV spectra plus hardness/variability scalars; JWST supplies small deep NIR image/spectral-field products. | High value only after a dedicated IFU/X-ray/JWST modality exists. These are not merely another row shape for the DESI spectrum or Legacy image pipeline. |

The collection also includes the small individual transient catalogs listed above.
The inventory is grouped by modality rather than pretending their schemas are
interchangeable. Counts are the current HATS-card figures, not expected
crossmatch yields.

## Recommended initial mixture and phases

### Phase A — increase the current galaxy corpus, one edge at a time

**Select:** Legacy Survey DR10 South × DESI EDR/SV3, preserving all South
image-only records, all DESI spectrum-only records, and reciprocal/one-to-one
pairs in a separate match index. Carry only the existing trustworthy scalar
set at first: DESI `Z` gated by `ZWARN`, `EBV`/`ebv`, and clearly documented
photometry. This remains image + spectrum + scalar, avoids an N-way join, and
is maximally adjacent to the working corpus.

**Required acceptance tests before data work:**

1. Sample an overlapping sky region and record match rate, separations,
   duplicate/many-to-one rate, row-group bytes, and source revisions.
2. Establish the South image units, band names/order, masks, scales, and crop
   policy from the card/original survey documentation; add no transform until
   this is known. The present 3-channel encoder makes this a new model/data
   configuration, not a loader flag.
3. Hold out complete HATS cells across *all* connected sources and deduplicate
   source ids across image-only/pair streams. Fit mix weights from realised
   object and token counts, with caps on small paired populations.
4. Audit scalar provenance: observed measurement, pipeline estimate, or
   derived SED product. Exclude a scalar from a loss whenever it is computed
   from a modality supplied in the same training example unless the goal is
   explicitly reconstruction of that product.

### Phase B — add an instrument-transfer bridge

**Select:** HSC × SDSS first; optionally add HSC × Legacy South as a second
pairwise edge after Phase A. This gives 5-band deep images, an independent
optical spectral survey, native photometry/shape scalars, and a graph bridge
to the large Legacy/DESI population. It mirrors the five-survey topology used
by AION-1 while respecting AstroPT3's pairwise, optional-modality loader.

**Gate:** separate source-aware image band registry and an SDSS spectrum
normalizer that validates each wavelength grid and flux unit. Keep HSC's
magnitude/full-depth/quality selection and its deep-field footprint in
provenance; it cannot be interpreted as an unbiased survey-wide sample.

### Phase C — create a stellar subgraph

**Select:** Gaia × APOGEE. Start with Gaia BP/RP coefficients, G/BP/RP fluxes,
parallax, and quality flags; add an APOGEE spectral modality only after its
H-band grid/units/masks have a validated transform. Treat astrophysical
parameters/abundances as labelled scalar candidates with uncertainty and
quality gating, not as universally observed facts.

**Gate:** epoch propagation, reciprocal/many-to-one policy, crowding analysis,
and a stellar-vs-extragalactic source-mixture policy. Gaia's HATS card includes
only the BP/RP subset of DR3; APOGEE is concentrated on the Galactic
plane/bulge, so neither source is a uniform sky sample.

### Phase D — add temporal structure

**Select:** TESS × Gaia after a variable-length, irregularly sampled,
multiband-aware light-curve representation exists. Resample neither cadence
nor flux units invisibly; retain time system, flux-error, cadence/sector, and
bandpass metadata. BTSbot and real transient sets become a separate
alert/event problem, not a drop-in extension of a persistent-object corpus.

## Crossmatch and data-governance rules

1. **Use sky coordinates as join keys, never string `object_id` across
   surveys.** Store both source ids, both HATS cells, angular separation,
   radius, epoch treatment, and catalog revision in the published match index.
   Use a nearest-neighbour, reciprocal one-to-one policy for the first pass;
   inspect density-dependent false matches before relaxing it.
2. **Partition-level spatial splits must apply to the connected component.** A
   source can appear as image-only and pair; assignment by one source only
   leaks its companion into validation. Reserve the same sky cells (with a
   boundary policy) in all sources and check source-id and positional
   duplicates.
3. **Record selection functions and licenses at the source level.** Current
   HATS cards are mostly CC-BY-4.0, but HSC requires its survey terms and
   attribution, SDSS/APOGEE and Gaia require their acknowledgements, and
   derived products require additional citations. Preserve source revision
   hashes because AstroPT3 currently floats upstream revisions.
4. **Normalise by physical provenance, not corpus statistics.** The repo's
   image/spectrum/scalar registries intentionally use fixed, invertible
   transforms and reject unknown bands/grids. AION-1 instead fits empirical
   scalar CDFs; that is informative as an alternative design but conflicts
   with AstroPT3's checkpoint portability constraint.
5. **Do not mistake paired rows for population data.** AION-1 explicitly
   identifies magnitude/quality cuts, footprint, Gaia XP availability, and
   reciprocal matching as selection effects; its reciprocal match can favour
   bright, isolated, well-centred sources. Report pair yield and distributions
   by magnitude, colour, redshift, Galactic latitude, S/N, and crowding before
   setting mixture weights.

## Primary sources

- [MMU HATS official blog](https://huggingface.co/blog/hugging-science/multimodal-universe-hats)
  — HATS/HEALPix/Parquet architecture, LSDB crossmatching, streaming mechanics,
  and the official HATS collection link.
- [Official MMU HATS collection](https://huggingface.co/collections/UniverseTBD/multimodal-universe-hats)
  — current catalog inventory.
- Official HATS cards: [Legacy South](https://huggingface.co/datasets/hugging-science/mmu_legacysurvey_dr10_south_21),
  [HSC](https://huggingface.co/datasets/UniverseTBD/mmu_hsc_pdr3_dud_22.5),
  [DESI](https://huggingface.co/datasets/UniverseTBD/mmu_desi_edr_sv3),
  [SDSS](https://huggingface.co/datasets/UniverseTBD/mmu_sdss_sdss),
  [Gaia](https://huggingface.co/datasets/UniverseTBD/mmu_gaia_gaia),
  [APOGEE](https://huggingface.co/datasets/hugging-science/mmu_apogee_dr17),
  [TESS](https://huggingface.co/datasets/UniverseTBD/mmu_tess_spoc),
  [PLAsTiCC](https://huggingface.co/datasets/UniverseTBD/mmu_plasticc),
  [PROVABGS](https://huggingface.co/datasets/UniverseTBD/mmu_desi_provabgs),
  [GZ10](https://huggingface.co/datasets/UniverseTBD/mmu_gz10),
  [MaNGA](https://huggingface.co/datasets/hugging-science/mmu_manga), and
  [Chandra](https://huggingface.co/datasets/UniverseTBD/mmu_chandra_spectra).
- [MMU paper, arXiv:2412.02527](https://arxiv.org/abs/2412.02527) — source
  corpus description.
- [AION-1 paper, arXiv:2510.17960](https://arxiv.org/abs/2510.17960) and its
  [official Hugging Face paper page](https://huggingface.co/papers/2510.17960)
  — five-survey pairwise graph, heterogeneous tokenisation, scalar inventory,
  transfer results, and selection-function cautions.

# galaxies-with-hats scalar inventory

**Pinned source:** [`Smith42/galaxies-with-hats@c0188b7`](https://huggingface.co/datasets/Smith42/galaxies-with-hats/blob/c0188b776c4ce6312a805a04cbc25c891a075933/README.md)

**Morphology definition:** [Walmsley et al. 2023, *Galaxy Zoo DESI*](https://arxiv.org/abs/2309.11425)

**Decision:** accept only the 34 documented Galaxy Zoo answer fractions in the first spoke. Block every other numeric metadata family until its upstream catalogue semantics and sentinels are pinned; no values are guessed from column names.

## Accepted fields

All accepted values are dimensionless predicted volunteer answer fractions. The pinned card declares them as `float32`; Galaxy Zoo DESI defines the percentage/fraction as the predicted proportion of volunteers selecting that answer. There is no per-row uncertainty column paired with these published fraction fields.

**Shared predicate:** field exists, is finite, and `0 <= value <= 1`. Missing/null/NaN/out-of-domain values omit that scalar span.

**Fixed invertible normalization:** `y = 2x - 1`; inverse `x = (y + 1) / 2`. This maps the documented closed unit interval to `[-1, 1]` without fitted corpus statistics.

- `smooth-or-featured_{smooth,featured-or-disk,artifact}_fraction`
- `disk-edge-on_{yes,no}_fraction`
- `has-spiral-arms_{yes,no}_fraction`
- `bar_{strong,weak,no}_fraction`
- `bulge-size_{dominant,large,moderate,small,none}_fraction`
- `how-rounded_{round,in-between,cigar-shaped}_fraction`
- `edge-on-bulge_{boxy,none,rounded}_fraction`
- `spiral-winding_{tight,medium,loose}_fraction`
- `spiral-arm-count_{1,2,3,4,more-than-4,cant-tell}_fraction`
- `merging_{none,minor-disturbance,major-disturbance,merger}_fraction`

The Galaxy Zoo decision tree makes some questions conditional, but the catalogue publishes a value for each answer field. The first-spoke predicate does not infer applicability or renormalize answers.

## Blocked fields

| Family | Examples | Reason blocked |
| --- | --- | --- |
| Identifiers/spatial/index | `_healpix_29`, `dr8_id`, `brickid`, `objid`, `ra`, `dec`, `galaxy_size`, `__index_level_0__` | Join/technical metadata, not target science scalars. |
| DESI/NSA photometry and structure | `mag_*_desi`, `petro_*`, `elpetro_*`, `sersic_*`, `redshift_nsa`, `elpetro_mass*` | The pinned card lists types but does not pin all units, validity flags, or sentinel rules. |
| OSSY | `redshift_ossy`, `log_l_oiii`, `fwhm`, `equiv_width`, `log_m_bh`, uncertainties | Missing/sentinel and quality semantics are not documented in the pinned card. |
| ALFALFA | `W50`, `W20`, `HIflux`, `SNR`, `RMS`, `Dist`, `logMH`, uncertainties | Units and acceptance flags are not documented in the pinned card. |
| JHU-MPA | `fibre_*`, `total_*`, percentile/flag fields | Catalogue-specific posterior and flag semantics are not pinned. |
| Photo-z | `photo_z`, `photo_zerr`, `spec_z`, `mass_*_photoz`, `sfr_*_photoz`, `ssfr_*_photoz` | Quality/missing rules and posterior interpretation are not pinned. |
| Derived convenience | `redshift`, `est_petro_th50_kpc`, colours | Provenance and reliability predicates are ambiguous across merged catalogues. |

## Implementation contract

Streaming emits accepted fields as `gwh_<original_name>`. `scalar_registry.py` recognizes only the 34 names above and applies the affine map above. Adding any blocked field requires a primary-source citation, explicit units, sentinel/quality predicate, and a fixed invertible normalization in this inventory first.

"""Deterministic synthetic records matching the verified MMU pilot schemas.

Every test and CPU smoke run uses these — no network, no real data. The
records mimic:

- ``UniverseTBD/mmu_ssl_legacysurvey_north``: ``image.flux`` float32
  (3, 152, 152) in LegacySurvey nanomaggies (galaxy cores ~0.1 nMgy, sky
  noise ~0.001 nMgy — the real pilot flux scale, so the physical band
  normalization's 0.01 nMgy arcsinh knee lands in the same regime as on
  real data), ``image.band`` = des-g/r/z, plus catalog scalars.
- ``UniverseTBD/mmu_desi_edr_sv3``: ``spectrum`` with 7781-bin ``flux``,
  ``lambda`` (3600-9824 A), ``ivar``, ``lsf_sigma``, ``mask``, plus ``Z``.

Images contain a redshift-correlated Gaussian blob and spectra a continuum +
emission line, so a smoke-trained model has real structure to learn. A
fraction of records are image-only (no ``spectrum``), matching the pilot
corpus where the DESI crossmatch covers only ~1/14 of the images.
"""

import numpy as np

IMAGE_SIDE = 152
IMAGE_BANDS = ["des-g", "des-r", "des-z"]
SPECTRUM_LENGTH = 7781
LAMBDA_MIN = 3600.0
LAMBDA_MAX = 9824.0


def make_record(index: int, image_only_fraction: float = 0.3) -> dict:
    """Build one deterministic synthetic record keyed by ``index``."""
    rng = np.random.default_rng(index)
    z = float(rng.uniform(0.01, 1.5))

    yy, xx = np.mgrid[0:IMAGE_SIDE, 0:IMAGE_SIDE].astype(np.float32)
    cx, cy = IMAGE_SIDE / 2 + rng.uniform(-10, 10), IMAGE_SIDE / 2 + rng.uniform(-10, 10)
    # large smooth structure so most patches are learnable rather than pure
    # sky noise (per-patch standardization turns flat sky into irreducible
    # N(0,1) targets); size still correlates with the redshift proxy
    sigma = 15.0 + 25.0 * z
    blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2)))
    amps = rng.uniform(0.01, 0.1, size=3).astype(np.float32)
    flux = amps[:, None, None] * blob[None, :, :]
    flux += rng.normal(0.0, 0.001, size=flux.shape).astype(np.float32)

    record = {
        "object_id": f"synth_{index:08d}",
        "ra": float(rng.uniform(0, 360)),
        "dec": float(rng.uniform(-1.6, 81.5)),
        "_healpix_29": int(rng.integers(0, 2**40)),
        "image": {
            "flux": flux.astype(np.float32),
            "band": IMAGE_BANDS,
            "psf_fwhm": float(rng.uniform(1.0, 2.0)),
            "scale": 0.262,
        },
        "z_spec": z,
    }

    if rng.uniform() >= image_only_fraction:
        lam = np.linspace(LAMBDA_MIN, LAMBDA_MAX, SPECTRUM_LENGTH, dtype=np.float32)
        # steep continuum: intra-patch slope dominates the noise, so the
        # standardized patch shape is learnable (flat continua standardize
        # to pure noise)
        continuum = 5.0 + 20.0 * (lam - LAMBDA_MIN) / (LAMBDA_MAX - LAMBDA_MIN)
        line_centre = 6563.0 * (1 + z) / (1 + 0.5)  # keep the line on-grid
        line = 20.0 * np.exp(-((lam - line_centre) ** 2) / (2 * 25.0**2))
        sflux = continuum + line + rng.normal(0.0, 0.05, size=SPECTRUM_LENGTH)
        record["spectrum"] = {
            "flux": sflux.astype(np.float32),
            "lambda": lam,
            "ivar": np.full(SPECTRUM_LENGTH, 1.0 / 0.09, dtype=np.float32),
            "lsf_sigma": np.full(SPECTRUM_LENGTH, 1.0, dtype=np.float32),
            "mask": np.zeros(SPECTRUM_LENGTH, dtype=bool),
        }
        record["Z"] = z

    return record


def record_stream(n: int, image_only_fraction: float = 0.3, start: int = 0):
    """Yield ``n`` deterministic records starting at ``start``."""
    for i in range(start, start + n):
        yield make_record(i, image_only_fraction=image_only_fraction)

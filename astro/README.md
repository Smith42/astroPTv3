# AstroPTv3

AstroPTv3 is a from-scratch suite of astronomical foundation
model configurations spanning 70M–12B parameters. Its Pythia-mirrored sizes
and checkpoint schedule support a research question: **how does learning from
continuous sky-survey measurements change as model size and training progress
increase?** The size ladder is an experimental design, not a claim that every
size has been trained or evaluated.

## Astronomy pilot

The current pilot draws from two [Multimodal Universe](https://huggingface.co/collections/UniverseTBD/multimodal-universe-hats)
HATS catalogs:

- **LegacySurvey North:** three-band (g/r/z) optical image cutouts, with
  associated image-side measurements.
- **DESI EDR SV3:** optical spectra sampled on a wavelength grid, with
  associated spectroscopic measurements.

The live LSDB path can supply image+spectrum matches and spectrum-only
sources, plus image-only sources recovered from LegacySurvey partitions
visited by the crossmatch. **This is not a complete pass over all LegacySurvey
images.** Available modalities vary by source and by model config; the model
learns from whichever spans a record actually carries. The ingestion design
is still experimental, so consult the [training guide](docs/training.md) and
active [run configs](configs/nanotron/) rather than treating any one stream
implementation as permanent.

## From measurements to predictions

A LegacySurvey image arrives as a 3×152×152 flux cube; the sequencer takes a
central 96×96 crop, physically normalizes each named band, and forms 144
8×8×3 patches. DESI spectra are normalized onto an AB-flux scale and form
31 patches of 256 bins with wavelength positions. There is **no per-patch
standardization**: the flow-based likelihood needs the record-to-token
mapping to retain patch information. Optional catalog scalars can become
one-token spans when included in the config.

A small 64-id vocabulary marks modality spans; it is **not** a text
vocabulary. The SmolLM3 decoder predicts the next continuous patch using a
per-modality JetFormer flow and Gaussian-mixture likelihood head. Whole
astronomical sources are packed into training sequences without splitting an
object. The nanotron fork (`../nanotron`) runs pretraining; the transformers
implementation in `src/astropt3/` supports checkpoint conversion, generation,
and probing. See [architecture](docs/architecture.md) for the model contract
and [the plan](PLAN.md) for the size ladder.

## Set up and verify

From this directory:

```bash
uv sync --extra dev
uv run pytest -m 'not gpu and not network'  # offline CPU checks
uv run python scripts/count_params.py       # nominal-size tolerance
```

Live data checks need Hub access:

```bash
uv run pytest -m network tests/test_lsdb_stream.py
```

Training needs nanotron, flash-attn, GPUs, and access to the live catalogs;
see [training.md](docs/training.md) for the environment, launch, and resume
instructions. The former synthetic-dependent `train_smoke` gate and
source-backed evaluation sweep are not currently available under the
experimental LSDB cutover; do not interpret the checks above as a completed
training-phase verification. Pure model-side evaluation and sample rendering
remain in `src/astropt3/eval/` and `scripts/generate.py`.

The [loader experiment log](EXPERIMENTS.md) records measured throughput and
coverage limits. [ADR 0015](docs/adr/0015-lsdb-infinite-stream-training.md)
documents the initial LSDB cutover; its original image-only source decision
is historical, not a description of every subsequent pilot config.

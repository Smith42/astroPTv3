# AstroPT3

SmolLM3-architecture multimodal astronomical foundation models, pretrained
from scratch on the [Multimodal Universe](https://huggingface.co/collections/UniverseTBD/multimodal-universe-hats)
with AstroPT-style continuous-token regression (NAIRR260009).

- **Architecture**: SmolLM3 decoder body (GQA, NoPE every 4th layer) with a
  64-id special-token vocabulary; images/spectra enter as affine-projected
  continuous patch tokens and leave through per-modality regression decoders
  (Huber next-token loss). See `src/astropt3/modeling_astropt3.py`.
- **Training** runs on the nanotron fork (git submodule at `../nanotron`,
  Phase 3+); the transformers implementation here is the release/probing
  artifact and the CPU test target.
- **Pilot data**: `UniverseTBD/mmu_ssl_legacysurvey_north` (3×152×152 flux
  cubes → 361 patch-8 tokens) × `UniverseTBD/mmu_desi_edr_sv3` (7781-bin
  spectra → 31 patch-256 tokens), lsdb-crossmatched offline.

## Setup

```bash
cd astro
uv sync --extra dev          # CPU-safe: model, packing, tests
uv sync --extra data         # + lsdb/hats (login-node data prep only)
uv sync --extra train        # + nanotron/flash-attn (training machine only)
```

## Develop / verify (no GPU, no network)

```bash
uv run pytest                          # unit tests (gpu-marked tests excluded)
uv run python scripts/count_params.py  # size table, ±10% assert
uv run python -m astropt3.train_smoke --config configs/model/test-tiny.yaml \
    --steps 50 --assert-decrease       # end-to-end CPU smoke on synthetic data
```

Model-size configs live in `configs/model/` (Pythia-mirrored 70M–12B).
The implementation plan (phases, verification, parallelism recipes) is
[`PLAN.md`](PLAN.md); per-phase PRs land on feature branches.

## Pilot data (login node, network)

Both MMU sources are HATS collections with margin caches, so the crossmatch
is a single lsdb call. Prepare once, then everything downstream is offline:

```bash
uv run --extra data python scripts/prepare_pilot_data.py
    # LEFT-crossmatch (1", nearest) → ~256MB parquet shards under
    # {root}/{train,val}/ + provenance.json; resumable per partition;
    # logs matched/image-only counts. Smoke: --cone RA DEC RADIUS_ARCSEC
uv run python scripts/compute_norm_stats.py
    # per-band asinh p1/p99 → normalization block of
    # configs/data/pilot_images_spectra.yaml (+ before/after histograms)
uv run python scripts/check_pilot_data.py --target-tokens-per-sec N
    # decoded-object sanity (~N(0,1) patches, λ range) + dataloader
    # throughput bench (want ≥2× training consumption)
```

Training streams the shards with `astropt3.data.mmu.MMUIterableDataset`
(`HF_DATASETS_OFFLINE=1`; DP-rank and DataLoader-worker sharded; keep
`num_workers ≤ n_shards / world_size`).

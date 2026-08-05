# AstroPTv3 labbook

This directory is the project labbook: durable notes recording what was
planned, run, and learned on each part of AstroPTv3. The authoritative
phase plan with all fixed decisions is [`../PLAN.md`](../PLAN.md); the
agent-facing rules are in [`../../AGENTS.md`](../../AGENTS.md). Code-level
architecture and the operational run guide live here as **reference** docs;
everything else is a **workstream** entry.

Convention: filenames are `lowercase_snake_case.md`. New workstream entries
are appended to the relevant section below with a one-line status so the
index stays auditable.

---

## Reference (stable background)

| Doc | What it covers |
|-----|----------------|
| [`architecture.md`](architecture.md) | Model design: SmolLM3 body + per-modality regression heads, tokenization, packing, two-implementation contract (HF release + nanotron training). Read first when touching the model. |
| [`training.md`](training.md) | Operational guide: environments, data, launching, checkpoint/resume, eval, and the traps already hit. The counterpart to `architecture.md`. |

---

## Workstreams (plan → run → diagnosis, in order)

### JetFormer tokeniser (additive `tokeniser: jetformer` path)

Per-modality `TinyFlow1D` + `GMMHead`, exact patch-space likelihood
`mean(NLL_GMM(z) − logdet)`; standardization skipped so the record→token
map stays invertible. Tracked as `astro-phase5`; gates in
`tests/test_jetformer_gpu.py`.

| Entry | Kind | Status |
| ------- | ------ | -------- |
| [`jetformer_plan.md`](jetformer_plan.md) | Plan (J1–J4) | J1–J3 implemented & CPU-tested; J4 (GPU verify + test run) on the reserved GH200 node. |
| [`jetformer_run_guide.md`](jetformer_run_guide.md) | Run record | First 70M run (`astropt3-70m-jetformer`, wandb `17k4i9n1`, 2×GH200, 20k steps) completed; image NLL +799→≈−38, reconstruction corr 0.69–0.90; grad-norm explosion + null-spectrum red flags flagged. |
| [`jetformer_noise_diagnosis.md`](jetformer_noise_diagnosis.md) | Diagnosis | Measured 2026-07-14 on step-20000 ckpt of the low-LR follow-up (`y3oak0l0`): two independent problems — optimisation drift and uncalibrated per-pixel noise generation. |

### Physical image normalization (port of galactiktok `feat/norm`)

Replace the data-driven Platonic-Universe asinh stretch with galactiktok's
physical, band-registry-keyed normalization for the image modality.
Spectra unchanged. Additive; gated on `uv run pytest` + the
`train_smoke --assert-decrease` smoke gate.

| Entry | Kind | Status |
|-------|------|--------|
| [`physical_norm_plan.md`](physical_norm_plan.md) | Plan (chunked, dependency-ordered) | Plan only; not yet implemented. Source of truth `../galactiktok` branch `feat/norm`. |

---

## MMU corpus expansion

| Entry | Kind | Status |
| --- | --- | --- |
| [`mmu_modality_expansion_plan.md`](mmu_modality_expansion_plan.md) | Plan | **Governing.** Order: galaxies-with-hats → SDSS → PROVABGS → HSC. |
| [`mmu_modality_expansion_test_plan.md`](mmu_modality_expansion_test_plan.md) | Test plan | Structural, live, CPU, and GPU acceptance gates for ADR 0013. |
| [`evidence/adr0013-anchor-scout-2026-08-04/README.md`](evidence/adr0013-anchor-scout-2026-08-04/README.md) | Evidence | Pinned North/South scout; both capped inconclusive, owner selected North first and retained South. |
| [`research/galaxies_with_hats_scalar_inventory.md`](research/galaxies_with_hats_scalar_inventory.md) | Research | Accepts 34 documented Galaxy Zoo fractions; blocks metadata with unpinned semantics. |
| [`evidence/adr0013-source-spokes-2026-08-05/README.md`](evidence/adr0013-source-spokes-2026-08-05/README.md) | Evidence | Bounded schema-v2 index, live stream, transform, and transferred-byte smoke for all four North spokes. GPU gates remain open. |
| [`mmu_crossmatch_research.md`](mmu_crossmatch_research.md) | Research | Superseded for sequencing by the plan above; still the reference for governance rules, per-catalog schemas and exclusions. |

---

## ADRs (decision records, `adr/`)

| ADR | Status |
| ----- | -------- |
| [`adr/0001-jetformer-inverse-variance-loss.md`](adr/0001-jetformer-inverse-variance-loss.md) | Rejected — ivar-weighted loss does not transfer to the jetformer likelihood head. |
| [`adr/0002-ivar-weighted-huber-loss.md`](adr/0002-ivar-weighted-huber-loss.md) | Proposed (Parked) — ivar-weighted Huber for the affine tokeniser. |
| [`adr/0007-physical-spectra-normalization.md`](adr/0007-physical-spectra-normalization.md) | Accepted — DESI spectra → AB nanomaggies, `arcsinh(f_ν/10 nMgy)` (`data/spectral.py`), the symmetric counterpart of `band_registry.py`. |
| [`adr/0013-legacy-centred-mmu-expansion.md`](adr/0013-legacy-centred-mmu-expansion.md) | Proposed — prototype one Legacy anchor, retain the second when marginal evidence warrants it, then add ordered source spokes. |

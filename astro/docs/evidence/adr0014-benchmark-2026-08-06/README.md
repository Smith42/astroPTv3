# ADR 0014 Phase 0/0b implementation evidence — 2026-08-06

Bounded CPU/network/GPU evidence for the ADR 0014 §3–§8 implementation. It is
not the §7c quality verdict: no arm has been trained to an `E_science`
comparison, and none of the gated experiments (§7c, §9, §10) were run.

## What was built

| ADR section | Landed |
| --- | --- |
| §3 instrumentation | `astropt3/data/telemetry.py` — `_fetch_range` byte probe (per-worker JSONL), main-process per-step counters, `TelemetryLoader` proxy. Env-gated by `$ASTROPT3_TELEMETRY_DIR`. |
| §2a MFU | Fork: `modality_flops_per_token`, `AstroPT3ForTraining.get_mfu_report`, decomposed logging in `train_step_logs`. |
| §4 composition | Per-modality loss-bearing token counts per step, reported by `bench_report.py`. Split change **cancelled** — see A1. |
| §5 fingerprint | `nanotron_loader.sequence_fingerprint` over assembly + revisions, modality config, image crop, band policy, `ar_replicas`, replica placement, span-order version, `seq_len`. |
| §6 projection | `_SOURCE_COLUMNS` leaf paths + `_spectrum_part` whitelist. |
| §7a/7b replay | `packing.span_order`, `ObjectSeq.order`, `PackedCollator.collate_rows`; `_replica_objects` (distinctness) and `_place` (decorrelation) in the loader. |
| §8 per-band | `ModalityConfig.channel_tokenization`/`band_order`, `ObjectSequencer._per_band_tokens`, `bench-north-5spoke-p0-perband.yaml`. |

## Measured

**§6 leaf projection, on real partitions** (instrument: `_fetch_range`
payload bytes; the §6 gate demanded a large single-row-group partition, not
only the 60.7 MB one, and SDSS measured independently):

| Catalogue | Partition | Row groups | Whole struct | Leaves only | Saving |
| --- | --- | ---: | ---: | ---: | ---: |
| DESI | `Norder=5/Npix=2356` (436.8 MB) | 1 | 436.9 MB | 229.6 MB | **47.5%** |
| SDSS | `Norder=4/Npix=268` (72.9 MB) | 1 | 72.9 MB | 30.7 MB | **57.9%** |

Both beat the 40% headline: the unread struct siblings drop with `ivar`.
Footer overhead is 0.01 MB and does not scale with partition size. Decoded
records are byte-identical across the change
(`test_spectrum_leaf_projection_drops_ivar_and_decodes_identically`).

**§4 validation split, on the live five-spoke index** (`north-v2-merged-5spoke`,
2,182,875 anchors, 5,488 cells): `split_of_cell` reserves **252 cells (4.6%)
and 4.0% of anchors** for validation. `VAL_PARTITIONS = 8` governs only the
retired DESI-only path. §4's premise was wrong; the split change and its
assembly bump are cancelled (amendment A1).

## Verification run

CPU gates, all green: `uv run pytest` (174 passed, 1 skipped),
`scripts/count_params.py` (all named sizes within ±10%), `train_smoke`
(0.2468 final vs 2.6695 initial over 50 steps).

GPU gates, all green on 2×A100 80GB: `pytest -m gpu` over
`test_nanotron_gpu.py`, `test_phase4_gpu.py`, `test_jetformer_gpu.py` —
12 passed in 16 min, covering HF↔nanotron parity, TP=2 replicated grads, the
50-step smoke plus conversion, the Pythia checkpoint schedule, and
kill/resume. These are the gates that guard the §7b packing rewrite: resume
now snapshots at the micro-batch boundary rather than the row boundary.

New CPU coverage: §6 byte-identity and the missing-leaf guard; fingerprint
separation by `ar_replicas`, `seq_len`, tokenisation policy and replica
placement, plus stream-state rejection; one-span records emitting no replica,
replica-order distinctness, the `n_spans!` cap, and no two replicas sharing a
packed row; `span_order` purity; per-band 432×64 shape, band-major order
against a hand-built reference, and its config validation; telemetry byte
accounting, step counters, and `loader_state_dict` reaching through the
wrapper.

## Benchmark

`scripts/run_benchmark.sh <arm>` runs a frozen 120-step window with
telemetry on and reports it with `scripts/bench_report.py`. Arms:
`bench-north-5spoke-{b0,b1,b1d,p0-perband}`. Each repetition starts from the
same frozen state — a rep that resumed would sample a different part of the
cell order and stop being comparable.

**Throughput results (one rep per arm) are in `benchmark.md`.** Headline:

| Arm | mean s/step | MFU | MFU_busy | stall_share | packing |
| --- | ---: | ---: | ---: | ---: | ---: |
| B0 (replicas 1) | 8.53 | 2.42% | 27.29% | 90.9% | 0.979 |
| B1 (2, adjacent) | 4.06 | 4.98% | 21.63% | 75.9% | 0.954 |
| B1-D (2, decorrelated) | 3.81 | 5.42% | 27.67% | 80.0% | 0.980 |
| P0 (per-band) | 3.52 | 5.81% | 25.14% | 76.2% | 0.970 |

B0 runs at 9.06% of its own stall-free MFU, against the ADR's "roughly 9%"
prediction. Decorrelation does not cost the replay gain — B1-D beats B1 on
every axis, and the decomposition attributes B1's deficit to packing, not to
model shape. `E_values` is flat across all arms while `E_AR` doubles under
replay and rises 2.77× under per-band, which is what those two metrics are
defined to do. MFU is only ever reported decomposed; §11 refuses the headline
number as an acceptance criterion, and none of these windows is an
`E_science` verdict.

## Remaining non-waivable gates

§7c's quality A/B to an `E_science` verdict at matched bytes and matched wall
clock, §8's per-pixel validation loss and probe comparison, §9's cache
prototype and its stable-ownership prerequisite, §10's worker sweep, and the
owner's learning-evidence disposition all remain open.

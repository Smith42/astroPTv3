# ADR 0014 §3 frozen benchmark — B and P arm windows, 2026-08-06

120-step windows on the five-spoke North corpus, dp=2 × 8 loader workers on
2×A100 80GB, `north-v2-merged-5spoke` index, seed 42, each arm started from
the same frozen state. One repetition per arm. Instrument: ADR 0014 §3
(`_fetch_range` byte probe + per-step counters); report:
`scripts/bench_report.py`.

## Result

| Arm | Replay | Placement | mean s/step | p50 | p95 | p99 | slow >60 s | MFU | MFU_busy | stall_share | packing | E_values | E_AR | E_AR primary |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 1 | — | 8.53 | 0.72 | 51.48 | 96.77 | 6 | **2.42%** | 27.29% | 90.9% | 0.979 | 74,651 | 388 | 388 |
| B1 | 2 | adjacent | 4.06 | 0.92 | 7.76 | 46.42 | 1 | **4.98%** | 21.63% | 75.9% | 0.954 | 73,791 | 775 | 389 |
| B1-D | 2 | decorrelated | 3.81 | 0.70 | 10.43 | 34.32 | 1 | **5.42%** | 27.67% | 80.0% | 0.980 | 74,908 | 786 | 395 |
| P0 | 1 | per-band | 3.52 | 0.71 | 6.59 | 42.43 | 1 | **5.81%** | 25.14% | 76.2% | 0.970 | 70,984 | 1,076 | 1,076 |

`E_values` = distinct target values per MiB fetched; `E_AR` = loss-bearing AR
tokens per MiB. The decomposition identity
`MFU = MFU_busy × (1 − stall_share) × utilisation_packing` reproduces each
arm's MFU to nine digits (checked in the report and in
`tests/test_bench_report.py`).

## What it says

**The ADR's baseline characterisation is confirmed to two figures.** B0 sits
at 2.42% MFU against a stall-free 27.29% — **9.06% of its own stall-free
MFU**, where §2a predicted "roughly 9%". Time-weighted `stall_share` is
**90.9%**, against the ADR's implied ~91%.

**Replay is worth ~2.2× and the metrics stay honest.** `E_values` is flat
across all three arms (74,651 / 73,791 / 74,908) exactly as §2 requires —
replay re-presents observations, it does not buy new ones — while `E_AR`
doubles (388 → 775 / 786) with the primary component held constant
(388 / 389 / 395). The replay factor measures 1.99×, not 2.00×: §7a's
distinctness rule correctly withholds a replica from one-span records.

**Decorrelation does not cost the gain — it adds to it.** §7c required that
"most of the measured MFU/stall gain survives decorrelation". B1-D beats B1
on every axis: MFU 5.42% vs 4.98%, mean step 3.81 s vs 4.06 s, p99 34 s vs
46 s. The decomposition says why, which is the whole reason for insisting on
it: adjacent placement stacks a record's replicas into one row, fragmenting
the row tail, so packing falls to 0.954 and `MFU_busy` to 21.63%. B1-D's
`MFU_busy` of 27.67% matches B0's 27.29% — the same model shape, as it must
be. An undecomposed comparison would have shown B1-D ahead and attributed it
to the wrong cause.

**Exactly-once holds under replay.** All arms: zero duplicate lines across
43,030–115,170 emitted sequences.

**Per-band (P0) buys its stall reduction and pays the predicted price.** §8
said per-band "moves two factors in opposite directions"; both moved, and the
decomposition sizes them:

- `stall_share` **90.9% → 76.2%** — 3× the tokens per object means fewer
  unique objects per step and fewer cell-boundary fetches. This is the win.
- `MFU_busy` **27.29% → 25.14%**, a 7.9% relative loss. Per amendment A5 the
  modality heads are exactly cost-neutral under per-band (FLOPs/token falls
  to 1/3 as tokens triple), so this is *not* less arithmetic — it is kernel
  efficiency at width 64 plus a longer attention span per object.
- `utilisation_packing` 0.979 → 0.970: essentially unaffected. Measured
  average is 355 non-padding tokens per emitted sequence against
  `seq_len: 4096`, so the length gate §8 asked about is comfortable.

Net MFU **2.42% → 5.81%**, a 2.4× gain at identical bytes. `E_AR` rises
388 → 1,076 (2.77×, short of the theoretical 3× because scalar spans do not
triple) while `E_values` stays flat at 70,984 vs 74,651 — the finer
factorisation, not more information, exactly as §8 states. **None of this is
an acceptance signal**: §8 accepts only on `E_science`, and no quality
comparison has been run.

## Caveats

- **The B0 window is not composition-matched to the later windows.** B0 drew
  2,306 MiB of HSC (2.72% of its loss-bearing tokens); the B1, B1-D and P0
  windows drew 0.1 MiB (~0%). This is precisely the 160× composition swing §4
  warns about, now measured rather than inferred. Every comparison *against
  B0* — including per-band's headline 2.4× — is therefore **indicative
  only**. B1 vs B1-D *are* well matched (near-identical composition and byte
  mix), so the decorrelation conclusion is the solid one here.
- One repetition per arm. §3 asks for several, plus an HSC-enriched window;
  neither is done.
- The wall-clock gain (2.1–2.2×) is smaller than the 3.1× the ADR measured
  over its first 117 steps. Different window, different cell order, shared
  hub at a different time — per the ADR's own caveat, the direction is the
  finding, not the magnitude.
- MFU's backbone term scales linearly with the non-padding fraction although
  attention is quadratic in `seq_len`, so these are upper bounds
  (amendment A4).

## Not evidenced here

No `E_science` verdict: these are throughput windows, not quality runs. §7c's
quality A/B at matched bytes and matched wall clock, §8's per-pixel
validation loss and probe comparison, and the jetformer flow/GMM behaviour
check at `input_size: 64` all remain open. `ar_replicas: 4` stays refused
(§11) — B1-D passing the distinctness and correlation gates is necessary, but
the quality comparison is not done. Arm P1 (per-band × the §7 winner) is not
run: §8 gates it on B1-D and P0 both passing, and passing means `E_science`.

## Reproduce

```bash
REPS=1 bash astro/scripts/run_benchmark.sh bench-north-5spoke-b0    # or b1, b1d
```

Raw telemetry and per-rep reports live under
`astroPTv3_checkpoints/bench-north-5spoke-*/` (`telemetry-rep1/`,
`report-rep1.json`); they are run outputs, not committed corpus data.

# MMU streaming throughput audit (2026-07-21)

Follow-up to the [shakeout handoff](2026-07-21-streaming-shakeout-handoff.md).
Attempt 3 died at step 3989/20000 on a walltime timeout — which follows
directly from the finding below, not from a new failure mode.

## TL;DR

The streaming code is fine — the pipe is too small. gpu5 has a **1 Gbit
NIC** (`/sys/class/net/eno8303/speed`), this corpus costs ~250 KB per image
on the wire, and feeding the 70M model at its compute-bound 0.55 s/step
would need ~3 Gbit/s. No software change reaches that on this box. What
software can do is saturate the link we have — attempt 3 used only ~37% of
it — so `num_loading_workers` is back to 8/rank. Expected: ~4.6 s/step →
~1.8–2 s/step, the physical floor here.

## The "IO downtime between batches", diagnosed

`train.log` shows 3 fast steps (~0.55 s) then one ~15 s step, repeating
with period 4: the DataLoader round-robins over 4 workers, each of which
needs ~17 s (~31 obj/s) to build one micro-batch. Mean step time was
~4.6 s (~28 K tok/s) **from step 0** — the ~240 K tok/s readings just
after each launch (including the handoff's "healthy-throughput reference")
were prefetch buffers draining, never sustained throughput. The run was
always data-bound; 20k × 4.6 s ≈ 25 h blew the walltime.

The arithmetic:

- One micro-batch = 16 × 4096 tokens ≈ ~420 images ≈ **105 MB/rank/step**
  (weights 0.8/0.2 images/spectra without pairs, ~125 tokens/record).
- `image.flux` is 50 MB of every 56 MB row group (parquet metadata, checked
  live): ~250 KB/image compressed — noisy float32, incompressible. There
  are **no fat unused columns** to project away; the remaining columns are
  scalars, and the spectra side uses everything it downloads.
- Compute-bound (0.55 s/step, dp=2) ⇒ ~380 MB/s ≈ 3 Gbit/s. NIC: 1 Gbit.
- Link-saturated floor: ~125 MB/s ⇒ ~1.7–1.9 s/step ≈ 70–77 K tok/s.
- Observed: ~46 MB/s (37% of link) at 4 workers/rank — the halving that
  was DNS insurance cost ~2.5x.

## Audit verdict on the code

Keep it. The `datasets` stack already does the heavy lifting correctly
(per-worker sharding, threaded interleave prefetch, row-group reads, exact
resume), `streaming.py` is ~400 lines with nothing speculative, and the
rebuild-from-snapshot retry fired exactly once in 4k steps. Moving off
`datasets` buys nothing: the bottleneck is the NIC, not the library.
Secondary (not acted on, YAGNI until the link is saturated and we're still
short): the per-image arrow → python-lists → numpy decode chain costs
~10–20 ms/image; re-publishing the catalog pre-cropped to 96×96 would cut
wire bytes ~60% but is a data-engineering job, not a code change.

## Changes

- `configs/nanotron/astropt3-70m-jetformer-mmu-nopairs.yaml`:
  `num_loading_workers` 4 → 8. Both DNS-triggered failure modes (ENOTFOUND
  and hub's `close_session` race) are survived by the loader's retry, so
  run enough workers to fill the link.
- Deleted `scripts/spike_profile_streaming.py`: self-described throwaway,
  and broken (imports `row_to_record`/`spectra_row_to_record`, which no
  longer exist).

## Relaunch decision

Checkpoint 3000's `dataset_state` was saved with 4 workers and the loader
refuses a mismatched resume. Resume at 4 workers ≈ 20.5 h remaining;
fresh start at 8 ≈ 10.5 h for all 20k — **fresh is faster in wall clock
than resuming slow**. Archive the run dir (as with attempt 2) and start
fresh. Set the walltime from ~1.9 s/step, not 0.55.

## Shared-machine etiquette: DNS vs. the link

Worker count barely moves DNS load. The streams talk to two stable
hostnames (`huggingface.co`, `cas-bridge.xethub.hf.co`) over pooled
keep-alive connections, so lookups happen at worker startup, at connection
drops, and after blips — not per request. The observed ENOTFOUND events
were two isolated incidents ~4 h apart (one at 16 workers, one at 8),
i.e. router flakiness, not saturation we caused; both are now survivable
via the rebuild retry. There is no good userspace DNS cache without root —
don't build one.

The real courtesy issue is the **1 Gbit uplink**: at 8 workers/rank the
run sits near 100% of the node's internet for ~10 h, which other users
will feel in every download. The knob is `num_loading_workers` (6/rank
≈ ~75% of link, proportionally slower run). Simplest fix: run overnight,
or agree it with the other users — don't leave throughput on the table
by default.

## For real pretraining, this shrinks on its own

Bandwidth need scales with tokens/s and bigger models are slower per
token: 70M at ~116 K tok/s/GPU is the worst case (~220 MB/s/GPU); a 1B
model at ~10 K tok/s/GPU needs ~20 MB/s/GPU. Planning number for the
training cluster: aggregate NIC bandwidth ≈ 1.9 KB/token × global tok/s.
Check that against the cluster uplink before the big runs; for 1B+ it is
unlikely to bind.

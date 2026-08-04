# MMU streaming shakeout — handoff note (2026-07-21)

Status: 20k-step `astropt3-70m-jetformer-mmu-nopairs` run on gpu5 (2× A100) is
DOWN, killed at step ~368 by failure mode #3 below. Nothing is running; GPUs
are free. Relaunch: `bash astro/scripts/launch_mmu_nopairs_gpu5.sh` from the
repo root (config resumes from `latest.txt` — checkpoint 256 exists, but see
"resume caveat" at the bottom).

## What the evening looked like

Three distinct failure modes, found and (two) fixed in order:

### 1. Loader clamped to one worker (FIXED, committed 5adb304)

Symptom: with `num_loading_workers: 8` × dp 2, effective throughput was one
worker's worth.

Cause: double sharding. `PackedMicroBatches._mmu_records` manually called
`split_dataset_by_node(shard=rank*nw+wid, world_size=world*nw)` (16 ways), but
`datasets.IterableDataset._iter_pytorch` **also** shards across DataLoader
workers automatically whenever it detects one. After our manual split the
interleave's `num_shards` = min(images 5480, spectra 298) = 298, and
`298 % 16 ≠ 0`, so datasets abandoned shard-splitting for a 1-shard
`StepExamplesIterable`; its per-worker split then handed shard 0 to worker 0
and stopped workers 1–7 ("Number of dataset shards < num_workers (1<8)").

Fix (`nanotron_loader.py`): shard by DP rank only (`298 % 2 == 0` → clean
`shard_data_sources`, ~18 shards per worker) and leave the worker split to
datasets (`shift_ex_examples_rngs` also decorrelates the interleave RNG per
worker). Verified offline: all workers stream, all 35 loader/streaming tests
pass. Residual constraint: `min_source_partitions % dp == 0` or datasets falls
back to example-stepping at the rank level too (fine at dp∈{1,2}: 298 even).

### 2. `httpx.ConnectError: Name or service not known` (mitigated, committed 5adb304)

This box resolves DNS through a single LAN router (`192.168.2.254`,
`/etc/resolv.conf`, NetworkManager-managed, no sudo). Under 16 workers of
parallel HF traffic it intermittently returns ENOTFOUND for seconds at a time.
`HfFileSystemFile.read` retries only once, so a blip killed the run (it also
killed the pre-datasets 512-step run at 20:59, same error).

Fix: `PackedMicroBatches.__iter__` catches `(httpx.HTTPError, OSError)` from
`next(records)` and rebuilds the record stream from the last per-record
snapshot (`prev_state`) — exact, no replay/skip, uses the same
`load_state_dict` path as checkpoint resume. 60 retries, 5s→120s exponential
backoff, counter resets on any successful draw (~2h sustained-outage budget).
Covered by `test_mmu_stream_survives_a_transient_network_error`
(bit-identical batches after a mid-stream ConnectError).

### 3. `RuntimeError: Cannot send a request, as the client has been closed` (OPEN — killed step ~368)

Timeline in `train.log`: 00:11:31 DNS blip → huggingface_hub's `http_backoff`
logs "thrown while requesting … Retrying in 1s [Retry 1/5]" → on
`httpx.ConnectError` the backoff calls **`close_session()`** (hub 1.24.0,
`utils/_http.py:513`) which closes the **global shared httpx client** →
datasets' interleave prefetch threads (`ThreadPoolExecutor` in
`RandomlyCyclingMultiSourcesExamplesIterable._iter_arrow`) are concurrently
using that same client in the same worker process → one of them dies with
`RuntimeError: Cannot send a request, as the client has been closed.` →
worker 7 exits → torch elastic tears the run down.

Key facts:

- It's a plain `RuntimeError`, **not** `httpx.HTTPError`/`OSError`, so the
  fix-#2 retry wrapper did not catch it (0 "rebuilding the stream" lines in
  the log).
- `close_session()` is documented to recreate the client on the next
  `get_session()` — so only threads holding the old client die; a rebuild
  would recover.
- This is a huggingface_hub 1.24.0 concurrency race: `close_session` from
  backoff while other threads use the shared client. Ironically the same
  class of bug that killed the hand-rolled reader (see `streaming.py`
  docstring, "died at step 149 … shared-httpx-client lifecycle bug").

## Next steps (tomorrow, in order)

1. **Extend the retry catch** in `PackedMicroBatches.__iter__` to
   `RuntimeError` message-matched on "client has been closed" (broad
   RuntimeError would mask real bugs). After a rebuild, `get_session()`
   hands out a fresh client, so the exact-resume path should ride it out.
   Extend `test_mmu_stream_survives_a_transient_network_error` with a
   RuntimeError case.
2. **Expect it to recur often** here: every DNS blip now risks the close
   race, because backoff's `close_session()` fires on any ConnectError while
   2–3 prefetch threads per worker hold the client. If the rebuild rate
   becomes silly, consider:
   - filing/patching upstream (backoff shouldn't close a shared client it
     doesn't own), or
   - pinning `huggingface_hub` differently in `gpuenv` (check whether a
     requests-based 0.x avoids the global-httpx-client lifecycle), or
   - reducing DNS pressure (fewer workers = fewer parallel queries; or a
     userspace resolver cache — no root on gpu5).
3. **Resume caveat**: checkpoint 256's `dataset_state/` was saved with the
   current (rank-only) sharding, so `bash
   astro/scripts/launch_mmu_nopairs_gpu5.sh` resumes cleanly in principle —
   but the run only lived 368 steps; a fresh start costs ~nothing and keeps
   the 20k curve contiguous. If resuming, verify the loader state loads
   without the num_workers mismatch error.

## Artifacts

- Branch `adr-0006-stream-mmu-upstream`, HEAD `5adb304` (pushed): worker-clamp
  fix + network-rebuild retry + gpu5 launch script + config resume-path fix.
- Archived pre-datasets run (hand-rolled reader, 512 steps, died on the same
  DNS blip):
  `…/astroPTv3_checkpoints/astropt3-70m-jetformer-mmu-nopairs.prereader-512`
- Current run dir: `…/astroPTv3_checkpoints/astropt3-70m-jetformer-mmu-nopairs`
  (`train.log`, checkpoints 1–256, per-worker `objects.log.dp{0,1}.w{0..7}`)
- Healthy-throughput reference: ~250K tok/s, 0.5s/step, both GPUs 100%, all
  16 worker logs growing; loss goes negative (jetformer exact-likelihood NLL —
  expected).
- Env: `/beegfs/general/mjsmith/gpuenv` (torch 2.8.0+cu128, flash-attn 2.8.3,
  editable nanotron + astropt3, huggingface_hub 1.24.0, datasets 5.0.0).

## Update (same night, commit 1a09996)

Failure mode #3 is now handled: the retry catch matches the closed-client
`RuntimeError` by message and rebuilds from the last snapshot (test covers
both blip variants; 16/16 loader tests pass). Workers halved to 4/rank to cut
router-DNS pressure. Attempt 2 (368 steps) archived at
`…nopairs.attempt2-368`; attempt 3 relaunched fresh ~00:25, healthy at
~240K tok/s (0.55s/step — 8 workers still saturate; not data-bound as
feared), ETA ~3h. Check `train.log` for "[data] … rebuilding the stream"
lines to see how often the retry path fires.

## Update (2026-07-21 afternoon): the "healthy" numbers above were prefetch

Attempt 3 died at step 3989 on walltime. The ~240K tok/s readings at launch
were never sustained — mean was ~4.6s/step from step 0. gpu5's 1 Gbit NIC is
the hard ceiling. Full diagnosis, fix (workers back to 8), and relaunch
guidance: [streaming throughput audit](2026-07-21-streaming-throughput-audit.md).

## Update (2026-07-21 evening): pairs + scalars run, three bugs, healthy then SIGINT

Moved from the nopairs shakeout to the **full corpus**: new run
`astropt3-70m-jetformer-mmu` (config `astropt3-70m-jetformer-mmu.yaml`,
launch `bash astro/scripts/launch_mmu_gpu5.sh` — the script now takes the run
name as `$1`, defaulting to this run). It adds the crossmatched **pairs**
source (published match index
`hf://datasets/Smith42/mmu_desi_edr_sv3_x_mmu_ssl_legacysurvey_north/match_index.parquet`,
validated: 137,906 pairs across 173 image cells, 0 missing from the current
catalog revision) at the ADR 0006 0.60/0.15/0.25 weights, plus the **ADR 0008
scalar modalities** Z/ebv/photometry. Fresh run dir because the scalar heads
change the model shape — the nopairs checkpoints cannot resume into it.

Launching it surfaced **three streaming bugs**, all fixed and pushed
(branch `adr-0006-stream-mmu-upstream`):

1. **Shard count vs DP world size** (`f2fd5e0`). The pairs source has 165
   train cells after the val split, and `165 % dp 2 != 0` tripped datasets'
   example-stepping fallback: `n_shards` collapsed to 1, the loader clamped
   to one worker. `aligned()` in `streaming.py` truncates each source's
   shuffled train list to a multiple of `num_shards` (≤ dp−1 cells dropped,
   rotating with the epoch shuffle).

2. **Nested datasets streams inside DataLoader workers** (`a3a3860`, the real
   clamp cause — #1 alone did not fix it). `_paired_examples` built inner
   `load_dataset(streaming=True)` readers per partition, and datasets'
   `_iter_pytorch` worker-splits **any** stream iterated inside a worker: each
   inner single-shard reader landed on worker 0, and workers 1..N-1 silently
   read empty streams — no pairs in 7 of 8 workers and an instantly-exhausted
   pairs source (which also drove the earlier restart hot-loop). Inner reads
   now use plain pyarrow row-group iteration (`_pq_rows`), which has no worker
   magic. `open_stream` now logs per-source and combined `n_shards` on every
   build so this class of failure is visible.

3. **96 GiB slurm cgroup OOM** (`a0ed8ce`). The box has 502 GB but the job
   runs under a 96 GiB cgroup (`memory.max` up the hierarchy; `oom_kill`
   incremented). A bulk `to_pylist` of a 56 MB image row group is ~0.5 GB of
   Python objects per worker, and the Python-dict spectra map was ~0.75 GB
   resident per worker — 16 workers blew the cap. `_pq_rows` now converts one
   row per `slice`, and `_paired_examples` holds matched spectra as a filtered
   Arrow table, converting per-row only on a hit. Memory then sat flat at
   ~81 GiB.

After the three fixes the run was **healthy**: reached step 759 at ~0.59s/step
(~110K tok/s/GPU, ~3.7s/step mean including the periodic ~15-25s DNS/link
stalls — link-bound with pairs, as the audit predicted), memory flat ~81 GiB,
all five losses converging (lm_loss ~ −20, images/spectra/Z/ebv/photometry all
trending down), DNS blips ridden out by the rebuild retry.

**It then died at 17:25:28 from an external SIGINT** (torch elastic root
cause: "Signal 2 (SIGINT) received by PID 59987"), NOT a pipeline fault — rank
0 was blocked in the dataloader queue and took the signal there
(KeyboardInterrupt in `queue.get`). The streaming path itself was working. The
SIGINT source is undetermined (manual interrupt or a stray signal to the
process group); the slurm job still had ~21 h of its 24 h walltime.

**State for relaunch:** checkpoint **512** exists with
`dataset_state/dp_{0,1}.pt` saved at the current 8-worker layout, so
`bash astro/scripts/launch_mmu_gpu5.sh` resumes cleanly from `latest.txt`
(512). Resume must keep `num_loading_workers: 8` (the loader refuses a worker
count mismatch). Only ~760 steps in, so a fresh restart also costs little and
keeps the 20k curve contiguous — either is fine. GPUs are free.

## Update (2026-07-21 late): 6-worker rerun reveals a slow OOM leak in the stream

Relaunched fresh at `num_loading_workers: 6` (commit `f6e5be6`) after the
8-worker run saturated the 1 Gbit NIC hard enough to block interactive SSH to
gpu5. The previous run dir is archived at
`astropt3-70m-jetformer-mmu.sigint-759`. Six workers fixed the SSH problem
(job memory started at ~62.5 GiB vs ~81 at eight) and trained cleanly to
**step ~2121**, then a DataLoader worker was **OOM-killed** (`signal: Killed`;
cgroup `oom_kill` 5 → 8). `max-restarts=0`, so the whole run tore down.
Checkpoints 1…2000 saved (`latest.txt` = 2000).

**This is a slow memory leak, not a worker-count problem.** Memory started at
62.5 GiB and climbed to the 96 GiB cgroup cap over ~2100 steps. Fewer workers
only lengthened the fuse (the 8-worker run never reached this — it died at 759
to the external SIGINT, so no long run had previously exercised the pipeline
past ~760 steps). Strong correlation with the **DNS-rebuild path**: 14
"rebuilding the stream" events this run, accelerating toward the end (steps
16, 49, 616, then a cluster at 1436/1494/1567/1873/1940/1948/1972/2115), OOM
right after 2115. Hypothesis: each DNS blip triggers `open_stream` →
`_mmu_records`' `open_records(prev_state)`, building a fresh `datasets`
streaming pipeline (new interleave, new `HfFileSystem`, new prefetch threads
in `RandomlyCyclingMultiSourcesExamplesIterable._iter_arrow`) while the
abandoned one's background threads/buffers are never torn down — so RSS climbs
per rebuild, which slows I/O, which makes DNS blip more, which rebuilds more.
gpu5's flaky single-router DNS is the accelerant; a stable-DNS training
cluster would rebuild rarely and may not hit this — so the leak may be
gpu5-specific in practice, but it is a real unbounded-growth path either way.

**NOT relaunched** — a resume from 2000 just re-OOMs ~2000 steps later. Needs
one of: (a) confirm + fix the rebuild-path leak (explicitly tear down the old
stream before rebuilding — datasets has no clean `close()` for an
`IterableDataset`'s prefetch threads, so this needs care; a `gc.collect()`
alone won't join threads), (b) cut rebuild frequency at the source (reduce
gpu5 DNS blips — userspace resolver cache, or absorb blips inside
huggingface_hub's own retry so they don't escalate to a full stream rebuild),
or (c) accept it as gpu5-specific and move the real run to the training
cluster. Confirm the leak is rebuild-driven vs a steady per-step leak before
picking: an offline RSS-growth probe over repeated `open_stream` builds is the
cheap check. GPUs are free; run dir `astropt3-70m-jetformer-mmu` holds ckpt
2000 (6-worker layout — resume needs `num_loading_workers: 6`).

### Resolution (2026-07-22): fixed — `gc.collect()` on the rebuild path

Took option (a). The offline probe was built (drives the real `datasets`
machinery over local parquet via `tests/fake_mmu.fake_open_stream`, CPU-only)
and confirms the leak is rebuild-driven, 3 arms, growth over 40 rebuilds:
**abandon +79.5 MiB / gc.collect() +1.9 MiB / close()+gc.collect() +1.8 MiB.**
So `gc.collect()` alone bounds it — contra the note above, it need not *join*
the abandoned threads, only reclaim their buffers, which the collector does.
An explicit `records.close()` adds nothing here: on this path the stream
generator has already died by exception (the `client has been closed`
`RuntimeError`), so `.close()` is a no-op.

Fix (`nanotron_loader.py` rebuild block): drop the last live reference
(`self._stream = None`) and `gc.collect()` before reopening. Guarded by
`tests/test_nanotron_loader.py::test_transient_error_rebuilds_and_reclaims`
(first test to exercise the rebuild path: asserts recovery + reclaim). Resume
from ckpt 2000 (`num_loading_workers: 6`) is unblocked. The leak was a real
unbounded-growth path, so this helps any run, not just gpu5's flaky DNS —
though a stable-DNS cluster rebuilds rarely and would rarely trigger it.

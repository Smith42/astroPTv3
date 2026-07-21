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

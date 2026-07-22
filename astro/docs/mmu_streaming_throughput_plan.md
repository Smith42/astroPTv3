# MMU Streaming Throughput Implementation Plan

Implement [ADR 0012](adr/0012-gate-mmu-streaming-throughput.md) for the
current image, spectrum, and scalar pipeline. Do not build generalized
multiway modality plumbing in this milestone.

Every optimization is sequentially gated. Only winners advance. No GPU or
training run is performed on this machine.

## Gate 0 — Freeze and instrument the baseline

**Files**

- NEW `scripts/benchmark_streaming.py`
- `src/astropt3/data/streaming.py`
- `src/astropt3/data/nanotron_loader.py`
- `tests/test_streaming.py`
- `tests/test_nanotron_loader.py`

**Work**

- Fix benchmark inputs: catalog revisions, partition manifest, match index,
  split, source policy, object/token horizon, worker count, and cache state.
- Report:
  - non-padding tokens/sec and bundles/sec;
  - distinct bundle count, multimodal fraction `P`, standalone row coverage,
    and repeated emissions;
  - wall time by read, join/filter, decode/transform, and packing stage;
  - process CPU/RSS and retry count;
  - network bytes when attributable, otherwise isolated-host NIC deltas;
  - configured maximum resume drift in bytes, rows, and bundles.
- Keep instrumentation off the hot path unless benchmarking is enabled.
- Measure instrumentation on/off once.

**Verify**

- Offline fixtures produce stable bundle/source counts.
- Network benchmarking is explicitly opt-in or `network`-marked.
- The current CPU suite remains green.

## Gate 1 — Reopen source assembly and bucket leakage

Benchmark source assembly before deeper transport changes. These are
benchmark variants, not permanent configuration flags unless one wins.

### Variant A — Current partition-local leaky bucket

Current ADR 0011 behavior:

- each pair adds `images_per_pair` credit;
- each emitted unmatched image spends one credit;
- credit starts at zero and is discarded at each partition boundary.

This remains the baseline.

### Variant B — Carry credit across partitions

Carry fractional or unspent credit across successive partitions within one
worker. This tests whether partition-boundary leakage materially lowers the
realized skim ratio.

Resume may discard at most the current prefetch window plus the scalar bucket
balance. If the balance is not checkpointed, quantify the resulting drift.

### Variant C — Pre-funded partition quota

The match index already gives the number of pairs in a partition. Initialize
that partition's image quota from:

`floor(images_per_pair × matched_rows)`

Select unmatched rows using a deterministic row-id hash or evenly spaced
quota, rather than waiting for preceding pairs to fund the bucket. This tests
whether removing row-order bias improves coverage without buffering rows.

Do not retain this variant if hashing or quota machinery adds complexity
without a source-mix or coverage benefit.

### Variant D — Simple three-streamer model

Reopen the original independent:

1. images-only stream;
2. spectra-only stream;
3. pairs stream.

Use the existing HF interleave machinery and current weights as one trial.
This is expected to transfer more image bytes than the skim assembly, but it
is operationally simpler and provides broader standalone coverage.

Do not restore a production `three_streamers` flag merely for comparison.
Keep the implementation only if it lands on the measured
throughput–`P`–coverage frontier.

### Required comparisons

Compare all variants at:

- the same emitted token horizon;
- the same requested `P` where feasible;
- identical catalog revision, cells, worker topology, and cache state;
- at least low, median, and high match-density partition strata.

Report:

- realized image/spectrum/pair mix;
- distinct multimodal bundles;
- distinct standalone rows;
- repeated bundle count;
- bytes per non-padding token;
- discarded unmatched rows;
- unused bucket credit at partition boundaries;
- selection by row position within each partition.

### New offline tests

Add tests covering:

1. current bucket credit never goes negative;
2. emitted images never exceed funded or quota images;
3. explicit accounting of credit lost at partition boundaries;
4. the same partition composition in different row orders exposes current
   order sensitivity;
5. carry-credit mode realizes the target ratio over uneven partitions;
6. pre-funded quota is deterministic and insensitive to pair position;
7. no variant emits a matched image as an image-only discard;
8. source exhaustion or repetition cannot silently inflate distinct-bundle
   `P`;
9. three-streamer and skim variants decode equivalent paired rows;
10. DP ranks and DataLoader workers remain disjoint for every variant.

## Gate 2 — Establish the native-I/O ceiling

Run this gate separately for at most the two winning source assemblies. Do not
add a custom prefetcher.

Benchmark:

- loader workers and read concurrency;
- Arrow batch size;
- `ParquetFragmentScanOptions` cache, prefetch, and range-size settings;
- HfFileSystem or fsspec block size;
- Arrow versus filesystem readahead, ensuring only one layer owns it.

Use a small grid per controlled link profile, then confirm finalists on the
real link. Keep benchmark-only values out of production configuration;
promote only the winning minimal set.

**Gate:** advance only if the real link is better utilized or end-to-end data
wait falls. Added production complexity still requires the final 10%
end-to-end gate.

## Gate 3 — Verify physical byte avoidance

### 3A. Column projection

Pass only fields consumed by `decode_record` and the join. Test nested leaf
projection against actual MMU parquet metadata and measured transferred bytes.

Expected result: a small image-side gain because `image.flux` dominates.
Delete production projection plumbing if it saves no meaningful bytes and
complicates union features.

### 3B. Row-group location metadata

Evaluate extending the lightweight match index with catalog revision plus
row-group and row locations.

- Image-side skipping is unlikely: ADR 0011 found matches in essentially every
  relevant image row group.
- Spectrum-side skipping is retained only if matched rows occupy sufficiently
  few row groups to reduce physical reads.
- Never claim row-level filtering as a network saving when the containing
  column chunk must still be fetched.

**Verify**

- Joined bundle identities match the ID-based implementation.
- A stale catalog revision fails clearly.
- Missing or new rows follow the existing safe skip/error policy.
- Keep only measured physical-byte savings.

## Gate 4 — Measure topology and file ownership

Test:

- one host with a shared link;
- multiple hosts sharing an upstream cap;
- hosts with independent links.

Keep DP partition assignment disjoint and leave DataLoader worker splitting to
HF Datasets. Do not reintroduce manual rank×worker double sharding.

Expose only:

- per-host read concurrency;
- an optional job-wide cap supplied by the launcher or operator;
- file ownership where it prevents measured duplicate reads.

**Verify**

- Every worker receives shards.
- No duplicate partition ownership occurs unless required by pair upweighting.
- Source proportions and `P` stay within benchmark tolerance.
- Shared-link tests do not exceed the configured aggregate cap.

## Gate 5 — Test bounded SSD caching

This gate reopens only the bounded-cache portion of ADR 0011's rejected local
materialization option.

**Work**

- First measure which matched files or ranges are reread under
  `all_exhausted`.
- Use installed fsspec or HF cache facilities before custom cache code.
- Cache only hot immutable files or ranges under an explicit byte budget.
- Key entries by resolved catalog revision.
- Support `cache_bytes=0`.
- Do not add cross-process locking until duplicate downloads are observed.

Run the cache trial against both the current skim assembly and the
three-streamer variant. Their uncached ranking may reverse when repeated reads
are cached.

**Compare**

- cold cache;
- warm cache;
- cache disabled;
- budgets covering none, part, or all of the repeatedly scanned hot set.

Report unique remote bytes, duplicate remote bytes, cache-served bytes,
eviction/re-fetch bytes, and the hot-set size needed to eliminate 50%, 90%,
and 100% of measured rereads.

**Verify**

- Eviction never affects correctness.
- Cache size never exceeds its budget.
- Stale revisions are not reused.
- Interrupted writes are ignored or recovered.
- Warm results are never reported as cold Internet throughput.

## Gate 6 — Bound Arrow conversion cost

Run only if earlier gates expose CPU or conversion as material.

**Files**

- `src/astropt3/data/streaming.py`
- `tests/test_streaming.py`

**Work**

Compare the current one-row conversion with small bounded Arrow batches. Do
not restore whole-row-group `to_pylist`: it previously caused cgroup OOM.
Vectorize filtering and join work while retaining Arrow-native storage as long
as possible.

**Verify**

- Decoded records and bundles are identical.
- Peak RSS remains bounded at the production worker count.
- Retry/rebuild memory reclamation does not regress.
- Nontrivial batching code reaches the 10% end-to-end acceptance threshold.

## Gate 7 — Source-set frontier

For each viable source set, report:

- tokens/sec;
- distinct multimodal bundle fraction `P`;
- distinct standalone rows;
- bytes and repeated reads.

Include:

- current skim plus spectra-only policy;
- carried-credit and pre-funded skim winners, if any;
- the simple three-streamer model;
- reduced or removed standalone sources when crossmatched coverage dominates;
- alternative cost-aware anchors where available.

Do not choose `P` in code. Operators select a non-dominated point and record it
in run metadata.

## Gate 8 — Custom asynchronous prefetch, only if still needed

Proceed only when:

- the accelerator still waits for data;
- the link remains below its usable ceiling;
- native Arrow and HF controls cannot close the gap.

Use one finite byte-bounded queue per worker. Checkpoints need not serialize
payloads; resume drift is at most one queue window. Avoid cross-process shared
scans unless a prototype proves sharding, teardown, retries, and memory remain
bounded.

Delete the custom queue if it misses the 10% end-to-end threshold.

## Autoresearch-style experiment loop

Use the useful constraints from Karpathy's autoresearch—fixed evaluation,
fixed time, one hypothesis, keep/discard logging—without its unrestricted
"loop forever" behavior on a shared Internet connection.

### Frozen harness

The benchmark harness, partition manifest, metric calculation, safety limits,
and acceptance rules are read-only during a campaign. Experiments may modify
only declared loader/source files and configuration.

Do not add dependencies. Test one hypothesis per experiment.

### Fixed experiment budget

Each loader-only experiment runs for a fixed wall-clock measurement window
after startup and warmup. Initial campaign settings:

- 10-minute measurement window;
- three repeats per candidate;
- operator-supplied total experiment count `N`;
- fixed controlled link profile and topology;
- explicit aggregate bandwidth and concurrency cap.

Short smoke runs may reject crashes, but cannot promote a result.

### Results ledger

Record one TSV row per candidate:

`commit  variant  median_tokens_s  p95_wait_ms  bytes_per_token  P  distinct_rows  peak_rss_gb  cache_gb  retries  status  description`

Statuses are `keep`, `discard`, or `crash`.

Keep a candidate only when it:

- improves median loader throughput enough to justify an end-to-end trial;
- satisfies `P`, coverage, memory, resume, and stability constraints;
- remains simpler than an equal-performing alternative.

The final production gate remains at least 10% higher real-host end-to-end
training tokens/sec.

### Drift control

Network variance can overwhelm small changes. Therefore:

- run baseline and candidate in alternating AB/BA order;
- rerun the untouched baseline every five experiments;
- randomize only experiment order, never the partition manifest;
- report median and spread across repeats;
- invalidate a comparison when baseline drift exceeds a declared tolerance;
- confirm all finalists on the real training link.

### Safe iteration

Use one temporary worktree and commit per experiment. Keep winning commits and
discard losing worktrees. Do not repeatedly hard-reset the main working tree.

The campaign stops after `N` experiments, the operator's wall-clock or
bandwidth budget, or exhaustion of declared hypotheses. It does not run
indefinitely on shared egress.

## Additional required trials

### Long soak and failure injection

Short repeated benchmarks do not catch the rebuild leak previously found only
after thousands of steps. Finalists require one long soak with:

- repeated DNS and connect failures;
- closed-client `RuntimeError`;
- truncated or failed range reads;
- repeated stream rebuild and `gc.collect()`;
- cache interruption or partial entries;
- bounded RSS, thread count, and file-descriptor count.

### Tail latency

Report p50, p95, and p99 batch wait and step time, not only means. Historical
periodic stalls were hidden by healthy prefetch bursts.

### Demand regimes

Confirm at least:

- the 70M high-token-rate model, where the NIC is known to bind;
- one representative larger and slower model, where loader work may not
  improve end-to-end throughput.

Do not make an optimized path the default for compute-bound models unless it
is neutral in memory and stability.

### Validation invariants

For every finalist:

- validation remains bit-exact;
- train and validation HEALPix partitions remain disjoint;
- bundle IDs and `P` accounting survive repeats and resume;
- replay or skip remains within one configured prefetch window;
- source distributions remain within declared confidence bounds.

### Cache and CDN controls

Separate process or OS cold start, local SSD cold and warm states, and likely
HF or CDN warming. A local-cache result must never be labeled as cold Internet
throughput.

## Final interaction check

Test only surviving settings across:

- low-bandwidth and high-latency;
- high-bandwidth and moderate-latency;
- the actual training link;
- cache disabled and the approved bounded-cache case.

This is a small finalist matrix, not a factorial search.

## Acceptance

A nontrivial change ships only if it provides:

- at least 10% higher real-host end-to-end non-padding training tokens/sec;
- identical model, sequence length, source policy, and `P`;
- no OOM or unbounded queue/cache growth;
- no increased failure rate;
- no spatial train/validation leakage;
- source-distribution drift within declared tolerance;
- resume replay or skip no greater than one prefetch window per worker.

Projection and essential instrumentation may remain below 10% only when their
maintenance cost is negligible.

## Required repository gates

After each retained change, run its targeted CPU tests. Before promotion, run
from `astro/`:

1. `uv run pytest`
2. `uv run python scripts/count_params.py`
3. `uv run python -m astropt3.train_smoke --config
   configs/model/test-tiny.yaml --steps 50 --assert-decrease`

GPU and end-to-end trials run only on the training system.

## Deferred

- Generalized multiway modality loader.
- Runtime auto-tuning.
- Public dataset republishing or cropping.
- Bulk local mirrors.
- Cross-process cache coordination without duplicate-read evidence.
- Custom async queues unless native controls fail.

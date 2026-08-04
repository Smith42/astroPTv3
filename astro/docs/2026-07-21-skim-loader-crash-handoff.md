# Handoff: ADR 0011 image-only skim crashes on the real loader path

**Date:** 2026-07-21
**Branch:** `adr-0006-stream-mmu-upstream`
**Status:** **RESOLVED, same day (later session).** The skim assembly was
never broken — the crash was a config error, and the leading hypothesis
below (generator-first source ordering) was a red herring.

## Resolution (read this, skip the hypothesis sections)

The skim YAML's `resume_checkpoint_path` was copied from the baseline
config and left pointing at the **baseline's** run dir. The "fresh" skim
run therefore resumed the baseline's checkpoint 2000 — the log shows
`Loading weights from .../astropt3-70m-jetformer-mmu/2000`,
`Resuming astropt3 stream state from .../2000/dataset_state`, and
`start_iteration_step: 2000` (it never crashed "at step 0"; it crashed on
the sanity-check batch of an unintended resume). That loaded the
baseline's 3-source non-skim stream state (map source first) into the
2-source skim stream (generator first). In `datasets`, a
`MappedExamplesIterable` state carries a nested `examples_iterable` key
and a `GeneratorExamplesIterable` state does not, so the per-source state
reload at the top of `CyclingMultiSourcesExamplesIterable.__iter__` dies
with exactly `KeyError: 'examples_iterable'`. On a truly fresh start that
reload line cannot execute (`previous_states` is all-None) — which is why
no fresh repro ever crashed, regardless of source order or shard counts.

Fixes:
- `astropt3-70m-jetformer-mmu-skim.yaml`: `resume_checkpoint_path` now
  points at the skim run's own dir (baseline convention: resume path ==
  checkpoints path).
- `PackedMicroBatches.load_state_dict` now records/checks `skim_images`
  in the stream state, so a cross-assembly state load fails with an
  actionable ValueError instead of a worker KeyError
  (`test_load_rejects_cross_skim_state`).
- Validated offline on this node, CPU-only, `.venv-train`: (red) the
  baseline's real `2000/dataset_state/dp_1.pt` loaded into a skim
  12-worker StatefulDataLoader hits the new guard; (green) a fresh skim
  stream at the crash config (dp_size=2, dp_rank=1, num_workers=12, real
  hub + real match index) yields batches.

Everything below is the original (superseded) analysis, kept for the
record.

---

**Original status:** image-only skim (ADR 0011) is implemented and passes the
CPU suite, but **crashes at step 0 on the real training path** (DP=2 +
StatefulDataLoader, `num_loading_workers: 12`). Feature is opt-in and
off by default, so nothing else is affected. Baseline run recovered and
healthy. **Do not relaunch the skim run until the loader bug is fixed and
validated offline.**

## TL;DR for whoever picks this up

- The skim assembly in `open_stream(skim_images=True)` builds a **2-source
  interleave `[scan(from_generator), spectra(map)]` with the generator
  first**. The working non-skim path is a **3-source `[map, map, generator]`
  with the generator last**.
- Under `split_dataset_by_node` (DP=2) + torchdata `StatefulDataLoader`
  spreading shards across 12 workers, a worker dies with
  `KeyError: 'examples_iterable'` inside datasets'
  `RandomlyCyclingMultiSourcesExamplesIterable.__iter__` → `load_state_dict`
  (saved `previous_states[i]` structure ≠ live iterable structure).
- **Leading fix hypothesis (unvalidated):** reorder the skim parts so the
  `from_generator` source is **last**, matching the layout that works
  (`[spectra(map), scan(gen)]`, weights `[0.15, 0.85]`). May also need a
  deeper restructure if the trigger is the datasets `from_generator` +
  interleave + StatefulDataLoader combo rather than order alone.
- **The CPU tests miss this** — they use plain iteration, never the
  `StatefulDataLoader + DP-split + num_workers=12` path. The bug is not
  reproducible with the tiny 24-record `fake_mmu` fixtures (needs the real
  unequal shard counts, 164 vs 298). **A fix MUST be validated against a
  realistic sharding config, GPU-free (see below), before any relaunch.**

## Context — what the skim is

ADR 0011 (`astro/docs/adr/0011-skim-crossmatch-scans.md`, amended same day to
"coarse determinism"). The crossmatch scan already downloads every image row
group and discards the unmatched rows. `skim_images=True` turns that scan into
a demux: one pass yields both **pairs** (matched) and **image-only** records
(skimmed from the discards), so the standalone images-catalog download is
dropped. Images-only skim; spectra stay single-sourced. Expected win at a
saturated 1 GbE link: ~1.2–1.6×, scaling with match density d.

The goal of the run that crashed: an **A/B on streaming speed** — skim-on vs
the current skim-off `astropt3-70m-jetformer-mmu` run. The baseline's
streaming cost shows up as periodic multi-second loader stalls; measured
baseline: **322 / 2641 iterations ≥ 5 s (12.2%)** at step 2641. That stall
rate / RX bandwidth is the metric to beat — NOT steady-state tokens/sec
(~320 ms/iter, compute-bound).

## What was done (timeline, 2026-07-21)

1. Implemented the skim (committed): see commits below.
2. Prepped the A/B: added the fork config field + a `-skim` run config
   (still **uncommitted**, see "Working-copy state").
3. Halted the baseline cleanly (SIGINT to torchrun master) at step 2641.
   Baseline had **no exit checkpoint** — last checkpoint is step 2000.
4. Launched skim (`bash astro/scripts/launch_mmu_deltaai.sh
   astropt3-70m-jetformer-mmu-skim`) → **crashed at step 0** (below).
5. Recovered: relaunched the baseline (`... astropt3-70m-jetformer-mmu`),
   which auto-resumes from checkpoint 2000 (`resume_checkpoint_path` +
   `latest.txt=2000`). Confirmed healthy at step 2127, GPU 100%.

## The crash

Log: `/work/nvme/bfvh/msmith10/astroPTv3_checkpoints/astropt3-70m-jetformer-mmu-skim/train.log`

Config parsed fine (`skim_images=True` reached the dataset args) and the
loader printed:

```
[data] open_stream split=train epoch=0 shard=1/2 source n_shards=[164, 298] -> stream n_shards=82
```

Then the first `next(dataloader)` (nanotron `sanity_check_dataloader`) died in
DataLoader **worker 8 of 12**, DP rank 1:

```
File ".../datasets/iterable_dataset.py", line 877, in __iter__
  self.ex_iterables[i].load_state_dict(self._state_dict["previous_states"][i])
File ".../datasets/iterable_dataset.py", line 264, in _inner_load_state_dict
  state[key] = _inner_load_state_dict(state[key], new_state[key])
KeyError: 'examples_iterable'
```

`self` is `RandomlyCyclingMultiSourcesExamplesIterable` (the
`interleave_datasets` core). It is reloading each source's saved
`previous_states[i]` into the live `ex_iterables[i]`; the live iterable has a
nested `examples_iterable` key that the saved state lacks → structural
mismatch. Both `MappedExamplesIterable` and `GeneratorExamplesIterable` carry
`examples_iterable` in their `_init_state_dict`, so the mismatch is about the
**sharded/wrapped** structure under `_iter_pytorch` + `split_dataset_by_node`,
not the raw source type.

**Key evidence the bug is the skim ASSEMBLY, not the code change generally:**
the non-skim path runs my new `streaming.py` (astropt3 is an editable install
in `.venv-train`) and resumed + trains fine. The only structural differences
skim introduces: (a) generator source at index 0 instead of last, (b) 2
sources instead of 3, (c) unequal source shard counts (164 scan vs 298
spectra) into the DP-split + worker split.

Relevant code (all in `astro/src/astropt3/data/streaming.py`):
- `open_stream(..., skim_images=False)` — the `if match_index is not None and
  skim_images:` branch builds `parts = [_pairs_dataset(...images_per_pair),
  spectra_part]`, `weights = [0.85, 0.15]`. **Generator is parts[0].** This is
  the suspect ordering.
- `_paired_examples(..., images_per_pair)` — the demux generator + budget
  governor (works; the governor itself is fine, tested on CPU).
- `interleaved(parts, weights, seed, shard, num_shards)` — calls
  `interleave_datasets(...)` then `split_dataset_by_node` when `num_shards>1`.

## Repro attempts (so the next model doesn't repeat them)

Repro script (was at `$CLAUDE_JOB_DIR/tmp/repro_skim_loader.py`, reproduced
here — run in `.venv-train`, CPU-only, no GPU needed; it routes the real
loader at the local-parquet fake stream):

```python
import sys, types
sys.path.insert(0, "astro/src")
sys.path.insert(0, "astro/tests")

import astropt3.data.streaming as streaming
from fake_mmu import fake_open_stream, _fixtures
from astropt3.config_io import load_model_config
from astropt3.data.nanotron_loader import build_astropt3_dataloader

streaming.open_stream = fake_open_stream   # local-parquet skim stream
_fixtures()                                # build fixtures before workers fork

model_config, _ = load_model_config("astro/configs/model/test-tiny.yaml")
dataset_args = types.SimpleNamespace(
    data_root="mmu", match_index="present", skim_images=True,
    object_id_log=None,
    synthetic_image_only_fraction=0.3, synthetic_spectrum_only_fraction=0.0,
)
loader = build_astropt3_dataloader(
    dataset_args, model_config, micro_batch_size=2, sequence_length=896,
    dp_rank=1, dp_size=2, num_workers=12, seed=0,   # <-- match the real run
)
it = iter(loader)
for k in range(5):
    b = next(it); print("batch", k, "ok", tuple(b["input_ids"].shape))
print("NO CRASH")
```

Env to run it:
```bash
source /opt/cray/pe/lmod/lmod/init/bash
module use /sw/user/modules/python
module load python/miniforge3_pytorch/2.12.0
source .venv-train/bin/activate
CUDA_VISIBLE_DEVICES="" python repro_skim_loader.py
```

Results:
- `dp_size=1, num_workers=2` → **no crash** (no `split_dataset_by_node`).
- `dp_size=2, num_workers=2` → **no crash**.
- `dp_size=2, num_workers=12` → **hung** (2-min timeout), did not crash — the
  24-record fixtures can't sustain 12 workers under `all_exhausted`.

Conclusion: the **fixtures are too small** to reproduce. To reproduce you need
a source layout with **many, unequal shard counts** like the real run
(scan 164 cells, spectra 298 files), split across DP=2 then 12 workers.

## Recommended next steps (all GPU-free — can run alongside the baseline)

1. **Reproduce faithfully.** Either enlarge the `fake_mmu` fixtures
   (`_FILES_PER_SOURCE`, add records) so scan and spectra have many, *unequal*
   shard counts, OR run a **data-only dry run over the real hub stream** on a
   login node: build `build_astropt3_dataloader(..., dp_size=2, dp_rank=1,
   num_workers=12)` with `data_root="mmu"`, real `match_index`,
   `skim_images=True`, and pull a few batches. No GPU, no training — the crash
   is purely in data loading.
2. **Apply the fix and confirm it flips red→green** in that repro:
   - First try: reorder the skim parts so the generator is **last** —
     `parts = [spectra_part, _pairs_dataset(...images_per_pair)]`,
     `weights = [DEFAULT_WEIGHTS[1], DEFAULT_WEIGHTS[0]+DEFAULT_WEIGHTS[2]]`
     in `open_stream` (and mirror in `fake_mmu.fake_open_stream`). Order does
     not change the mix; interleave weights are per-source.
   - If reorder alone doesn't fix it, the trigger is the datasets
     `from_generator` + interleave + StatefulDataLoader combination under
     worker sharding. Options: wrap the generator so its state_dict matches a
     map source; or check the installed `datasets` version for a known
     interleave/`from_generator` state_dict bug and pin/patch; or avoid
     interleaving the generator (fold spectra into the same generator — but
     that reintroduces the coupled-source risk ADR 0011 warns about).
3. **Add the missing regression test:** a CPU test that builds
   `build_astropt3_dataloader(..., num_workers>0)` (torchdata
   StatefulDataLoader) over the skim stream and pulls a few batches. This is
   the coverage gap that let the bug ship. Guard with `skipif` if torchdata
   isn't in the `astro` uv venv.
4. **Only then** relaunch the A/B (see below).

## Launching the A/B once fixed

```bash
# from repo root, on a DeltaAI GH200 node with 2 free GPUs:
bash astro/scripts/launch_mmu_deltaai.sh astropt3-70m-jetformer-mmu-skim
```
Matches the baseline (12 loaders, DP=2, seq 4096, mbs 16, 20000 steps,
seed 42) except `skim_images: true`. Halt the baseline first (SIGINT the
torchrun master; it resumes from its latest checkpoint on relaunch).
**Compare:** fraction of iterations with `time_per_iteration_ms >= 5000`
(baseline = 12.2%) and RX bandwidth — not steady-state tokens/sec.

## Commit / working-copy state

Committed on `adr-0006-stream-mmu-upstream`:
- `e2653e8` docs(adr-0011): relax skimmed train stream to coarse determinism
- `e000647` feat(data): image-only skim from the crossmatch scan (ADR 0011)
  — astro-side: `streaming.py`, `nanotron_loader.py`, `fake_mmu.py`,
  `test_streaming.py`. **This is the code that crashes on the real path.**

Uncommitted in the working copy (prep for the A/B; hold until the fix lands):
- `nanotron/src/nanotron/config/astropt3_config.py` — added
  `skim_images: bool = False` to `AstroPT3StreamingDatasetsArgs` (submodule;
  needed for the YAML to set it).
- `astro/configs/nanotron/astropt3-70m-jetformer-mmu-skim.yaml` — base config
  + `skim_images: true`, run/checkpoint/object_id_log renamed to `-skim`.

## Live-system state / cautions

- **Baseline `astropt3-70m-jetformer-mmu` is RUNNING** (resumed from step
  2000; was at step 2127 when this was written). Do not kill it without
  intending to; it holds both GH200s on node gh034.
- The bg-isolation guard was disabled for this repo to allow direct edits:
  `"worktree": {"bgIsolation": "none"}` in `.claude/settings.local.json`
  (gitignored). Remove it to restore the guard.
- `astropt3` is an **editable install** in `.venv-train`, so working-copy
  edits to `astro/src/astropt3/**` take effect on the next run launch (this is
  why the crash picked up the new skim code, and why the baseline runs the new
  non-skim code safely).

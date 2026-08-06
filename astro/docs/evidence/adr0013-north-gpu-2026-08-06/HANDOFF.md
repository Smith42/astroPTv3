# ADR 0013 North three-spoke GPU handoff — 2026-08-06

First end-to-end GPU runs of the ADR 0013 source graph: lsdb crossmatch ->
pointer index -> streaming -> packing -> nanotron with 11 modalities. The
pipeline works and learns. The corpus is **transfer-bound**, and that is the
finding that should shape the next session.

## Recovery point

Branch `mmu-corpus-expansion`, six commits, **none pushed**:

```
11b1652 docs(astro): bytes-per-megabyte study, spoke build script, run configs
2d13652 feat(astro): three-spoke GPU check configs; one source-graph predicate
9a2c920 feat(data): one row per anchor object in the merged index
771776a refactor(data): every spoke is an lsdb positional crossmatch
7d9e148 perf(data): let lsdb and dask do the spoke-index build
c4401cc refactor(data): build spoke indexes with lsdb's default crossmatch
```

Submodule `nanotron` branch `astropt3`: `703a3fe5` (family-loss output keys),
**not pushed**. The repo commit `2d13652` bumps the submodule pointer to it.

`uv run pytest`: **147 passed, 1 skipped, 12 deselected**. Ruff clean.

Two stashes, to restore **together** if ever revisited:
- `stash@{0}` order-equivalence test for the prefetch
- `stash@{1}` cell prefetch + fsspec/pyarrow readahead

They are correct but measured no better than baseline (see Findings), so they
were parked rather than committed.

## Decisions taken this session

1. **Every spoke is an LSDB positional crossmatch** (owner). The PROVABGS
   lineage join is gone, along with `_build_lineage`, `_via_edges`, the
   `--join-kind`/`--via-*` flags and the five `via_*` schema columns.
   Consequence: 1,075 anchors now carry PROVABGS targets with no DESI
   spectrum on the record — impossible under the lineage join.
2. **Merged index is one row per anchor** (schema v3), a column block per
   spoke, null where absent. `load_source_graph` sniffs the layout and returns
   an identical `MatchGraph` from either.
3. **Three byte-efficiency directions selected**: per-band tokens, AR
   span-order replay, block-shuffled cell ordering. See the study.

## Corpus state on disk

```
astroPTv3_index/north-v2/           desi.parquet hsc.parquet provabgs.parquet
astroPTv3_index/north-v2-merged/    match_index.parquet   (149,491 anchors)
astroPTv3_index/build-logs/         *.build.log  (kept OUT of the index dir)
```

| spoke | matches | wall | many-to-one | notes |
| --- | ---: | ---: | ---: | --- |
| desi | 137,906 | 571 s | 39 | **reproduces the published ADR 0011 index exactly** |
| provabgs | 111,798 | 650 s | 44 | 1,075 anchors have PROVABGS but no DESI |
| hsc | 15,777 | 611 s | 7 | only 25 of 181 cells |

Cumulative: 181 cells, 149,491 anchors, 265,481 edges; spokes per anchor
`{1: 37502, 2: 107988, 3: 4001}`.

**Not built: SDSS and galaxies-with-hats.** Run
`bash astro/scripts/build_remaining_spokes.sh` (parallel, ~2.4 h and ~4.3 h at
8 workers), then it regenerates the merged index into `north-v2-merged-5spoke`.

## What ran on the GPU

| run | wandb | reached | note |
| --- | --- | ---: | --- |
| 1 worker | `nlzadnmt` | 14/100 | 24 s/step baseline |
| 8 workers | `m4vqjc7p` | 100/100 | only clean exit |
| 1k baseline | `v8vhktzo` | 538/1000 | killed |
| 1k prefetch | `a9mvs5cw` | 142/1000 | killed |
| 1k readahead | `08utrqeh` | 320/1000 | killed |
| long | `vm7029t2` | **3035/20000** | 12.5 h, 15.5 s/step, stopped on request |

Checkpoints for the long run at 1,2,4,...,512,1000,2000,3000; `latest.txt` =
3000. Its stream state was saved at `num_loading_workers: 8` and **can only
resume at that same count**; weights load regardless.

Loss did descend post-warmup (medians over 400-step windows):
`133.5 -> 33.5 -> 57.5 -> 3.5 -> -26.6 -> -13.4`. Negative is correct for the
jetformer exact-likelihood objective. Single steps swing by orders of
magnitude; only windowed medians are readable.

`grad_norm` sat at 240-400 against `clip_grad: 100` throughout — every step
clipped 2-4x. Worth understanding before a real pretraining run.

## Findings — see `docs/2026-08-06-useful-bytes-per-megabyte.md`

The corpus is transfer-bound: ~0.63 s median step against ~15 s wall, so
**~96% of wall clock is waiting for bytes**. Headlines:

- `spectrum.ivar` is **41%** of every DESI spectrum byte and the model never
  reads it. Projecting the used leaves cut a real partition read
  **60.7 -> 36.4 MB (40%)**. Lowest-risk win available.
- `image.flux` is **92.3%** of an anchor row group; the 96x96 crop discards
  **60%** of it, and the discarded periphery is **not** empty sky (median
  patch std 0.6246 crop vs 0.6189 full frame).
- **39% of partner-partition fetches are re-downloads** because the epoch cell
  shuffle destroys HATS spatial locality. Block-shuffling 8-16 contiguous
  cells with a small LRU recovers 83-91%.
- Unmatched rows are already **36% of trained tokens at zero marginal bytes**
  — keep them. But HSC swung **0.1% -> 16.4%** of objects between runs;
  composition variance is the bigger risk.
- **Row-group and page skipping are impossible**: one row group per DESI
  partition (up to 520 MB), no page index written.

### Negative results — recorded so they are not repeated

8 loader workers, a cell-boundary prefetch thread, and fsspec
`cache_type=background` + pyarrow `pre_buffer` **all measured within noise of
baseline** over an identical window (steps 11-142, startup excluded):
2.88 / 3.20 / 3.73 s per step, with the same 21-22 slow steps in each.

Earlier claims of "12s -> 6s -> 3s" in this session were a **sampling
artifact** of comparing runs of 538, 142 and 65 steps. Concurrency does not
create bandwidth; with 8 workers each owning different cells, a boundary needs
~8 x 1.18 GB concurrently.

## Traps found the hard way

- `source_assembly_for_index` and `open_stream` each independently re-derived
  "is this a source graph"; adding schema v3 to one sent a three-spoke index
  down the DESI-only path. Now one `is_source_graph()`. A third copy in
  `_crossmatch_dataset` would have streamed unpinned revisions.
- `PipelineBlock` asserts an **exact** output-key set; `AstroPT3Loss` emitted
  `{family}_family_loss` only for families present in that batch, so the set
  changed batch to batch. All three families now always emit.
- A build log inside the index directory failed the whole `pq.read_table`.
  `load_source_graph` now globs `*.parquet`; keep logs out anyway.
- `matched.partitions[...]` on an lsdb crossmatch builds a graph with missing
  dependencies; narrow the **anchor** with `pixel_search` instead.
- `to_dask_dataframe()` before `map_partitions` breaks on divisions, and
  `to_parquet` cannot tokenize its `ResetIndex`. Use lsdb's own
  `Catalog.map_partitions(include_pixel=True)`.
- Anchor cells in HEALPix order need not overlap a small-footprint spoke;
  pick smoke cells from the **alignment**, not the catalogue order.
- `nanotron` logs `iteration: N / M` **with spaces**. Grep patterns without
  them match nothing and look like a stalled run.

## Suggested next steps

1. Push the six repo commits and the fork commit.
2. Build SDSS + galaxies-with-hats (`build_remaining_spokes.sh`), re-merge.
3. Ship the `spectrum.ivar` projection — 40% measured, no modelling change.
4. Prototype the three selected directions, each with its own A/B.
5. Derive a 5-spoke config from
   `astropt3-70m-jetformer-source-graph-check.yaml` (47 modalities, vocab
   145). The three-spoke configs drop SDSS/galaxies and use vocab 64.
6. Note ADR 0013 remains **Proposed**; the per-spoke GPU pilots and owner
   learning-evidence disposition in the test plan are still open.

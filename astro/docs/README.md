# AstroPTv3 documentation

| Read this | For |
| --- | --- |
| [`../README.md`](../README.md) | Project overview and quick start |
| [`architecture.md`](architecture.md) | Model, data, and implementation architecture |
| [`training.md`](training.md) | Environments, launches, checkpoints, and verification |
| [`../EXPERIMENTS.md`](../EXPERIMENTS.md) | The single source for project decisions, phase history, and experiment results |
| [`adr/`](adr/) | Architecture decision records, including superseded decisions |

## Data-path status

The active training corpus is defined by the mandatory match index and its
crossmatch-only assembly; see [ADR 0011](adr/0011-skim-crossmatch-scans.md)
and [`nanotron_loader.py`](../src/astropt3/data/nanotron_loader.py). [ADR 0015](adr/0015-lsdb-infinite-stream-training.md)
is an experimental, incomplete alternative—not the current training path.

Historical MMU scouting evidence is retained in
[`evidence/adr0013-anchor-scout-2026-08-04/`](evidence/adr0013-anchor-scout-2026-08-04/)
and
[`evidence/adr0013-source-spokes-2026-08-05/`](evidence/adr0013-source-spokes-2026-08-05/).

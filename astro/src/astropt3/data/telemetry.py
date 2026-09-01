"""ADR 0014 §3 instrumentation: wire bytes in, loss-bearing tokens out.

Two halves, because the numbers live in different processes:

- **Byte probe** — patches ``HfFileSystemFile._fetch_range``, the only
  accepted instrument (§11 refuses HTTP-request counting, which counts
  *ranges*, and contiguous-log-run counting, which cannot separate 16
  interleaved workers). It runs inside each loader worker, so each worker
  writes its own ``bytes.dp{rank}.w{worker}.jsonl``: one line per fetch with
  the fields §3 asks for — rank, worker, source, path, byte range, payload
  bytes, fetch wait.
- **Step counters** — accumulated in the MAIN process by :class:`TelemetryLoader`,
  because worker-process counters have no way back to the trainer. Everything
  §3 wants per step is recoverable from the flat micro-batch itself
  (non-padding tokens, loss-bearing tokens and target values per modality,
  packing utilisation) plus the measured wait on ``next()``.

The two halves are joined offline by ``scripts/bench_report.py`` on
(rank, wall-clock window). Its optional ``--object-log`` join (distinct base
objects and replica counts via ``#n`` suffixes) needs a per-object log the
loader no longer writes (ADR 0015 dropped ``object_id_log`` with the rest of
the dataset-checkpoint machinery); byte and step-counter telemetry are
unaffected.

Everything here is a no-op unless ``$ASTROPT3_TELEMETRY_DIR`` is set, so
production runs pay nothing and the benchmark opts in.
"""

from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..tokenization import PAD_ID

TELEMETRY_DIR_ENV = "ASTROPT3_TELEMETRY_DIR"

# ADR 0015: the legacy anchor catalog is the only surviving source
_CATALOG_SOURCES = {
    "legacy": "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north",
}
_REPO_TO_SOURCE = {
    catalog.split("hf://datasets/")[-1].split("@")[0]: source
    for source, catalog in _CATALOG_SOURCES.items()
}


@lru_cache(maxsize=4)
def _resolved_dir(raw: str) -> Path:
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def telemetry_dir() -> Path | None:
    """The configured output directory, or None when instrumentation is off.

    Cached: the trainer calls this once per step, and an mkdir per step on
    BeeGFS is a needless round trip in the logging path.
    """
    raw = os.environ.get(TELEMETRY_DIR_ENV)
    return _resolved_dir(raw) if raw else None


def source_of_path(path: str) -> str:
    """Attribute a hub path to a catalog source, for per-source byte totals."""
    for repo, source in _REPO_TO_SOURCE.items():
        if repo in path:
            return source
    return "unknown"


# -- byte probe (runs inside each loader worker) ------------------------------

_probe_installed = False


def install_byte_probe(rank: int = 0, worker: int = 0) -> bool:
    """Patch ``_fetch_range`` to log every payload this process pulls.

    Idempotent per process (each DataLoader worker is its own process and
    calls this once from :meth:`PackedMicroBatches.__iter__`). Returns whether
    the probe is active.
    """
    global _probe_installed
    if _probe_installed:
        return True
    directory = telemetry_dir()
    if directory is None:
        return False
    try:
        from huggingface_hub.hf_file_system import HfFileSystemFile
    except ImportError:  # hub not installed (synthetic-only runs)
        return False

    # pi-lens-ignore: unchecked-throwing-call-python
    log = open(directory / f"bytes.dp{rank}.w{worker}.jsonl", "a", buffering=1)
    original = HfFileSystemFile._fetch_range

    def _fetch_range(self, start: int, end: int) -> bytes:
        began = time.time()
        payload = original(self, start, end)
        path = str(getattr(self, "path", ""))
        log.write(
            json.dumps(
                {
                    "t": began,
                    "wait_s": time.time() - began,
                    "dp": rank,
                    "worker": worker,
                    "source": source_of_path(path),
                    "path": path,
                    "start": start,
                    "end": end,
                    "bytes": len(payload),
                }
            )
            + "\n"
        )
        return payload

    HfFileSystemFile._fetch_range = _fetch_range  # type: ignore[method-assign]
    _probe_installed = True
    return True


# -- per-step counters (main process) -----------------------------------------

_counters: dict[str, Any] = {}


def _blank() -> dict[str, Any]:
    return {
        "micro_batches": 0,
        "loader_wait_s": 0.0,
        "tokens_total": 0,
        "tokens_nonpad": 0,
        "loss_tokens": {},
        "target_values": {},
    }


def observe_micro_batch(flat: dict, wait_s: float) -> None:
    """Accumulate one flat micro-batch into the current step's counters.

    ``{m}_mask`` counts loss-bearing tokens per modality (which is also the
    per-step composition measurement ADR 0014 §4 wants — exact, unlike
    inferring composition from object ids); ``{m}_values`` counts the valid
    target dimensions behind ``E_values``.
    """
    if not _counters:
        _counters.update(_blank())
    input_ids = flat.get("input_ids")
    if input_ids is None:
        return
    _counters["micro_batches"] += 1
    _counters["loader_wait_s"] += wait_s
    # pi-lens-ignore: unchecked-throwing-call-python
    _counters["tokens_total"] += int(input_ids.numel())
    # pi-lens-ignore: unchecked-throwing-call-python
    _counters["tokens_nonpad"] += int((input_ids != PAD_ID).sum().item())
    for key, value in flat.items():
        if key.endswith("_mask"):
            name = key[: -len("_mask")]
            # pi-lens-ignore: unchecked-throwing-call-python
            count = int(value.sum().item())
            if count:
                _counters["loss_tokens"][name] = (
                    _counters["loss_tokens"].get(name, 0) + count
                )
        elif key.endswith("_values"):
            name = key[: -len("_values")]
            # pi-lens-ignore: unchecked-throwing-call-python
            count = int(value.numel())
            if count:
                _counters["target_values"][name] = (
                    _counters["target_values"].get(name, 0) + count
                )


def drain_step() -> dict[str, Any]:
    """Return and reset the accumulated counters, with derived ratios.

    ``utilisation_packing`` is the §2a MFU factor: padding earns no credit,
    so a padded-out packed row is wasted compute exactly as a stall is
    wasted time.
    """
    record = dict(_counters) if _counters else _blank()
    _counters.clear()
    total = record["tokens_total"]
    record["utilisation_packing"] = (
        record["tokens_nonpad"] / total if total else 0.0
    )
    return record


def write_step(step: int, record: dict, rank: int = 0) -> None:
    """Append one per-step record to ``steps.dp{rank}.jsonl``."""
    directory = telemetry_dir()
    if directory is None:
        return
    # pi-lens-ignore: unchecked-throwing-call-python
    with open(directory / f"steps.dp{rank}.jsonl", "a") as log:
        log.write(json.dumps({"step": step, **record}) + "\n")


# -- loader wrapper -----------------------------------------------------------


class TelemetryLoader:
    """Times ``next()`` on a DataLoader and observes each micro-batch.

    Everything else proxies through, so ``loader_state_dict`` (which asks for
    ``state_dict``/``num_workers``) and checkpoint resume are unaffected. The
    wait measured here IS the GPU-visible stall: it is the main process
    blocking on the loader, regardless of how many workers feed it.
    """

    _loader: Any = None  # class attribute: keeps __getattr__ from recursing

    def __init__(self, loader):
        self._loader = loader

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        iterator = iter(self._loader)
        while True:
            began = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            observe_micro_batch(batch, time.perf_counter() - began)
            yield batch


def instrument(loader):
    """Wrap a loader when telemetry is on, else hand it back untouched."""
    return TelemetryLoader(loader) if telemetry_dir() is not None else loader

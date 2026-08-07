"""ADR 0014 §3 benchmark report: join the byte log, step log and object audit.

Usage:
    python scripts/bench_report.py --telemetry <dir> [--object-log <prefix>]
                                   [--arm B0] [--out report.json]

Reads what the instruments wrote during a benchmark window:

- ``bytes.dp*.w*.jsonl``  — one line per ``_fetch_range`` payload (§3)
- ``steps.dp*.jsonl``     — one line per step, with the decomposed MFU (§2a)
- ``{object_log}.dp*``    — one line per emitted sequence (the no-replay audit)

and reports the four metrics of §2 plus the timing distribution. Two rules
from the ADR are enforced here rather than left to the reader:

- MFU is only ever printed decomposed (§11 refuses the headline number);
- ``E_AR`` is broken into primary and replay tokens, because replay raises it
  without adding information and the split is what makes that visible.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MIB = 1024 * 1024
TIB = 1024**4


def _read_jsonl(paths):
    for path in paths:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile; no numpy dependency for a reporting script."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(fraction * len(ordered)) - 1))
    return ordered[index]


def summarise_bytes(directory: Path) -> dict:
    by_source: Counter = Counter()
    fetches: Counter = Counter()
    wait = 0.0
    for entry in _read_jsonl(sorted(directory.glob("bytes.dp*.w*.jsonl"))):
        by_source[entry["source"]] += entry["bytes"]
        fetches[entry["source"]] += 1
        wait += entry.get("wait_s", 0.0)
    total = sum(by_source.values())
    return {
        "total_bytes": total,
        "total_mib": total / MIB,
        "fetch_wait_s": wait,
        "by_source": {
            source: {
                "bytes": count,
                "mib": count / MIB,
                "share": count / total if total else 0.0,
                "fetches": fetches[source],
            }
            for source, count in by_source.most_common()
        },
    }


def _step_seconds(record: dict) -> float:
    """Wall time of one step, preferring the value the trainer recorded."""
    if "step_seconds" in record:
        return record["step_seconds"]
    # older records: recover it from wait / stall_share (exact whenever the
    # step stalled at all; a step with no stall contributes ~0 either way)
    stall = record.get("stall_share") or 0.0
    return record["loader_wait_s"] / stall if stall else record.get("loader_wait_s", 0.0)


def summarise_steps(directory: Path, slow_threshold: float) -> dict:
    rank_files = sorted(directory.glob("steps.dp*.jsonl"))
    records = list(_read_jsonl(rank_files))
    if not records:
        return {}
    n_ranks = max(len(rank_files), 1)

    # Every DP rank writes its own series for the SAME steps — they run
    # concurrently, so concatenating them would report 2x the steps. Group by
    # step number: sum the per-rank token counts (each rank packs its own
    # data), take the slowest rank's wall time (the step ends when the last
    # rank does), and take the flops estimate once (it is already global).
    by_step: dict[int, dict] = {}
    for record in records:
        step = record.get("step", 0)
        merged = by_step.setdefault(
            step,
            {"step": step, "seconds": 0.0, "wait": 0.0, "flops": 0.0,
             "nonpad": 0, "total": 0, "loss_tokens": Counter(),
             "target_values": 0, "ranks": 0},
        )
        merged["ranks"] += 1
        merged["seconds"] = max(merged["seconds"], _step_seconds(record))
        merged["wait"] = max(merged["wait"], record.get("loader_wait_s", 0.0))
        # `model_flops` is already the GLOBAL per-step estimate and every rank
        # writes the same number, so take it rather than summing. Older
        # records carry only the ratio: invert it (mfu's denominator includes
        # world_size) so a pre-`model_flops` window still reports a real MFU.
        flops = record.get("model_flops")
        if flops is None:
            flops = (
                record.get("mfu", 0.0)
                * _step_seconds(record)
                * record.get("peak_tflops_per_gpu", 0.0)
                * 1e12
                * n_ranks
            )
        merged["flops"] = max(merged["flops"], flops)
        merged["nonpad"] += record.get("tokens_nonpad", 0)
        merged["total"] += record.get("tokens_total", 0)
        merged["loss_tokens"].update(record.get("loss_tokens", {}))
        merged["target_values"] += sum(record.get("target_values", {}).values())

    steps = [by_step[k] for k in sorted(by_step)]
    durations = [s["seconds"] for s in steps]
    total_wall = sum(durations) or 1e-9
    total_wait = sum(s["wait"] for s in steps)
    loss_tokens: Counter = Counter()
    for step in steps:
        loss_tokens.update(step["loss_tokens"])
    total_loss_tokens = sum(loss_tokens.values())
    nonpad = sum(s["nonpad"] for s in steps)
    tokens_total = sum(s["total"] for s in steps)

    # TIME-WEIGHTED, not a mean of per-step ratios. This corpus is bimodal —
    # most steps stall for nothing, a few for a minute — so averaging ratios
    # reports a stall share several times lower than the one the run actually
    # paid, and buries the very behaviour being measured.
    peak = next(
        (r["peak_tflops_per_gpu"] for r in records if "peak_tflops_per_gpu" in r), 0.0
    )
    gpus = n_ranks
    total_flops = sum(s["flops"] for s in steps)
    denominator = peak * 1e12 * gpus
    busy = max(total_wall - total_wait, 1e-9)
    packing = nonpad / tokens_total if tokens_total else 0.0
    # `model_flops` already refuses padding credit, so the packing factor is
    # baked into it. MFU_busy must therefore be DEFLATED by packing to be the
    # full-occupancy busy number §2a means, or the printed identity
    # MFU = MFU_busy x (1 - stall) x packing double-counts it. With this,
    # the three factors multiply back to MFU exactly.
    busy_flops = total_flops / packing if packing else 0.0

    return {
        "n_steps": len(steps),
        "dp_ranks": gpus,
        "step_seconds": {
            "mean": total_wall / len(steps),
            "p50": percentile(durations, 0.50),
            "p95": percentile(durations, 0.95),
            "p99": percentile(durations, 0.99),
            "slow_steps": sum(1 for d in durations if d > slow_threshold),
            "slow_threshold_s": slow_threshold,
            "total_s": total_wall,
        },
        # §11: never a headline MFU on its own — the factors always travel
        # with it, or a pipeline effect gets read as a model-shape effect
        "mfu": {
            "mfu": total_flops / (total_wall * denominator) if denominator else 0.0,
            "mfu_busy": busy_flops / (busy * denominator) if denominator else 0.0,
            "stall_share": total_wait / total_wall,
            "utilisation_packing": packing,
            "flops_per_token": (
                total_flops / (3 * total_loss_tokens) if total_loss_tokens else 0.0
            ),
            "peak_tflops_per_gpu": peak,
        },
        "loader_wait_s": total_wait,
        "tokens_nonpad": nonpad,
        "tokens_total": tokens_total,
        "target_values": sum(s["target_values"] for s in steps),
        "loss_tokens": dict(loss_tokens.most_common()),
        # §4: realised modality composition. Short windows swing 160x on this
        # corpus, so an A/B that does not report it is not comparing like
        # with like.
        "composition": {
            name: count / total_loss_tokens
            for name, count in loss_tokens.most_common()
        }
        if total_loss_tokens
        else {},
    }


def summarise_objects(prefix: Path | None) -> dict:
    """Split emitted sequences into primary and replay, from the audit log."""
    if prefix is None:
        return {}
    logs = sorted(prefix.parent.glob(f"{prefix.name}.dp*"))
    if not logs:
        return {}
    primary, replicas, seen = 0, 0, set()
    bases = set()
    duplicates = 0
    for path in logs:
        with open(path) as handle:
            for line in handle:
                object_id = line.strip()
                if not object_id:
                    continue
                if object_id in seen:
                    duplicates += 1
                seen.add(object_id)
                base, _, suffix = object_id.partition("#")
                bases.add(base)
                if suffix:
                    replicas += 1
                else:
                    primary += 1
    return {
        "sequences": primary + replicas,
        "primary": primary,
        "replay": replicas,
        "distinct_base_objects": len(bases),
        # the exactly-once gate: replica 0 keeps the base id, replicas are
        # suffixed, so every logged line must be unique
        "duplicate_lines": duplicates,
        "replay_factor": (primary + replicas) / len(bases) if bases else 0.0,
    }


def build_report(directory: Path, object_log: Path | None, arm: str, slow: float):
    wire = summarise_bytes(directory)
    steps = summarise_steps(directory, slow)
    objects = summarise_objects(object_log)

    mib = wire["total_mib"] or 1.0
    loss_tokens = sum(steps.get("loss_tokens", {}).values())
    # §2: E_values counts OBSERVED target dimensions. A replica re-presents
    # the same values under a different factorisation, so the raw per-batch
    # count double-counts it and E_values would rise with ar_replicas — which
    # is precisely the gaming §2 says it must be immune to. Deflate by the
    # measured replay factor so only distinct observations are counted.
    primary_fraction = (
        objects["primary"] / objects["sequences"] if objects.get("sequences") else 1.0
    )
    report = {
        "arm": arm,
        "telemetry_dir": str(directory),
        "wire": wire,
        "steps": steps,
        "objects": objects,
        # E_values must NOT move under replay or per-band; E_AR should, and
        # the primary split is what shows how much of it is new information.
        "E_values_per_mib": steps.get("target_values", 0) * primary_fraction / mib,
        "E_values_presented_per_mib": steps.get("target_values", 0) / mib,
        "E_AR_per_mib": loss_tokens / mib,
        "E_AR_primary_per_mib": loss_tokens * primary_fraction / mib,
        "tib_downloaded": wire["total_bytes"] / TIB,
    }
    return report


def render(report: dict) -> str:
    lines = [f"# ADR 0014 §3 benchmark — arm {report['arm']}", ""]
    wire = report["wire"]
    lines.append(f"Wire: {wire['total_mib']:.1f} MiB over "
                 f"{sum(s['fetches'] for s in wire['by_source'].values())} fetches")
    for source, stats in wire["by_source"].items():
        lines.append(
            f"  {source:<18} {stats['mib']:9.1f} MiB  {stats['share']:6.1%}  "
            f"{stats['fetches']:5d} fetches"
        )
    steps = report.get("steps") or {}
    if steps:
        timing = steps["step_seconds"]
        lines += [
            "",
            f"Steps: {steps['n_steps']}  mean {timing['mean']:.2f}s  "
            f"p50 {timing['p50']:.2f}s  p95 {timing['p95']:.2f}s  "
            f"p99 {timing['p99']:.2f}s  "
            f"slow(>{timing['slow_threshold_s']:.0f}s) {timing['slow_steps']}",
            "",
            "MFU (decomposed — never quote the headline alone, §11):",
        ]
        mfu = steps["mfu"]
        identity = (
            mfu["mfu_busy"]
            * (1 - mfu["stall_share"])
            * mfu["utilisation_packing"]
        )
        lines += [
            f"  MFU               {mfu['mfu']:.3%}",
            f"  = MFU_busy        {mfu['mfu_busy']:.3%}",
            f"  x (1 - stall)     {1 - mfu['stall_share']:.3f}   "
            f"(stall_share {mfu['stall_share']:.1%})",
            f"  x packing         {mfu['utilisation_packing']:.3f}",
            f"  = {identity:.3%}   (identity check)",
            f"  FLOPs/token       {mfu['flops_per_token']:.3e}   "
            f"(peak {mfu['peak_tflops_per_gpu']:.0f} TFLOP/s/GPU)",
        ]
        if steps.get("composition"):
            lines += ["", "Composition (share of loss-bearing tokens):"]
            for name, share in list(steps["composition"].items())[:10]:
                lines.append(f"  {name:<28} {share:6.2%}")
    objects = report.get("objects") or {}
    if objects:
        lines += [
            "",
            f"Sequences: {objects['sequences']} "
            f"({objects['primary']} primary + {objects['replay']} replay) "
            f"over {objects['distinct_base_objects']} distinct objects "
            f"= {objects['replay_factor']:.2f}x",
            f"Exactly-once audit: {objects['duplicate_lines']} duplicate lines "
            f"({'PASS' if not objects['duplicate_lines'] else 'FAIL'})",
        ]
    lines += [
        "",
        f"E_values  {report['E_values_per_mib']:,.0f} distinct target values / MiB"
        f"  ({report['E_values_presented_per_mib']:,.0f} presented)",
        f"E_AR      {report['E_AR_per_mib']:,.0f} loss-bearing tokens / MiB "
        f"({report['E_AR_primary_per_mib']:,.0f} primary)",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telemetry", required=True, help="$ASTROPT3_TELEMETRY_DIR")
    parser.add_argument("--object-log", default=None, help="object_id_log prefix")
    parser.add_argument("--arm", default="unnamed")
    parser.add_argument(
        "--slow-seconds",
        type=float,
        default=60.0,
        help="a step slower than this counts as a stall (§3 slow-step count)",
    )
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    report = build_report(
        Path(args.telemetry),
        Path(args.object_log) if args.object_log else None,
        args.arm,
        args.slow_seconds,
    )
    print(render(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

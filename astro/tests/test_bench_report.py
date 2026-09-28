"""ADR 0014 §3 report aggregation — the arithmetic that decides the arms."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import bench_report  # noqa: E402


def _write(directory: Path, ranks=2, steps=None):
    steps = steps or [
        # (seconds, wait) — bimodal on purpose: this corpus stalls rarely and
        # hugely, which is what a mean of ratios gets wrong
        (1.0, 0.0),
        (1.0, 0.0),
        (1.0, 0.0),
        (97.0, 96.0),
    ]
    for rank in range(ranks):
        lines = []
        for i, (seconds, wait) in enumerate(steps):
            lines.append(
                json.dumps(
                    {
                        "step": i,
                        "step_seconds": seconds,
                        "loader_wait_s": wait,
                        # global estimate: every rank writes the SAME number
                        "model_flops": 1e14,
                        "tokens_total": 1000,
                        "tokens_nonpad": 900,
                        "rows": 50,
                        "utilisation_packing": 0.9,
                        "loss_tokens": {"images": 100},
                        "target_values": {"images": 19200},
                        "peak_tflops_per_gpu": 312.0,
                    }
                )
            )
        (directory / f"steps.dp{rank}.jsonl").write_text("\n".join(lines) + "\n")


def test_ranks_are_grouped_not_concatenated(tmp_path):
    """Two DP ranks run the SAME steps concurrently, not twice as many."""
    _write(tmp_path, ranks=2)
    steps = bench_report.summarise_steps(tmp_path, slow_threshold=60.0)
    assert steps["n_steps"] == 4
    assert steps["dp_ranks"] == 2
    # token counts sum across ranks (each packs its own data)
    assert steps["tokens_total"] == 4 * 1000 * 2


def test_rows_per_s_is_end_to_end_wall_clock_throughput(tmp_path):
    """rows/s uses total wall time, not loader-wait time (that's telemetry's own metric)."""
    _write(tmp_path, ranks=2)
    steps = bench_report.summarise_steps(tmp_path, slow_threshold=60.0)
    # rows sum across ranks (each packs its own data), like token counts
    assert steps["rows"] == 4 * 50 * 2
    total_wall = 1.0 + 1.0 + 1.0 + 97.0
    assert steps["rows_per_s"] == pytest.approx(steps["rows"] / total_wall)


def test_stall_share_is_time_weighted_not_a_mean_of_ratios(tmp_path):
    """The bug this guards: 3 free steps + 1 huge stall.

    Mean of per-step ratios = (0 + 0 + 0 + 0.99)/4 = 25%.
    The share actually paid = 96 / 100 = 96%. Only the second is the truth.
    """
    _write(tmp_path)
    mfu = bench_report.summarise_steps(tmp_path, slow_threshold=60.0)["mfu"]
    assert mfu["stall_share"] == pytest.approx(96.0 / 100.0)
    assert mfu["stall_share"] > 0.9  # never the 25% a ratio mean would give


def test_the_mfu_decomposition_identity_holds(tmp_path):
    """MFU = MFU_busy x (1 - stall) x packing, exactly (ADR 0014 §2a).

    It only holds if MFU_busy is deflated by packing: `model_flops` already
    refuses padding credit, so leaving it in double-counts the factor.
    """
    _write(tmp_path)
    mfu = bench_report.summarise_steps(tmp_path, slow_threshold=60.0)["mfu"]
    identity = mfu["mfu_busy"] * (1 - mfu["stall_share"]) * mfu["utilisation_packing"]
    assert identity == pytest.approx(mfu["mfu"], rel=1e-9)
    assert mfu["utilisation_packing"] == pytest.approx(0.9)


def test_flops_are_taken_once_across_ranks_not_summed(tmp_path):
    """`model_flops` is already global; summing it would double the MFU."""
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    _write(one, ranks=1)
    _write(two, ranks=2)
    # same flops, twice the GPUs in the denominator -> half the MFU
    a = bench_report.summarise_steps(one, slow_threshold=60.0)["mfu"]["mfu"]
    b = bench_report.summarise_steps(two, slow_threshold=60.0)["mfu"]["mfu"]
    assert b == pytest.approx(a / 2)


def test_e_values_does_not_move_under_replay(tmp_path):
    """§2: replay re-presents the same observations; E_values must be immune.

    E_AR is expected to double — that split is the point.
    """
    _write(tmp_path)
    (tmp_path / "bytes.dp0.w0.jsonl").write_text(
        json.dumps(
            {"t": 0, "wait_s": 0.0, "dp": 0, "worker": 0, "source": "legacy",
             "path": "p", "start": 0, "end": 1, "bytes": bench_report.MIB}
        )
        + "\n"
    )
    log = tmp_path / "objects.log"
    # 2 base objects, each with one replica: replay factor 2.0
    (tmp_path / "objects.log.dp0").write_text("a\nb\na#1\nb#1\n")

    plain = bench_report.build_report(tmp_path, None, "no-audit", 60.0)
    replayed = bench_report.build_report(tmp_path, log, "replay2", 60.0)

    # presented values are identical; distinct values halve once the audit
    # reveals that half the sequences were replicas
    assert replayed["E_values_presented_per_mib"] == plain["E_values_presented_per_mib"]
    assert replayed["E_values_per_mib"] == pytest.approx(
        plain["E_values_per_mib"] / 2
    )
    assert replayed["E_AR_per_mib"] == pytest.approx(2 * replayed["E_AR_primary_per_mib"])
    assert replayed["objects"]["duplicate_lines"] == 0

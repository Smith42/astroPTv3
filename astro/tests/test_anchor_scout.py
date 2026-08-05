"""Offline checks for the deterministic ADR 0013 anchor scout."""

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/scout_legacy_anchor.py"
SPEC = importlib.util.spec_from_file_location("scout_legacy_anchor", SCRIPT)
assert SPEC and SPEC.loader
scout = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scout)


def test_scout_sampling_and_bootstrap_are_deterministic():
    skymap = np.arange(16)
    assert scout.rows_in_cell(skymap, 10, 3) == 3
    assert scout.rows_in_cell(skymap, 9, 1) == sum(range(4, 8))

    rows = [
        {"stratum": stratum, "order": 1, "pixel": pixel}
        for stratum in ("a", "b")
        for pixel in range(3)
    ]
    first = scout.deterministic_order(rows, seed=13)
    second = scout.deterministic_order(rows, seed=13)
    assert first == second
    assert {row["stratum"] for row in first[:2]} == {"a", "b"}

    metrics = [
        {"matches": 10, "physical_bytes": 100},
        {"matches": 20, "physical_bytes": 200},
    ]
    interval = scout.bootstrap_ratio(metrics, seed=14, replicates=100)
    assert interval["matches_per_byte"] == 0.1
    assert interval["ci95_low"] == interval["ci95_high"] == 0.1

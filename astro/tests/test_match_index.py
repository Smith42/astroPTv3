"""Small unit checks for the offline match-index builder."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
reciprocal_pairs = importlib.import_module("_match_index").reciprocal_pairs


def test_reciprocal_pairs_rejects_one_way_neighbours():
    forward = [
        ("image-a", "spectrum-1"),
        ("image-b", "spectrum-1"),
        ("image-c", "spectrum-2"),
    ]
    reverse = {("spectrum-1", "image-a"), ("spectrum-2", "image-c")}

    assert reciprocal_pairs(forward, reverse) == {
        ("image-a", "spectrum-1"),
        ("image-c", "spectrum-2"),
    }

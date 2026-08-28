"""Compatibility shim — the implementation lives in ``mmu_stream.match_index``.

Extracted from this module into the mmu-stream package (pinned in
pyproject). Kept so local imports survive the migration; new code should
import ``mmu_stream.match_index`` directly.
"""

from mmu_stream.match_index import Cell, MatchGraph, load_source_graph

__all__ = ["Cell", "MatchGraph", "load_source_graph"]

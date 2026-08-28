#!/usr/bin/env python
"""Alias for ``python -m mmu_stream.merge_match_index``.

The pivot moved to the mmu-stream package (pinned in pyproject); this
wrapper keeps ``uv run python scripts/merge_match_index.py ...`` working.
See the module docstring upstream for usage.
"""

from mmu_stream.merge_match_index import main

if __name__ == "__main__":
    raise SystemExit(main())

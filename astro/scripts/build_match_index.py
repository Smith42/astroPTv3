#!/usr/bin/env python
"""Alias for ``python -m mmu_stream.build_match_index``.

The builder moved to the mmu-stream package (pinned in pyproject); this
wrapper keeps the launch scripts (``build_remaining_spokes.sh``) and the
``uv run --extra data python scripts/build_match_index.py ...`` invocation
working. See the module docstring upstream for the full usage example.
"""

from mmu_stream.build_match_index import main

if __name__ == "__main__":
    raise SystemExit(main())

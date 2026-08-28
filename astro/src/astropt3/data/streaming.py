"""Hub-and-spoke MMU streaming — re-exported from the mmu-stream package.

The implementation (stream assembly, decode, split/shuffle, match-index
readers) was extracted into ``mmu_stream.streaming``, pinned in pyproject.
This shim keeps the loader, telemetry, eval, scripts, and tests on their
existing import sites — including the ``open_stream`` monkeypatch target in
the resume tests, which works because consumers import the name lazily.
New code should import ``mmu_stream.streaming`` directly.

The two data-root sentinels stay local: they name astropt3 loader modes,
not stream concepts.
"""

from mmu_stream.streaming import (
    HSC_IMAGE_SHAPE,
    IMAGES_CATALOG,
    MATCH_INDEX_ENV,
    SOURCE_CATALOGS,
    SOURCE_GRAPH_ASSEMBLY,
    SPECTRA_CATALOG,
    VAL_PARTITIONS,
    _SOURCE_COLUMNS,
    _SPECTRUM_LEAVES,
    _partition_owner,
    _source_graph_examples,
    _spectrum_owners,
    _spectrum_part,
    assembly_and_revisions,
    attach_source,
    catalog_files,
    crossmatch_dataset,
    decode_record,
    load_match_index,
    open_stream,
    owned_by_rank,
    resolve_match_index,
    shuffled,
    source_assembly_for_index,
    source_only_record,
    split_files,
    split_of_cell,
    union_features,
)

MMU_ROOT = "mmu"
SYNTHETIC_ROOT = "synthetic"

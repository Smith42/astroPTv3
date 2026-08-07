#!/bin/bash
# Build the SDSS and galaxies-with-hats spokes of the ADR 0013 North index,
# then regenerate the one-row-per-anchor merged index.
#
# DESI, HSC and PROVABGS are already built in $IDX. The two here are the long
# ones (~2.4 h and ~4.3 h at 8 workers, measured against DESI's 200 partitions
# in 571 s) so they run in parallel and the script waits for both.
#
#   bash astro/scripts/build_remaining_spokes.sh
#   WORKERS=8 bash astro/scripts/build_remaining_spokes.sh   # nothing else on the hub
#
# WORKERS is per build, so the default 4 keeps total hub traffic at the 8 that
# ran clean; raise it only if no training job is streaming at the same time.
set -euo pipefail

REPO=${REPO:-/beegfs/general/mjsmith/foundation/astroPT_all/astroPTv3}
IDX=${IDX:-/beegfs/general/mjsmith/foundation/astroPT_all/astroPTv3_index/north-v2}
LOGS=${LOGS:-/beegfs/general/mjsmith/foundation/astroPT_all/astroPTv3_index/build-logs}
WORKERS=${WORKERS:-4}
MERGED=${MERGED:-${IDX}-merged-5spoke}

# logs live outside $IDX: load_source_graph reads the index directory
mkdir -p "$IDX" "$LOGS"
cd "$REPO/astro"

NORTH_REV=f634744d3c44dd4fde0dee3172d4887c5e3c31c0
SDSS_REV=da175ca9bb931f4301ab950431922ba9f99089ba
GWH_REV=c0188b776c4ce6312a805a04cbc25c891a075933

anchor=(
	--anchor-catalog "hf://datasets/UniverseTBD/mmu_ssl_legacysurvey_north@$NORTH_REV"
	--anchor-source legacy_north
	--anchor-revision "$NORTH_REV"
	--workers "$WORKERS"
)

# SDSS ids are padded byte literals; --partner-strip-id canonicalizes them to
# what streaming._source_id produces, or every pointer misses at train time
uv run --extra data python scripts/build_match_index.py "${anchor[@]}" \
	--partner-catalog "hf://datasets/UniverseTBD/mmu_sdss_sdss@$SDSS_REV" \
	--partner-source sdss --partner-revision "$SDSS_REV" \
	--partner-strip-id \
	--out "$IDX/sdss.parquet" >"$LOGS/sdss.build.log" 2>&1 &
sdss=$!

# the collection root is .../galaxies-with-hats@REV/train, and this is the one
# spoke with no published margin cache — expect "margin cache MISSING" in the
# first log line, meaning matches are lossy at partition edges
uv run --extra data python scripts/build_match_index.py "${anchor[@]}" \
	--partner-catalog "hf://datasets/Smith42/galaxies-with-hats@$GWH_REV/train" \
	--partner-source galaxies_train --partner-revision "$GWH_REV" \
	--partner-id-column dr8_id \
	--out "$IDX/galaxies_train.parquet" >"$LOGS/galaxies.build.log" 2>&1 &
gwh=$!

echo "sdss pid $sdss -> $LOGS/sdss.build.log"
echo "galaxies pid $gwh -> $LOGS/galaxies.build.log"
echo "progress: tr '\\r' '\\n' < $LOGS/sdss.build.log | tail -2"

status=0
wait "$sdss" || { echo "SDSS BUILD FAILED, see $LOGS/sdss.build.log" >&2; status=1; }
wait "$gwh" || { echo "GALAXIES BUILD FAILED, see $LOGS/galaxies.build.log" >&2; status=1; }
[ "$status" -eq 0 ] || exit "$status"

# a new directory, so a running job's index is never swapped underneath it
uv run python scripts/merge_match_index.py --spokes "$IDX" --out "$MERGED/match_index.parquet"

uv run python - "$MERGED" <<'PY'
import sys
sys.path.insert(0, "src")
from astropt3.data.match_index import load_source_graph

graph = load_source_graph(sys.argv[1])
anchors = sum(len(cell) for cell in graph.matches.values())
edges = sum(len(s) for cell in graph.matches.values() for s in cell.values())
print(f"anchor {graph.anchor_source} | sources {sorted(graph.partner_revisions)}")
print(f"cells {len(graph.matches)} | anchors {anchors} | edges {edges}")
PY

cat <<EOF

Done. Point a run at it with:
  export ASTROPT3_MATCH_INDEX=$MERGED

Two consequences of adding spokes:
  - derive new runs from astropt3-70m-jetformer-north-5spoke-replay4.yaml
    (all 47 modalities, vocab 145);
  - more sources changes record ORDER, so stream states saved against the
    3-spoke index are rejected on resume. Weights still load.
EOF

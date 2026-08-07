#!/bin/bash
# ADR 0014 §3 frozen benchmark: run one arm's window with telemetry on and
# report it. Windows are short and repeated, not training runs.
#
# Usage, from the REPO ROOT:
#   bash astro/scripts/run_benchmark.sh bench-north-5spoke-b0
#   REPS=3 bash astro/scripts/run_benchmark.sh bench-north-5spoke-b1d
#
# Each repetition starts from the SAME frozen state (checkpoints and stream
# state are wiped between reps on purpose — a rep that resumed would measure
# a different part of the cell order and the arms would stop being
# comparable). Telemetry lands in <checkpoints>/telemetry-rep<N>/ and the
# report beside it.
set -euo pipefail

RUN=${1:?usage: run_benchmark.sh <config-basename>}
REPS=${REPS:-3}

REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$REPO_ROOT"
source /beegfs/general/mjsmith/gpuenv/bin/activate

CK=/beegfs/general/mjsmith/foundation/astroPT_all/astroPTv3_checkpoints/$RUN
export CUDA_DEVICE_MAX_CONNECTIONS=1 # required by nanotron's comm overlap
export WANDB_MODE=${WANDB_MODE:-disabled}

for rep in $(seq 1 "$REPS"); do
	echo "=== $RUN rep $rep/$REPS ==="
	rm -rf "$CK"
	mkdir -p "$CK"
	export ASTROPT3_TELEMETRY_DIR="$CK/telemetry-rep$rep"
	mkdir -p "$ASTROPT3_TELEMETRY_DIR"

	python -m torch.distributed.run \
		--nproc-per-node=${NPROC:-2} \
		--rdzv-backend=c10d \
		--rdzv-endpoint=localhost:0 \
		--max-restarts=0 \
		nanotron/run_train.py \
		--config-file "astro/configs/nanotron/$RUN.yaml" \
		>>"$CK/train-rep$rep.log" 2>&1 || {
		echo "rep $rep FAILED — see $CK/train-rep$rep.log"
		tail -30 "$CK/train-rep$rep.log"
		exit 1
	}

	python astro/scripts/bench_report.py \
		--telemetry "$ASTROPT3_TELEMETRY_DIR" \
		--object-log "$CK/objects.log" \
		--arm "$RUN-rep$rep" \
		--out "$CK/report-rep$rep.json" | tee "$CK/report-rep$rep.txt"
done

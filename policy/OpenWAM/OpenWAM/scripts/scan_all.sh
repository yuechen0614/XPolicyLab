#!/usr/bin/env bash
# Full-load verification pipeline for the currently active mixture datasets.
#
# Runs (sequentially, so each step saturates the CPU decode workers):
#   Phase 1  scan WorldEngine / AgiBotWorld / RoboCOIN / OXE-DROID /
#            InternData-A1 (episode mode: one high-frame window per episode —
#            finds truncated / undecodable clips) while bypassing _safe_get.
#   Emit     merge failures into each reader's per-bucket
#            meta/excluded_episodes.json, unioned with any existing exclusions.
#   Phase 2  verify the real five-source `mixture`
#            through the real retry path with a tolerant wrapper — confirms a full
#            mixed load will not crash now that bad episodes are excluded.
#
# The load path is CPU-decode + disk-IO bound (PyAV), NOT GPU-bound — the 4×H200
# GPUs stay idle; parallelism comes from NUM_WORKERS decode processes.
#
# RUN UNDER tmux (this may take a while and you may disconnect):
#   tmux new -s verify_load
#   bash scripts/scan_all.sh
#   # detach: Ctrl-b d   |   reattach: tmux attach -t verify_load
#
# Knobs (env vars):
#   MODE=episode|window   (default episode; window = every sample, decode-bound / days)
#   NUM_WORKERS=N         (default: nproc-2)
#   OUT_DIR=path          (default scan_out)
#   VERIFY_LIMIT=N        (phase-2 sampled windows; default 2000000; 0 = full mixture epoch)
#   DATASETS="worldengine agibotworld robocoin oxe_droid interndata_a1"
#            (phase-1 targets)
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${MODE:-episode}"
# nproc-2, but clamp to >=1 (nproc is 1-2 on small containers → 0/-1 is invalid).
_NCPU="$(nproc)"
NUM_WORKERS="${NUM_WORKERS:-$(( _NCPU > 2 ? _NCPU - 2 : 1 ))}"
OUT_DIR="${OUT_DIR:-scan_out}"
VERIFY_LIMIT="${VERIFY_LIMIT:-2000000}"
DATASETS="${DATASETS:-worldengine agibotworld robocoin oxe_droid interndata_a1}"
PY=python3
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "$LOG_DIR"

echo "=== scan_all: MODE=$MODE NUM_WORKERS=$NUM_WORKERS OUT_DIR=$OUT_DIR ==="
date

# ---- Phase 1: per-dataset deep scan + emit exclusions ----
for ds in $DATASETS; do
  echo ""
  echo ">>> [$(date +%H:%M:%S)] Phase 1 scan: $ds (mode=$MODE)"
  $PY scripts/scan_dataset.py scan --config "$ds" --mode "$MODE" \
      --num-workers "$NUM_WORKERS" --out-dir "$OUT_DIR" 2>&1 | tee "$LOG_DIR/scan_${ds}.log"

  echo ">>> [$(date +%H:%M:%S)] Emit exclusions: $ds"
  $PY scripts/scan_dataset.py emit --config "$ds" --out-dir "$OUT_DIR" 2>&1 | tee "$LOG_DIR/emit_${ds}.log"
done

# ---- Phase 2: verify the real mixture through the retry path ----
echo ""
echo ">>> [$(date +%H:%M:%S)] Phase 2 verify: mixture (limit=$VERIFY_LIMIT)"
VERIFY_ARGS=(--config mixture --num-workers "$NUM_WORKERS" --out-dir "$OUT_DIR")
if [ "$VERIFY_LIMIT" != "0" ]; then
  VERIFY_ARGS+=(--limit "$VERIFY_LIMIT")
fi
$PY scripts/scan_dataset.py verify-mixed "${VERIFY_ARGS[@]}" 2>&1 | tee "$LOG_DIR/verify_mixture.log"

echo ""
echo "=== scan_all DONE ==="
date
echo "Exclusions written per scanned dataset ($DATASETS):"
echo "  active LeRobot-v3 readers -> per-bucket meta/excluded_episodes.json."
echo "Logs under $LOG_DIR."

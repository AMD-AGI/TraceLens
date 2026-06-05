#!/bin/bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# End-to-end triage: discover tracelens folders, run checks in parallel,
# then aggregate into a summary report + reproducer packages.
#
# Usage:
#   bash run_triage.sh <traces_root> [report_dir] [parallelism] [num_reproducers]
#
# Examples:
#   bash run_triage.sh /wekafs/users/d2d10ce6177d46ed841e1f205b74b86f
#   bash run_triage.sh /wekafs/users/d2d10ce6177d46ed841e1f205b74b86f ./my_report 32
#
# Output:
#   <report_dir>/run_dirs.txt         - Discovered tracelens folders
#   <report_dir>/logs/*.log           - Per-run triage output
#   <report_dir>/aggregated_triage.csv
#   <report_dir>/summary_report.md
#   <report_dir>/reproducers/*.tar.gz
#
###############################################################################

set -euo pipefail

TRACES_ROOT="${1:?Usage: $0 <traces_root> [report_dir] [parallelism] [num_reproducers]}"
REPORT_DIR="${2:-./triage_report}"
PARALLELISM="${3:-4}"
TOP_REPRODUCERS="${4:-3}"

mkdir -p "$REPORT_DIR/logs"

echo "============================================================"
echo "TRIAGE"
echo "============================================================"
echo "  Traces root:  $TRACES_ROOT"
echo "  Report dir:   $REPORT_DIR"
echo "  Parallelism:  $PARALLELISM"
echo ""

# ---------------------------------------------------------------
# Step 1: Discover and run triage in parallel
# ---------------------------------------------------------------
echo "Step 1: Discovering tracelens folders..."

ls -d "$TRACES_ROOT"/*/*/kernel-agent/runs/*/*/tracelens 2>/dev/null \
    | sort -u > "$REPORT_DIR/run_dirs.txt"

N_DIRS=$(wc -l < "$REPORT_DIR/run_dirs.txt")
echo "  Found $N_DIRS tracelens folders."
echo "  Running triage (--detailed) with parallelism=$PARALLELISM..."

cat "$REPORT_DIR/run_dirs.txt" | xargs -P "$PARALLELISM" -I {} bash -c '
  SAFE=$(echo "{}" | tr "/" "_")
  python -m TraceLens.Agent.Analysis.triage.runner --run-dir "{}" --detailed \
      > "'"$REPORT_DIR"'/logs/${SAFE}.log" 2>&1 || true
'

N_LOGS=$(find "$REPORT_DIR/logs" -name "*.log" -size +0c | wc -l)
echo "  Triage complete. $N_LOGS non-empty log files."
echo ""

# ---------------------------------------------------------------
# Step 2: Post-process
# ---------------------------------------------------------------
echo "Step 2: Aggregating and building report..."

python -m TraceLens.Agent.Analysis.triage.postprocess \
    --traces-root "$TRACES_ROOT" \
    --report-dir "$REPORT_DIR" \
    --top-reproducers "$TOP_REPRODUCERS"

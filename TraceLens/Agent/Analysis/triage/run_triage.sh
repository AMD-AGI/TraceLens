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
# Discovery is layout-agnostic. By default it finds tracelens output folders by
# recursively locating any directory that contains perf_report_csvs/ under
# <traces_root>. To force a specific layout (faster for known trees), set
# TRIAGE_DISCOVERY_GLOB to a shell glob relative to <traces_root>, e.g. the
# Hyperloom session layout:
#
#   TRIAGE_DISCOVERY_GLOB='*/*/kernel-agent/runs/*/*/tracelens' \
#     bash run_triage.sh /path/to/sessions
#
# General example, any traces root:
#   bash run_triage.sh /path/to/tracelens_outputs ./my_report 8
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

# Optional explicit discovery glob (relative to TRACES_ROOT). Empty => auto-discover.
DISCOVERY_GLOB="${TRIAGE_DISCOVERY_GLOB:-}"

mkdir -p "$REPORT_DIR/logs"

echo "============================================================"
echo "TRIAGE"
echo "============================================================"
echo "  Traces root:  $TRACES_ROOT"
echo "  Report dir:   $REPORT_DIR"
echo "  Parallelism:  $PARALLELISM"
echo "  Discovery:    ${DISCOVERY_GLOB:-<auto: dirs containing perf_report_csvs/>}"
echo ""

# ---------------------------------------------------------------
# Step 1: Discover and run triage in parallel
# ---------------------------------------------------------------
echo "Step 1: Discovering tracelens folders..."

if [ -n "$DISCOVERY_GLOB" ]; then
    # A non-matching glob makes the pipeline non-zero
    { ls -d "$TRACES_ROOT"/$DISCOVERY_GLOB 2>/dev/null || true; } \
        | sort -u > "$REPORT_DIR/run_dirs.txt"
else
    # Layout-agnostic: a tracelens output dir is one that holds perf_report_csvs/.
    { find "$TRACES_ROOT" -type d -name perf_report_csvs 2>/dev/null || true; } \
        | sed 's:/perf_report_csvs$::' \
        | sort -u > "$REPORT_DIR/run_dirs.txt"
fi

N_DIRS=$(wc -l < "$REPORT_DIR/run_dirs.txt")
if [ "$N_DIRS" -eq 0 ]; then
    echo "ERROR: no tracelens folders found under $TRACES_ROOT" >&2
    echo "  Discovery: ${DISCOVERY_GLOB:-<auto: dirs containing perf_report_csvs/>}" >&2
    echo "  Set TRIAGE_DISCOVERY_GLOB to the layout under this root, or check the path." >&2
    exit 1
fi
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

# Reuse the already-discovered run dirs instead of re-walking the whole tree
python -m TraceLens.Agent.Analysis.triage.postprocess \
    --mapping "$REPORT_DIR/run_dirs.txt" \
    --report-dir "$REPORT_DIR" \
    --top-reproducers "$TOP_REPRODUCERS"

#!/usr/bin/env bash
set -euo pipefail

# Environment config (edit these for your setup)
REPO_ROOT="$(pwd)"
NODE="tw028"
CONTAINER="tracelens_evals"

# Eval directories
STANDALONE_DIR="TraceLens/AgenticMode/Standalone"
EVALS_DIR="$REPO_ROOT/evals"
CSV_FILE="$EVALS_DIR/test_traces.csv"
RESULTS_ROOT="$EVALS_DIR/results"
PREFIX="docker exec -w $REPO_ROOT $CONTAINER"

mkdir -p "$RESULTS_ROOT"
ssh "$NODE" "$PREFIX bash -c 'mkdir -p $RESULTS_ROOT && chmod -R 777 $RESULTS_ROOT'"

echo "========================================="
echo "  Standalone Analysis Evaluation"
echo "========================================="
echo ""

# -- Phase 1: Run standalone analysis, serial --

echo "Phase 1: Running standalone analysis..."
echo ""

while IFS=, read -r id sub_category trace_path reference_dir platform; do
    OUTPUT_DIR="$(dirname "$trace_path")/analysis_output"
    CASE_RESULTS="$RESULTS_ROOT/$id"
    mkdir -p "$CASE_RESULTS"
    ssh "$NODE" "$PREFIX bash -c 'mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR'"
    ssh "$NODE" "$PREFIX bash -c 'mkdir -p $CASE_RESULTS && chmod -R 777 $CASE_RESULTS'"

    # Phase 1: Analysis
    echo "  [$id] Analyzing $trace_path on $platform..."
    (
        cd "$STANDALONE_DIR"
        agent --print --force --trust "Run standalone analysis on $trace_path with platform $platform, node $NODE, container $CONTAINER, output to $OUTPUT_DIR"
    ) 2>&1 | tee "$CASE_RESULTS/analysis.log"
    echo "  [$id] Analysis complete."

    # Phase 2: Evals in parallel
    echo "  [$id] Running workflow + quality evals in parallel..."

    echo "    -> Launching sub-agent: workflow-eval"
    (
        cd "$EVALS_DIR"
        agent --print --force --trust "Run workflow eval skill on $OUTPUT_DIR for test case $id. Write results to $CASE_RESULTS/workflow_eval_results.csv"
    ) 2>&1 | tee "$CASE_RESULTS/workflow_eval.log" &

    echo "    -> Launching sub-agent: quality-eval"
    (
        cd "$EVALS_DIR"
        agent --print --force --trust "Run quality eval skill on $OUTPUT_DIR with reference $reference_dir for test case $id. Write results to $CASE_RESULTS/quality_eval_results.csv"
    ) 2>&1 | tee "$CASE_RESULTS/quality_eval.log" &

    wait
    echo "  [$id] Evals complete."

    # Merge results for this test case
    ssh "$NODE" "$PREFIX python3 $EVALS_DIR/eval_scripts/merge_results.py --results-dir $CASE_RESULTS --output $CASE_RESULTS/eval_summary.csv"
    echo "  [$id] Summary written to $CASE_RESULTS/eval_summary.csv"
    echo ""
done < <(tail -n +2 "$CSV_FILE")

echo ""
echo "========================================="
echo "  Eval run finished."
echo "========================================="

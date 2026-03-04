#!/usr/bin/env bash
set -euo pipefail

# Environment config (edit these for your setup)
NODE=""
CONTAINER=""

# Eval directories
REPO_ROOT="$(pwd)"
STANDALONE_DIR="TraceLens/AgenticMode/Standalone"
EVALS_DIR="$REPO_ROOT/evals"
RESULTS_ROOT="$EVALS_DIR/results"
TEST_TRACES_CSV="$EVALS_DIR/unit_test_traces.csv"
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

    # Create output directories
    CASE_RESULTS="$RESULTS_ROOT/$id"
    OUTPUT_DIR="$CASE_RESULTS/analysis_output"
    ssh "$NODE" "$PREFIX bash -c 'mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR'"
    ssh "$NODE" "$PREFIX bash -c 'mkdir -p $CASE_RESULTS && chmod -R 777 $CASE_RESULTS'"

    # Phase 1: Agentic Mode Standalone Analysis
    echo "  [$id] Analyzing $trace_path on $platform..."
    (
        cd "$STANDALONE_DIR"
        agent --print --force --trust "Run standalone analysis on $trace_path with platform $platform, node $NODE, container $CONTAINER, output to $OUTPUT_DIR"
    ) 2>&1 | tee "$CASE_RESULTS/analysis.log"
    echo "  [$id] Analysis complete."

    # Phase 2: Evals (4 parallel tasks: 2 scripted + 2 LLM)
    echo "  [$id] Running evals in parallel..."

    echo "    -> Scripted workflow evals"
    ssh "$NODE" "$PREFIX python3 $EVALS_DIR/eval_scripts/workflow_scripted_evals.py \
        --output-dir $OUTPUT_DIR \
        --results $CASE_RESULTS/workflow_scripted_results.csv" &

    echo "    -> LLM workflow evals"
    (
        cd "$EVALS_DIR"
        agent --print --force --trust "Run workflow LLM eval skill on $OUTPUT_DIR for test case $id. Write results to $CASE_RESULTS/workflow_llm_results.csv"
    ) 2>&1 | tee "$CASE_RESULTS/workflow_llm_eval.log" &

    echo "    -> Scripted quality evals"
    ssh "$NODE" "$PREFIX python3 $EVALS_DIR/eval_scripts/quality_scripted_evals.py \
        --output-dir $OUTPUT_DIR --reference-dir $reference_dir \
        --results $CASE_RESULTS/quality_scripted_results.csv" &

    echo "    -> LLM quality evals"
    (
        cd "$EVALS_DIR"
        agent --print --force --trust "Run quality LLM eval skill on $OUTPUT_DIR with reference $reference_dir for test case $id. Write results to $CASE_RESULTS/quality_llm_results.csv"
    ) 2>&1 | tee "$CASE_RESULTS/quality_llm_eval.log" &

    wait
    echo "  [$id] Evals complete."

    # Aggregate Results
    ssh "$NODE" "$PREFIX python3 $EVALS_DIR/eval_scripts/merge_results.py --results-dir $CASE_RESULTS --output $CASE_RESULTS/eval_summary.csv"
    echo "  [$id] Summary written to $CASE_RESULTS/eval_summary.csv"
    echo ""
done < <(tail -n +2 "$TEST_TRACES_CSV"; echo)

echo ""
echo "========================================="
echo "  Eval run finished."
echo "========================================="

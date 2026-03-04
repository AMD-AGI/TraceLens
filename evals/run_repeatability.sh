#!/usr/bin/env bash
set -euo pipefail

# Environment config (edit these for your setup)
NODE=""
CONTAINER=""

# Repeatability config
NUM_REPEATS=5

# Eval directories
REPO_ROOT="$(pwd)"
STANDALONE_DIR="TraceLens/AgenticMode/Standalone"
EVALS_DIR="$REPO_ROOT/evals"
RESULTS_ROOT="$EVALS_DIR/repeatability_results"
TEST_TRACES_CSV="$EVALS_DIR/unit_test_traces.csv"
PREFIX="docker exec -w $REPO_ROOT $CONTAINER"

mkdir -p "$RESULTS_ROOT"
ssh "$NODE" "$PREFIX bash -c 'mkdir -p $RESULTS_ROOT && chmod -R 777 $RESULTS_ROOT'"

echo "========================================="
echo "  Standalone Analysis Repeatability Test"
echo "  Repeats: $NUM_REPEATS"
echo "========================================="
echo ""

while IFS=, read -r id sub_category trace_path reference_dir platform; do

    echo "-----------------------------------------"
    echo "  [$id] Starting $NUM_REPEATS repeat runs"
    echo "-----------------------------------------"

    for ((i = 0; i < NUM_REPEATS; i++)); do

        # Create output directories
        CASE_RESULTS="$RESULTS_ROOT/$id/run_${i}"
        OUTPUT_DIR="$CASE_RESULTS/analysis_output"
        ssh "$NODE" "$PREFIX bash -c 'mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR'"
        ssh "$NODE" "$PREFIX bash -c 'mkdir -p $CASE_RESULTS && chmod -R 777 $CASE_RESULTS'"

        # Phase 1: Agentic Mode Standalone Analysis
        echo "  [$id] Run $((i + 1))/$NUM_REPEATS — output: $OUTPUT_DIR"
        (
            cd "$STANDALONE_DIR"
            agent --print --force --trust "Run standalone analysis on $trace_path with platform $platform, node $NODE, container $CONTAINER, output to $OUTPUT_DIR"
        ) 2>&1 | tee "$CASE_RESULTS/analysis.log"
        echo "  [$id] Run $((i + 1))/$NUM_REPEATS complete."

        # Phase 2: Evals (4 parallel tasks: 2 scripted + 2 LLM)
        echo "  [$id] Running evals for run $((i + 1)) (4 parallel tasks)..."

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

        wait || true
        echo "  [$id] Evals for run $((i + 1)) complete."

        # Aggregate Results
        ssh "$NODE" "$PREFIX python3 $EVALS_DIR/eval_scripts/merge_results.py --results-dir $CASE_RESULTS --output $CASE_RESULTS/eval_summary.csv" || true
        echo "  [$id] Summary written to $CASE_RESULTS/eval_summary.csv"
        echo ""
    done

    echo "  [$id] All $NUM_REPEATS runs finished."
    echo ""

done < <(tail -n +2 "$TEST_TRACES_CSV"; echo)

echo ""
echo "========================================="
echo "  Repeatability test finished."
echo "  Results in: $RESULTS_ROOT"
echo "========================================="
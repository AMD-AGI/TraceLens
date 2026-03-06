#!/usr/bin/env bash
set -uo pipefail

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
ssh -n "$NODE" "$PREFIX bash -c 'mkdir -p $RESULTS_ROOT && chmod -R 777 $RESULTS_ROOT'"

echo "========================================="
echo "  Standalone Analysis Repeatability Test"
echo "  Repeats: $NUM_REPEATS"
echo "========================================="
echo ""

while IFS=, read -r id sub_category trace_path reference_dir platform <&3; do
    [[ -z "$id" ]] && continue

    echo "-----------------------------------------"
    echo "  [$id] Starting $NUM_REPEATS repeat runs"
    echo "-----------------------------------------"

    for ((i = 0; i < NUM_REPEATS; i++)); do
        (
            set -euo pipefail

            # Create output directories
            CASE_RESULTS="$RESULTS_ROOT/$id/run_${i}"
            OUTPUT_DIR="$CASE_RESULTS/analysis_output"
            ssh -n "$NODE" "$PREFIX bash -c 'mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR'"
            ssh -n "$NODE" "$PREFIX bash -c 'mkdir -p $CASE_RESULTS && chmod -R 777 $CASE_RESULTS'"

            # Phase 1: Agentic Mode Standalone Analysis (with retry + backoff)
            echo "  [$id] Run $((i + 1))/$NUM_REPEATS — output: $OUTPUT_DIR"
            agent_attempts=0
            agent_success=false
            while [ "$agent_success" = false ] && [ "$agent_attempts" -lt 3 ]; do
                agent_attempts=$((agent_attempts + 1))
                (
                    cd "$STANDALONE_DIR"
                    agent --print --force --trust --output-format stream-json "Run standalone analysis on $trace_path with platform $platform, node $NODE, container $CONTAINER, output to $OUTPUT_DIR"
                ) < /dev/null 2>&1 | tee "$CASE_RESULTS/analysis_stream.ndjson"

                if head -c 2048 "$CASE_RESULTS/analysis_stream.ndjson" | grep -qiE 'Error:.*unavailable|Service Unavailable'; then
                    echo "  [$id] Attempt $agent_attempts/3 failed (agent unavailable). Backing off 30s..."
                    sleep 30
                else
                    agent_success=true
                fi
            done

            if [ "$agent_success" = false ]; then
                echo "  [$id] Run $((i + 1))/$NUM_REPEATS FAILED after 3 attempts (agent unavailable). Skipping evals."
                continue
            fi
            echo "  [$id] Run $((i + 1))/$NUM_REPEATS complete."

            sleep 30

            # Phase 2: Evals (4 parallel tasks: 2 scripted + 2 LLM)
            echo "  [$id] Running evals for run $((i + 1)) (4 parallel tasks)..."

            echo "    -> Scripted workflow evals"
            ssh -n "$NODE" "$PREFIX python3 $EVALS_DIR/eval_scripts/workflow_scripted_evals.py \
                --output-dir $OUTPUT_DIR \
                --results $CASE_RESULTS/workflow_scripted_results.csv" &

            echo "    -> LLM workflow evals"
            (
                cd "$EVALS_DIR"
                agent --print --force --trust "Run workflow LLM eval skill on $OUTPUT_DIR for test case $id. Write results to $CASE_RESULTS/workflow_llm_results.csv"
            ) < /dev/null 2>&1 | tee "$CASE_RESULTS/workflow_llm_eval.log" &

            echo "    -> Scripted quality evals"
            ssh -n "$NODE" "$PREFIX python3 $EVALS_DIR/eval_scripts/quality_scripted_evals.py \
                --output-dir $OUTPUT_DIR --reference-dir $reference_dir \
                --results $CASE_RESULTS/quality_scripted_results.csv" &

            echo "    -> LLM quality evals"
            (
                cd "$EVALS_DIR"
                agent --print --force --trust "Run quality LLM eval skill on $OUTPUT_DIR with reference $reference_dir for test case $id. Write results to $CASE_RESULTS/quality_llm_results.csv"
            ) < /dev/null 2>&1 | tee "$CASE_RESULTS/quality_llm_eval.log" &

            wait || true
            echo "  [$id] Evals for run $((i + 1)) complete."

            # Aggregate Results
            ssh -n "$NODE" "$PREFIX python3 $EVALS_DIR/eval_scripts/merge_results.py --results-dir $CASE_RESULTS --output $CASE_RESULTS/eval_summary.csv" || true
            echo "  [$id] Summary written to $CASE_RESULTS/eval_summary.csv"
            echo ""
        ) || echo "  [$id] Run $((i + 1))/$NUM_REPEATS FAILED"
    done

    echo "  [$id] All $NUM_REPEATS runs finished."
    echo ""

done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)

echo ""
echo "========================================="
echo "  Repeatability test finished."
echo "  Results in: $RESULTS_ROOT"
echo "========================================="
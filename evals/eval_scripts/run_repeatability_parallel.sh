#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_PARALLEL="${MAX_PARALLEL:-5}"
NUM_REPEATS="${NUM_REPEATS:-5}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
CONTAINER="${CONTAINER:?Set CONTAINER env var (e.g. CONTAINER=my_container)}"
TEST_IDS="${TEST_IDS:-}"

# Paths (run from repo root on the node)
REPO_ROOT="$(pwd)"
STANDALONE_DIR="TraceLens/AgenticMode/Standalone"
EVALS_DIR="$REPO_ROOT/evals"
RESULTS_ROOT="${RESULTS_ROOT:-$EVALS_DIR/repeatability_results}"
TEST_TRACES_CSV="${TEST_TRACES_CSV:-$EVALS_DIR/unit_test_traces.csv}"
DEXEC="docker exec -w $REPO_ROOT $CONTAINER"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() { date "+%H:%M:%S"; }

log_status() {
    flock 1 echo "$@"
}

expand_archive() {
    local name="$1"
    local archive="$EVALS_DIR/${name}.tar.gz"
    local target="$EVALS_DIR/$name"
    if [[ -f "$archive" ]]; then
        if [[ ! -d "$target" ]] || [[ "$archive" -nt "$target" ]]; then
            echo "Expanding ${name}.tar.gz..."
            tar xzf "$archive" -C "$EVALS_DIR"
            echo "Done."
        fi
    fi
}

# ---------------------------------------------------------------------------
# Single job: one (test_case, repeat) iteration
# ---------------------------------------------------------------------------

run_single_job() {
    local id="$1" repeat="$2" trace_path="$3" reference_dir="$4" platform="$5"
    local tag="[$id|run_$repeat]"

    local CASE_RESULTS="$RESULTS_ROOT/$id/run_${repeat}"
    local OUTPUT_DIR="$CASE_RESULTS/analysis_output"
    rm -rf "$CASE_RESULTS"
    mkdir -p "$OUTPUT_DIR"
    $DEXEC bash -c "mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR"
    $DEXEC bash -c "mkdir -p $CASE_RESULTS && chmod -R 777 $CASE_RESULTS"

    # -- Phase 1: Agent analysis with retry + backoff -----------------------
    log_status "  $tag [$(ts)] Phase 1: analysis starting"

    local agent_success=false
    local agent_attempts=0
    while [ "$agent_success" = false ] && [ "$agent_attempts" -lt 3 ]; do
        agent_attempts=$((agent_attempts + 1))
        (
            cd "$STANDALONE_DIR"
            timeout 1200 agent --print --force --trust --output-format stream-json \
                "Run standalone analysis on $trace_path with platform $platform, node $(hostname), container $CONTAINER, output to $OUTPUT_DIR"
        ) < /dev/null > "$CASE_RESULTS/analysis_stream.ndjson" 2>&1

        if head -c 2048 "$CASE_RESULTS/analysis_stream.ndjson" | grep -qiE 'Error:.*unavailable|Service Unavailable'; then
            log_status "  $tag Attempt $agent_attempts/3 failed (agent unavailable). Backing off 30s..."
            sleep 30
        else
            agent_success=true
        fi
    done

    if [ "$agent_success" = false ]; then
        log_status "  $tag FAILED after 3 attempts (agent unavailable). Skipping evals."
        return 1
    fi

    log_status "  $tag [$(ts)] Phase 1 complete."
    sleep "$SLEEP_BETWEEN"

    # -- Phase 2: 4 parallel evals ------------------------------------------
    log_status "  $tag [$(ts)] Phase 2: evals starting"
    local eval_pids=()

    $DEXEC python3 "$EVALS_DIR/eval_utils/workflow_scripted_evals.py" \
        --output-dir "$OUTPUT_DIR" \
        --results "$CASE_RESULTS/workflow_scripted_results.csv" \
        > "$CASE_RESULTS/workflow_scripted_eval.log" 2>&1 &
    eval_pids+=($!)

    (
        cd "$EVALS_DIR"
        agent --print --force --trust --output-format stream-json \
            "Run workflow LLM eval skill on $OUTPUT_DIR for test case $id. Write results to $CASE_RESULTS/workflow_llm_results.csv"
    ) < /dev/null > "$CASE_RESULTS/workflow_llm_eval.ndjson" 2>&1 &
    eval_pids+=($!)

    $DEXEC python3 "$EVALS_DIR/eval_utils/quality_scripted_evals.py" \
        --output-dir "$OUTPUT_DIR" --reference-dir "$reference_dir" \
        --results "$CASE_RESULTS/quality_scripted_results.csv" \
        > "$CASE_RESULTS/quality_scripted_eval.log" 2>&1 &
    eval_pids+=($!)

    (
        cd "$EVALS_DIR"
        agent --print --force --trust --output-format stream-json \
            "Run quality LLM eval skill on $OUTPUT_DIR with reference $reference_dir for test case $id. Write results to $CASE_RESULTS/quality_llm_results.csv"
    ) < /dev/null > "$CASE_RESULTS/quality_llm_eval.ndjson" 2>&1 &
    eval_pids+=($!)

    for pid in "${eval_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    log_status "  $tag [$(ts)] Phase 2 complete."

    # -- Merge results -------------------------------------------------------
    $DEXEC python3 "$EVALS_DIR/eval_utils/merge_results.py" \
        --results-dir "$CASE_RESULTS" \
        --output "$CASE_RESULTS/eval_summary.csv" || true
    log_status "  $tag Summary -> $CASE_RESULTS/eval_summary.csv"
}

# ---------------------------------------------------------------------------
# FIFO semaphore for concurrency control
# ---------------------------------------------------------------------------

FIFO="$RESULTS_ROOT/.job_fifo"
cleanup() {
    rm -f "$FIFO"
}

setup_semaphore() {
    rm -f "$FIFO"
    mkfifo "$FIFO"
    exec 4<>"$FIFO"
    for ((t = 0; t < MAX_PARALLEL; t++)); do echo >&4; done
    trap cleanup EXIT
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

mkdir -p "$RESULTS_ROOT"
$DEXEC bash -c "mkdir -p $RESULTS_ROOT && chmod -R 777 $RESULTS_ROOT"
expand_archive unit_tests
expand_archive e2e_tests

echo "========================================="
echo "  Standalone Analysis Repeatability Test"
echo "  Node:         $(hostname)"
echo "  Container:    $CONTAINER"
echo "  Repeats:      $NUM_REPEATS"
echo "  Max parallel: $MAX_PARALLEL"
if [[ -n "$TEST_IDS" ]]; then
    echo "  Test filter:  $TEST_IDS"
fi
echo "========================================="
echo ""

setup_semaphore

while IFS=, read -r id sub_category trace_path reference_dir platform <&3; do
    [[ -z "$id" ]] && continue
    if [[ -n "$TEST_IDS" ]]; then
        case " $TEST_IDS " in
            *" $id "*) ;;
            *) continue ;;
        esac
    fi

    for ((i = 0; i < NUM_REPEATS; i++)); do
        read -u4  # acquire semaphore slot
        (
            run_single_job "$id" "$i" "$trace_path" "$reference_dir" "$platform" || true
            echo >&4  # release semaphore slot
        ) &
    done
done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)

wait

echo ""
echo "========================================="
echo "  Repeatability test finished."
echo "  Results in: $RESULTS_ROOT"
echo "========================================="

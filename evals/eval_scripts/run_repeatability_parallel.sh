#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------------------
# Usage: bash run_repeatability_parallel.sh [standalone|comparative]
#
# MODE can also be set via the MODE environment variable.
# Defaults to standalone.
#
# CONTAINER is optional. If set, python/setup commands run via
# docker exec -w $REPO_ROOT $CONTAINER ... ; if unset, they run on the host.
# ---------------------------------------------------------------------------
MODE="${1:-${MODE:-standalone}}"

if [[ "$MODE" != "standalone" && "$MODE" != "comparative" ]]; then
    echo "ERROR: Unknown mode '$MODE'. Use 'standalone' or 'comparative'." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_PARALLEL="${MAX_PARALLEL:-5}"
NUM_REPEATS="${NUM_REPEATS:-5}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
CONTAINER="${CONTAINER:-}"
TEST_IDS="${TEST_IDS:-}"
SUITE_NAME="${SUITE_NAME:-eval}"
SKIP_POST_PROCESSING="${SKIP_POST_PROCESSING:-}"

# Paths (run from repo root on the node)
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ANALYSIS_DIR="TraceLens/Agent/Analysis"
EVALS_DIR="$REPO_ROOT/evals"

if [[ -n "$CONTAINER" ]]; then
    DEXEC=(docker exec -w "$REPO_ROOT" "$CONTAINER")
    RUNTIME_LABEL="container $CONTAINER"
    NODE_LABEL="node $(hostname)"
else
    DEXEC=()
    RUNTIME_LABEL="host (no container)"
    NODE_LABEL="local"
fi

TEST_TRACES_CSV="${TEST_TRACES_CSV:-$EVALS_DIR/analysis_tests/combined_traces_${MODE}.csv}"
RESULTS_ROOT="${RESULTS_ROOT:-$EVALS_DIR/repeatability_results_${MODE}}"
REPORT_DIR="${REPORT_DIR:-$RESULTS_ROOT/../reports_${MODE}}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() { date "+%H:%M:%S"; }

log_status() {
    flock 1 echo "$@"
}

expand_archive() {
    local name="$1"
    local archive="$EVALS_DIR/analysis_tests/${name}.tar.gz"
    local target="$EVALS_DIR/analysis_tests/$name"
    if [[ -f "$archive" ]]; then
        if [[ ! -d "$target" ]] || [[ "$archive" -nt "$target" ]]; then
            echo "Expanding ${name}.tar.gz..."
            tar xzf "$archive" -C "$EVALS_DIR/analysis_tests"
            echo "Done."
        fi
    fi
}

# ---------------------------------------------------------------------------
# Single job: one (test_case, repeat) iteration
#
# Args: id repeat trace1_path trace2_path reference_dir platform (comparative mode only)
# ---------------------------------------------------------------------------

run_single_job() {
    local id="$1" repeat="$2" trace1_path="$3" trace2_path="$4" reference_dir="$5" platform="$6"
    local tag="[$id|run_$repeat]"

    local CASE_RESULTS="$RESULTS_ROOT/$id/run_${repeat}"
    local OUTPUT_DIR="$CASE_RESULTS/analysis_output"
    "${DEXEC[@]}" rm -rf "$CASE_RESULTS" 2>/dev/null || true
    rm -rf "$CASE_RESULTS" 2>/dev/null || true
    mkdir -p "$OUTPUT_DIR"
    "${DEXEC[@]}" bash -c "mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR"
    "${DEXEC[@]}" bash -c "mkdir -p $CASE_RESULTS && chmod -R 777 $CASE_RESULTS"

    # -- Phase 1: Agent analysis with retry + backoff -----------------------
    log_status "  $tag [$(ts)] Phase 1: analysis starting"

    local agent_success=false
    local agent_attempts=0
    while [ "$agent_success" = false ] && [ "$agent_attempts" -lt 3 ]; do
        agent_attempts=$((agent_attempts + 1))
        (
            if [[ "$MODE" == "comparative" ]]; then
                timeout 1800 agent --model claude-opus-4-7-high --print --force --trust --output-format stream-json \
                    "Follow the Analysis Orchestrator installed with TraceLens and run the full agentic analysis workflow on $trace1_path and $trace2_path with platform $platform (baseline is trace1), $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR"
            else
                timeout 1800 agent --model claude-opus-4-7-high --print --force --trust --output-format stream-json \
                    "Follow the Analysis Orchestrator installed with TraceLens and run the full agentic analysis workflow on $trace1_path with platform $platform, $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR"
            fi
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

    "${DEXEC[@]}" python3 "$EVALS_DIR/eval_utils/workflow_scripted_evals.py" \
        --output-dir "$OUTPUT_DIR" \
        --results "$CASE_RESULTS/workflow_scripted_results.csv" \
        --comparison-scope "$MODE" \
        > "$CASE_RESULTS/workflow_scripted_eval.log" 2>&1 &
    eval_pids+=($!)

    (
        cd "$EVALS_DIR" || exit
        agent --model claude-opus-4-7-high --print --force --trust --output-format stream-json \
            "Run workflow LLM eval skill on $OUTPUT_DIR for test case $id mode=$MODE. Write results to $CASE_RESULTS/workflow_llm_results.csv"
    ) < /dev/null > "$CASE_RESULTS/workflow_llm_eval.ndjson" 2>&1 &
    eval_pids+=($!)

    "${DEXEC[@]}" python3 "$EVALS_DIR/eval_utils/quality_scripted_evals.py" \
        --output-dir "$OUTPUT_DIR" --reference-dir "$reference_dir" \
        --results "$CASE_RESULTS/quality_scripted_results.csv" \
        --comparison-scope "$MODE" \
        > "$CASE_RESULTS/quality_scripted_eval.log" 2>&1 &
    eval_pids+=($!)

    (
        cd "$EVALS_DIR" || exit
        agent --model claude-opus-4-7-high --print --force --trust --output-format stream-json \
            "Run quality LLM eval skill on $OUTPUT_DIR with reference $reference_dir for test case $id mode=$MODE. Write results to $CASE_RESULTS/quality_llm_results.csv"
    ) < /dev/null > "$CASE_RESULTS/quality_llm_eval.ndjson" 2>&1 &
    eval_pids+=($!)

    for pid in "${eval_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    log_status "  $tag [$(ts)] Phase 2 complete."

    # -- Merge results -------------------------------------------------------
    "${DEXEC[@]}" python3 "$EVALS_DIR/eval_utils/merge_results.py" \
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
"${DEXEC[@]}" bash -c "mkdir -p $RESULTS_ROOT && chmod -R 777 $RESULTS_ROOT"
if [[ "$MODE" == "comparative" ]]; then
    expand_archive unit_tests_comparative
    expand_archive e2e_tests_comparative
else
    expand_archive unit_tests
    expand_archive e2e_tests
fi

echo "========================================="
echo "  Analysis Repeatability Test"
echo "  Mode:         $MODE"
echo "  Node:         $NODE_LABEL"
echo "  Runtime:      $RUNTIME_LABEL"
echo "  Repeats:      $NUM_REPEATS"
echo "  Max parallel: $MAX_PARALLEL"
echo "  CSV:          $TEST_TRACES_CSV"
if [[ -n "$TEST_IDS" ]]; then
    echo "  Test filter:  $TEST_IDS"
fi
echo "========================================="
echo ""

_spawn_jobs() {
    local id="$1" trace1_path="$2" trace2_path="$3" reference_dir="$4" platform="$5"

    if [[ -n "$TEST_IDS" ]]; then
        case " $TEST_IDS " in
            *" $id "*) ;;
            *) return ;;
        esac
    fi

    for ((i = 0; i < NUM_REPEATS; i++)); do
        read -r -u4  # acquire semaphore slot
        (
            run_single_job "$id" "$i" "$trace1_path" "$trace2_path" "$reference_dir" "$platform" || true
            echo >&4  # release semaphore slot
            sleep 2  # stagger agent startup to avoid ~/.cursor/cli-config.json rename race
        ) &
        sleep 2  # stagger agent startup to avoid ~/.cursor/cli-config.json rename race
    done
}

setup_semaphore

if [[ "$MODE" == "comparative" ]]; then
    # comparative CSV: id,sub_category,trace1_path,trace2_path,reference_dir,platform
    while IFS=, read -r id sub_category trace1_path trace2_path reference_dir platform <&3; do
        [[ -z "$id" ]] && continue
        _spawn_jobs "$id" "$trace1_path" "$trace2_path" "$reference_dir" "$platform"
    done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
else
    # standalone CSV: id,sub_category,trace_path,reference_dir,platform
    while IFS=, read -r id sub_category trace_path reference_dir platform <&3; do
        [[ -z "$id" ]] && continue
        _spawn_jobs "$id" "$trace_path" "" "$reference_dir" "$platform"
    done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
fi

wait

echo ""
echo "========================================="
echo "  Repeatability test finished."
echo "  Results in: $RESULTS_ROOT"
echo "========================================="

# ---------------------------------------------------------------------------
# Post-processing: aggregate results and generate reports via Cursor agent
# ---------------------------------------------------------------------------

if [[ "$SKIP_POST_PROCESSING" == "1" ]]; then
    echo ""
    echo "  Post-processing skipped -- SKIP_POST_PROCESSING=1."
    echo "  To run later: agent 'Run eval post processing on results_root=$RESULTS_ROOT suite=$SUITE_NAME test_traces_csv=$TEST_TRACES_CSV report_dir=$REPORT_DIR container=${CONTAINER:-} $NODE_LABEL $RUNTIME_LABEL'"
else
    mkdir -p "$REPORT_DIR"

    echo ""
    echo "========================================="
    echo "  Running eval post-processing..."
    echo "========================================="

    (
        cd "$EVALS_DIR" || exit
        agent --model claude-opus-4-7-high --print --force --trust --output-format stream-json \
            "Run eval post processing on results_root=$RESULTS_ROOT suite=$SUITE_NAME test_traces_csv=$TEST_TRACES_CSV report_dir=$REPORT_DIR container=${CONTAINER:-} $NODE_LABEL $RUNTIME_LABEL"
    ) < /dev/null > "$REPORT_DIR/post_processing.ndjson" 2>&1

    echo "  Post-processing complete. Reports in: $REPORT_DIR"
fi

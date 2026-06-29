#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

set -uo pipefail

# ---------------------------------------------------------------------------
# Usage: bash run_repeatability_parallel.sh [standalone|comparative] [--pi]
#    or: bash run_repeatability_parallel.sh --pi [standalone|comparative]
#
# COMPARISON_SCOPE can also be set via the COMPARISON_SCOPE environment variable.
# Defaults to standalone.
#
# --pi  Use `pi` instead of the Cursor `agent` CLI. Also settable via USE_PI=1.
#
# CONTAINER is optional. If set, python/setup commands run via
# docker exec -w $REPO_ROOT $CONTAINER ... ; if unset, they run on the host.
# ---------------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: bash run_repeatability_parallel.sh [standalone|comparative] [--pi]

  standalone|comparative   Comparison scope (default: standalone or COMPARISON_SCOPE)
  --pi                     Use pi instead of the Cursor agent CLI (or USE_PI=1)
EOF
}

USE_PI="${USE_PI:-false}"
COMPARISON_SCOPE="${COMPARISON_SCOPE:-standalone}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pi)
            USE_PI=true
            shift
            ;;
        standalone|comparative)
            COMPARISON_SCOPE="$1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument '$1'." >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ "$USE_PI" == true || "$USE_PI" == 1 || "$USE_PI" == "1" ]]; then
    USE_PI=true
else
    USE_PI=false
fi

if [[ "$COMPARISON_SCOPE" != "standalone" && "$COMPARISON_SCOPE" != "comparative" ]]; then
    echo "ERROR: Unknown comparison scope '$COMPARISON_SCOPE'. Use 'standalone' or 'comparative'." >&2
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

AGENT_MODEL="${AGENT_MODEL:-claude-4.6-opus-high}"
PI_VENV_PREFIX="use venv_tracelens for all commands and tool calls. "

# Paths (REPO_ROOT may differ from the shell cwd)
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ANALYSIS_DIR="$REPO_ROOT/TraceLens/Agent/Analysis"
EVALS_DIR="$REPO_ROOT/agent_evals/Analysis"
RESULTS_ROOT="${RESULTS_ROOT:-$EVALS_DIR/repeatability_results_${COMPARISON_SCOPE}}"
TEST_TRACES_CSV="${TEST_TRACES_CSV:-$EVALS_DIR/analysis_tests/combined_traces_${COMPARISON_SCOPE}.csv}"

REPORT_DIR="${REPORT_DIR:-$RESULTS_ROOT/../reports_${COMPARISON_SCOPE}}"

if [[ -n "$CONTAINER" ]]; then
    DEXEC=(docker exec -w "$REPO_ROOT" "$CONTAINER")
    RUNTIME_LABEL="container $CONTAINER"
    NODE_LABEL="node $(hostname)"
else
    DEXEC=()
    RUNTIME_LABEL="host (no container)"
    NODE_LABEL="local"
fi

if [[ "$USE_PI" == true ]]; then
    AGENT_BACKEND="pi"
else
    AGENT_BACKEND="cursor agent"
fi

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() { date "+%H:%M:%S"; }

log_status() {
    flock 1 echo "$@"
}

repo_abs_path() {
    local p="$1"
    if [[ "$p" = /* ]]; then
        echo "$p"
    else
        echo "$REPO_ROOT/$p"
    fi
}

# Run an LLM agent step. Optional second arg: non-empty enables a 1800s timeout (cursor only).
run_llm_agent() {
    local prompt="$1"
    local with_timeout="${2:-}"

    if [[ "$USE_PI" == true ]]; then
        pi --mode json "${PI_VENV_PREFIX}${prompt}"
    elif [[ -n "$with_timeout" ]]; then
        timeout 1800 agent --model "$AGENT_MODEL" --print --force --trust --output-format stream-json \
            "$prompt"
    else
        agent --model "$AGENT_MODEL" --print --force --trust --output-format stream-json \
            "$prompt"
    fi
}

expand_archive() {
    local name="$1"
    local archive="$EVALS_DIR/analysis_tests/${name}.tar.gz"
    local target="$EVALS_DIR/analysis_tests/$name"
    if [[ -f "$archive" ]]; then
        if [[ ! -d "$target" ]] || [[ "$archive" -nt "$target" ]]; then
            echo "Expanding ${name}.tar.gz..."
            tar xzf "$archive" -C "$REPO_ROOT"
            echo "Done."
        fi
    fi
}

# Return 0 if $id should run given TEST_IDS (empty = all).
# Supports exact match and underscore-delimited prefix (e.g. gemm_01 -> gemm_01_compute_few_tiles).
should_run_id() {
    local id="$1"
    [[ -z "$TEST_IDS" ]] && return 0
    local token
    for token in $TEST_IDS; do
        if [[ "$id" == "$token" || "$id" == "${token}_"* ]]; then
            return 0
        fi
    done
    return 1
}

print_scheduled_tests() {
    local -a scheduled_ids=()
    local id sub_category trace1_path trace2_path trace_path reference_dir platform platform2

    if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
        while IFS=, read -r id sub_category trace1_path trace2_path reference_dir platform platform2; do
            [[ -z "$id" ]] && continue
            should_run_id "$id" && scheduled_ids+=("$id")
        done < <(tail -n +2 "$TEST_TRACES_CSV")
    else
        while IFS=, read -r id sub_category trace_path reference_dir platform; do
            [[ -z "$id" ]] && continue
            should_run_id "$id" && scheduled_ids+=("$id")
        done < <(tail -n +2 "$TEST_TRACES_CSV")
    fi

    if [[ ${#scheduled_ids[@]} -eq 0 ]]; then
        return
    fi

    echo "Tests to run (${#scheduled_ids[@]}):"
    local scheduled_id
    for scheduled_id in "${scheduled_ids[@]}"; do
        echo "  - $scheduled_id ($NUM_REPEATS repeat(s))"
    done
    echo ""
}

# ---------------------------------------------------------------------------
# Single job: one (test_case, repeat) iteration
#
# Args: id repeat trace1_path trace2_path reference_dir platform platform2 (comparative mode only)
# ---------------------------------------------------------------------------

run_single_job() {
    local id="$1" repeat="$2" trace1_path="$3" trace2_path="$4" reference_dir="$5" platform="$6" platform2="$7"
    local tag="[$id|run_$repeat]"

    log_status "  $tag [$(ts)] Running"

    trace1_path="$(repo_abs_path "$trace1_path")"
    if [[ -n "$trace2_path" ]]; then
        trace2_path="$(repo_abs_path "$trace2_path")"
    fi
    reference_dir="$(repo_abs_path "$reference_dir")"

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
            cd "$ANALYSIS_DIR" || exit
            if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
                run_llm_agent \
                    "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/.cursor/skills/ in the package installation directory) and run the full agentic analysis workflow on $trace1_path and $trace2_path with platform $platform (trace1) and $platform2 (trace2), analysis mode default, $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR" \
                    1
            else
                run_llm_agent \
                    "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/.cursor/skills/ in the package installation directory) and run the full agentic analysis workflow on $trace1_path with platform $platform, $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR" \
                    1
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
        --comparison-scope "$COMPARISON_SCOPE" \
        > "$CASE_RESULTS/workflow_scripted_eval.log" 2>&1 &
    eval_pids+=($!)

    (
        cd "$EVALS_DIR" || exit
        run_llm_agent \
            "Run workflow LLM eval skill on $OUTPUT_DIR for test case $id mode=$COMPARISON_SCOPE. Write results to $CASE_RESULTS/workflow_llm_results.csv"
    ) < /dev/null > "$CASE_RESULTS/workflow_llm_eval.ndjson" 2>&1 &
    eval_pids+=($!)

    "${DEXEC[@]}" python3 "$EVALS_DIR/eval_utils/quality_scripted_evals.py" \
        --output-dir "$OUTPUT_DIR" --reference-dir "$reference_dir" \
        --results "$CASE_RESULTS/quality_scripted_results.csv" \
        --comparison-scope "$COMPARISON_SCOPE" \
        > "$CASE_RESULTS/quality_scripted_eval.log" 2>&1 &
    eval_pids+=($!)

    (
        cd "$EVALS_DIR" || exit
        run_llm_agent \
            "Run quality LLM eval skill on $OUTPUT_DIR with reference $reference_dir for test case $id mode=$COMPARISON_SCOPE. Write results to $CASE_RESULTS/quality_llm_results.csv"
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
    log_status "  $tag [$(ts)] Finished"
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
if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
    expand_archive unit_tests_comparative
    expand_archive e2e_tests_comparative
else
    expand_archive unit_tests_standalone
    expand_archive e2e_tests_standalone
fi

echo "========================================="
echo "  Analysis Repeatability Test"
echo "  Mode:         $COMPARISON_SCOPE"
echo "  Agent:        $AGENT_BACKEND"
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

print_scheduled_tests

_spawn_jobs() {
    local id="$1" trace1_path="$2" trace2_path="$3" reference_dir="$4" platform="$5" platform2="$6"

    should_run_id "$id" || return
    JOBS_SPAWNED=$((JOBS_SPAWNED + 1))

    for ((i = 0; i < NUM_REPEATS; i++)); do
        read -r -u4  # acquire semaphore slot
        (
            run_single_job "$id" "$i" "$trace1_path" "$trace2_path" "$reference_dir" "$platform" "$platform2" || true
            echo >&4  # release semaphore slot
            sleep 2  # stagger agent startup to avoid ~/.cursor/cli-config.json rename race
        ) &
        sleep 2  # stagger agent startup to avoid ~/.cursor/cli-config.json rename race
    done
}

setup_semaphore

JOBS_SPAWNED=0

if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
    # comparative CSV: id,sub_category,trace1_path,trace2_path,reference_dir,platform,platform2
    while IFS=, read -r id sub_category trace1_path trace2_path reference_dir platform platform2 <&3; do
        [[ -z "$id" ]] && continue
        _spawn_jobs "$id" "$trace1_path" "$trace2_path" "$reference_dir" "$platform" "$platform2"
    done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
else
    # standalone CSV: id,sub_category,trace_path,reference_dir,platform
    while IFS=, read -r id sub_category trace_path reference_dir platform <&3; do
        [[ -z "$id" ]] && continue
        _spawn_jobs "$id" "$trace_path" "" "$reference_dir" "$platform" ""
    done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
fi

wait

if [[ -n "$TEST_IDS" && "$JOBS_SPAWNED" -eq 0 ]]; then
    echo ""
    echo "WARNING: TEST_IDS='$TEST_IDS' matched no trace ids in $TEST_TRACES_CSV." >&2
    echo "  Use exact ids or underscore-delimited prefixes (e.g. gemm_01 -> gemm_01_compute_few_tiles)." >&2
    echo "  No eval jobs were started; reports will be empty." >&2
fi

echo ""
echo "========================================="
echo "  Repeatability test finished."
echo "  Results in: $RESULTS_ROOT"
echo "========================================="

# ---------------------------------------------------------------------------
# Post-processing: aggregate results and generate reports via LLM agent
# ---------------------------------------------------------------------------

if [[ "$SKIP_POST_PROCESSING" == "1" ]]; then
    echo ""
    echo "  Post-processing skipped -- SKIP_POST_PROCESSING=1."
    if [[ "$USE_PI" == true ]]; then
        echo "  To run later: pi --mode json '${PI_VENV_PREFIX}Run eval post processing on results_root=$RESULTS_ROOT suite=$SUITE_NAME test_traces_csv=$TEST_TRACES_CSV report_dir=$REPORT_DIR container=${CONTAINER:-} $NODE_LABEL $RUNTIME_LABEL'"
    else
        echo "  To run later: agent 'Run eval post processing on results_root=$RESULTS_ROOT suite=$SUITE_NAME test_traces_csv=$TEST_TRACES_CSV report_dir=$REPORT_DIR container=${CONTAINER:-} $NODE_LABEL $RUNTIME_LABEL'"
    fi
else
    mkdir -p "$REPORT_DIR"

    echo ""
    echo "========================================="
    echo "  Running eval post-processing..."
    echo "========================================="

    (
        cd "$EVALS_DIR" || exit
        run_llm_agent \
            "Run eval post processing on results_root=$RESULTS_ROOT suite=$SUITE_NAME test_traces_csv=$TEST_TRACES_CSV report_dir=$REPORT_DIR container=${CONTAINER:-} $NODE_LABEL $RUNTIME_LABEL"
    ) < /dev/null > "$REPORT_DIR/post_processing.ndjson" 2>&1

    echo "  Post-processing complete. Reports in: $REPORT_DIR"
fi

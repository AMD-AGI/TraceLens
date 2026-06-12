#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------------------
# Usage: bash generate_ref.sh [standalone|comparative]
#
# COMPARISON_SCOPE can also be set via the COMPARISON_SCOPE environment variable.
# Defaults to standalone.
# ---------------------------------------------------------------------------
COMPARISON_SCOPE="${1:-${COMPARISON_SCOPE:-standalone}}"

if [[ "$COMPARISON_SCOPE" != "standalone" && "$COMPARISON_SCOPE" != "comparative" ]]; then
    echo "ERROR: Unknown comparison scope '$COMPARISON_SCOPE'. Use 'standalone' or 'comparative'." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuration (run from repo root on the node)
# ---------------------------------------------------------------------------
CONTAINER="${CONTAINER:-}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
TEST_IDS="${TEST_IDS:-}"

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ANALYSIS_DIR="TraceLens/Agent/Analysis"
EVALS_DIR="$REPO_ROOT/agent_evals/Analysis"
TEST_TRACES_CSV="${TEST_TRACES_CSV:-$EVALS_DIR/analysis_tests/combined_traces_${COMPARISON_SCOPE}.csv}"
STATUS_FILE="$(mktemp)"

if [[ -n "$CONTAINER" ]]; then
    DEXEC=(docker exec -w "$REPO_ROOT" "$CONTAINER")
    RUNTIME_LABEL="container $CONTAINER"
    NODE_LABEL="node $(hostname)"
else
    DEXEC=()
    RUNTIME_LABEL="host (no container)"
    NODE_LABEL="local"
fi

# ---------------------------------------------------------------------------
# Auto-extract test archives if trace CSV references them
# ---------------------------------------------------------------------------
for archive in "$EVALS_DIR"/analysis_tests/e2e_tests_${COMPARISON_SCOPE}.tar.gz "$EVALS_DIR"/analysis_tests/unit_tests_${COMPARISON_SCOPE}.tar.gz; do
    [ -f "$archive" ] || continue
    target_dir="${archive%.tar.gz}"
    if [ ! -d "$target_dir" ]; then
        echo "Extracting $(basename "$archive")..."
        tar -xzf "$archive" -C "$REPO_ROOT"
    fi
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() { date "+%H:%M:%S"; }

log_status() {
    flock 1 echo "$@"
}

# ---------------------------------------------------------------------------
# Single job: generate one golden reference
#
# Args: id trace1_path trace2_path reference_dir platform
# For standalone, trace2_path is empty string "".
# ---------------------------------------------------------------------------

generate_single_ref() {
    local id="$1" trace1_path="$2" trace2_path="$3" reference_dir="$4" platform="$5" platform2="$6"
    local tag="[$id]"

    local REF_DIR="$REPO_ROOT/$reference_dir"
    local CASE_DIR
    CASE_DIR="$(dirname "$REF_DIR")"
    local OUTPUT_DIR="$CASE_DIR/analysis_output"

    # Verify trace file(s) exist
    if [ ! -f "$REPO_ROOT/$trace1_path" ]; then
        log_status "  $tag ERROR: Trace file not found: $trace1_path — skipping."
        flock "$STATUS_FILE" bash -c "echo 'failed' >> '$STATUS_FILE'"
        return 1
    fi
    if [[ "$COMPARISON_SCOPE" == "comparative" ]] && [ ! -f "$REPO_ROOT/$trace2_path" ]; then
        log_status "  $tag ERROR: Trace2 file not found: $trace2_path — skipping."
        flock "$STATUS_FILE" bash -c "echo 'failed' >> '$STATUS_FILE'"
        return 1
    fi

    log_status "  $tag [$(ts)] Generating golden reference..."
    "${DEXEC[@]}" bash -c "mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR"

    # Run analysis with retry + backoff
    local agent_success=false
    local agent_attempts=0
    while [ "$agent_success" = false ] && [ "$agent_attempts" -lt 3 ]; do
        agent_attempts=$((agent_attempts + 1))
        (
            cd "$ANALYSIS_DIR" || exit
            if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
                agent --model claude-4.6-opus-high --print --force --trust --output-format stream-json \
                    "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/.cursor/skills/ in the package installation directory) and run the full agentic analysis workflow on $trace1_path and $trace2_path with platform $platform (trace1) and $platform2 (trace2), analysis mode default, $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR"
            else
                agent --model claude-4.6-opus-high --print --force --trust --output-format stream-json \
                    "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/.cursor/skills/ in the package installation directory) and run the full agentic analysis workflow on $trace1_path with platform $platform, analysis mode default, $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR"
            fi
        ) < /dev/null > "$CASE_DIR/analysis_stream.ndjson" 2>&1

        if grep -qiE 'Error:.*unavailable|Service Unavailable|usage limit|out of usage|You'\''ve reached your' "$CASE_DIR/analysis_stream.ndjson" 2>/dev/null; then
            log_status "  $tag Attempt $agent_attempts/3 failed (agent unavailable or usage limit). Backing off 30s..."
            sleep 30
        else
            agent_success=true
        fi
    done

    if [ "$agent_success" = false ]; then
        log_status "  $tag FAILED after 3 attempts (agent unavailable or usage limit)."
        "${DEXEC[@]}" rm -rf "$OUTPUT_DIR" 2>/dev/null || rm -rf "$OUTPUT_DIR"
        rm -f "$CASE_DIR/analysis_stream.ndjson"
        flock "$STATUS_FILE" bash -c "echo 'failed' >> '$STATUS_FILE'"
        return 1
    fi

    # Verify output was generated
    if [ ! -f "$OUTPUT_DIR/analysis.md" ]; then
        log_status "  $tag WARNING: analysis.md not found in output (agent may have exited without running analysis)."
        "${DEXEC[@]}" rm -rf "$OUTPUT_DIR" 2>/dev/null || rm -rf "$OUTPUT_DIR"
        rm -f "$CASE_DIR/analysis_stream.ndjson"
        flock "$STATUS_FILE" bash -c "echo 'failed' >> '$STATUS_FILE'"
        return 1
    fi

    # Copy output as reference (remove old ref first, then copy contents directly)
    rm -rf "$REF_DIR"
    cp -r "$OUTPUT_DIR" "$REF_DIR"

    # Remove unwanted files from reference dir (keep only analysis.md + perf_report_csvs/)
    rm -rf "$REF_DIR/category_data" \
           "$REF_DIR/category_findings" \
           "$REF_DIR/system_findings" \
           "$REF_DIR/metadata" \
           "$REF_DIR/cache" \
           "$REF_DIR/perf_improvement.png" \
           "$REF_DIR/perf_improvement_base64.txt" \
           "$REF_DIR/plot_data.json" \
           "$REF_DIR/perf_report.xlsx" \
           "$REF_DIR/priority_data.json"

    # Remove intermediate analysis output (docker-owned files need container cleanup)
    "${DEXEC[@]}" rm -rf "$OUTPUT_DIR"
    rm -f "$CASE_DIR/analysis_stream.ndjson"

    log_status "  $tag [$(ts)] Reference saved to $reference_dir (cleaned)"
    flock "$STATUS_FILE" bash -c "echo 'generated' >> '$STATUS_FILE'"
}

# ---------------------------------------------------------------------------
# FIFO semaphore for concurrency control
# ---------------------------------------------------------------------------

FIFO="$(mktemp -u)"
cleanup() {
    rm -f "$FIFO" "$STATUS_FILE"
}

setup_semaphore() {
    mkfifo "$FIFO"
    exec 4<>"$FIFO"
    for ((t = 0; t < MAX_PARALLEL; t++)); do echo >&4; done
    trap cleanup EXIT
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "========================================="
echo "  Golden Reference Generation"
echo "  Mode:         $COMPARISON_SCOPE"
echo "  Node:         $NODE_LABEL"
echo "  Runtime:      $RUNTIME_LABEL"
echo "  Max parallel: $MAX_PARALLEL"
echo "  CSV:          $TEST_TRACES_CSV"
if [[ -n "$TEST_IDS" ]]; then
    echo "  Test filter:  $TEST_IDS"
fi
echo "========================================="
echo ""

should_run_id() {
    local id="$1"
    [[ -z "$TEST_IDS" ]] && return 0
    case " $TEST_IDS " in
        *" $id "*) return 0 ;;
        *) return 1 ;;
    esac
}

setup_semaphore

if [[ "$COMPARISON_SCOPE" == "comparative" ]]; then
    # comparative CSV: id,sub_category,trace1_path,trace2_path,reference_dir,platform,platform2
    while IFS=, read -r id sub_category trace1_path trace2_path reference_dir platform platform2 <&3; do
        [[ -z "$id" ]] && continue
        should_run_id "$id" || continue

        read -u4  # acquire semaphore slot
        (
            generate_single_ref "$id" "$trace1_path" "$trace2_path" "$reference_dir" "$platform" "$platform2" || true
            sleep "$SLEEP_BETWEEN"
            echo >&4  # release semaphore slot
        ) &
    done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
else
    # standalone CSV: id,sub_category,trace_path,reference_dir,platform
    while IFS=, read -r id sub_category trace_path reference_dir platform <&3; do
        [[ -z "$id" ]] && continue
        should_run_id "$id" || continue

        read -u4  # acquire semaphore slot
        (
            generate_single_ref "$id" "$trace_path" "" "$reference_dir" "$platform" "" || true
            sleep "$SLEEP_BETWEEN"
            echo >&4  # release semaphore slot
        ) &
    done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
fi

wait

# Tally results from status file
generated="$(grep -c '^generated$' "$STATUS_FILE" 2>/dev/null || true)"
failed="$(grep -c '^failed$' "$STATUS_FILE" 2>/dev/null || true)"
generated="${generated:-0}"
failed="${failed:-0}"
total=$(( generated + failed ))

echo ""
echo "========================================="
echo "  Golden Reference Generation Complete"
echo "  Total: $total | Generated: $generated | Failed: $failed"
echo "========================================="

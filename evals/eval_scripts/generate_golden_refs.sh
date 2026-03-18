#!/usr/bin/env bash
set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration (run from repo root on the node)
# ---------------------------------------------------------------------------
CONTAINER="${CONTAINER:?Set CONTAINER env var (e.g. CONTAINER=my_container)}"
MAX_PARALLEL="${MAX_PARALLEL:-3}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"

REPO_ROOT="$(pwd)"
STANDALONE_DIR="TraceLens/AgenticMode/Standalone"
EVALS_DIR="$REPO_ROOT/evals"
TEST_TRACES_CSV="$EVALS_DIR/unit_test_traces.csv"
DEXEC="docker exec -w $REPO_ROOT $CONTAINER"
STATUS_FILE="$(mktemp)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ts() { date "+%H:%M:%S"; }

log_status() {
    flock 1 echo "$@"
}

# ---------------------------------------------------------------------------
# Single job: generate one golden reference
# ---------------------------------------------------------------------------

generate_single_ref() {
    local id="$1" trace_path="$2" reference_dir="$3" platform="$4"
    local tag="[$id]"

    local REF_DIR="$REPO_ROOT/$reference_dir"
    local CASE_DIR="$(dirname "$REF_DIR")"
    local OUTPUT_DIR="$CASE_DIR/analysis_output"

    # Verify trace file exists
    if [ ! -f "$REPO_ROOT/$trace_path" ]; then
        log_status "  $tag ERROR: Trace file not found: $trace_path — skipping."
        flock "$STATUS_FILE" bash -c "echo 'failed' >> '$STATUS_FILE'"
        return 1
    fi

    log_status "  $tag [$(ts)] Generating golden reference..."
    $DEXEC bash -c "mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR"

    # Run standalone analysis with retry + backoff
    local agent_success=false
    local agent_attempts=0
    while [ "$agent_success" = false ] && [ "$agent_attempts" -lt 3 ]; do
        agent_attempts=$((agent_attempts + 1))
        (
            cd "$STANDALONE_DIR"
            agent --print --force --trust --output-format stream-json \
                "Run standalone analysis on $trace_path with platform $platform, node $(hostname), container $CONTAINER, output to $OUTPUT_DIR"
        ) < /dev/null > "$CASE_DIR/analysis_stream.ndjson" 2>&1

        if head -c 2048 "$CASE_DIR/analysis_stream.ndjson" | grep -qiE 'Error:.*unavailable|Service Unavailable'; then
            log_status "  $tag Attempt $agent_attempts/3 failed (agent unavailable). Backing off 30s..."
            sleep 30
        else
            agent_success=true
        fi
    done

    if [ "$agent_success" = false ]; then
        log_status "  $tag FAILED after 3 attempts."
        flock "$STATUS_FILE" bash -c "echo 'failed' >> '$STATUS_FILE'"
        return 1
    fi

    # Verify output was generated
    if [ ! -f "$OUTPUT_DIR/standalone_analysis.md" ]; then
        log_status "  $tag WARNING: standalone_analysis.md not found in output."
        flock "$STATUS_FILE" bash -c "echo 'failed' >> '$STATUS_FILE'"
        return 1
    fi

    # Copy output as reference
    cp -r "$OUTPUT_DIR" "$REF_DIR"

    # Remove unwanted files from reference dir (keep only standalone_analysis.md + perf_report_csvs/)
    rm -rf "$REF_DIR/category_data" \
           "$REF_DIR/category_findings" \
           "$REF_DIR/system_findings" \
           "$REF_DIR/metadata" \
           "$REF_DIR/perf_improvement.png" \
           "$REF_DIR/perf_improvement_base64.txt" \
           "$REF_DIR/plot_data.json"

    # Remove intermediate analysis output and log
    rm -rf "$OUTPUT_DIR"
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
echo "  Node:         $(hostname)"
echo "  Container:    $CONTAINER"
echo "  Max parallel: $MAX_PARALLEL"
echo "========================================="
echo ""

setup_semaphore

while IFS=, read -r id sub_category trace_path reference_dir platform <&3; do
    [[ -z "$id" ]] && continue

    read -u4  # acquire semaphore slot
    (
        generate_single_ref "$id" "$trace_path" "$reference_dir" "$platform" || true
        sleep "$SLEEP_BETWEEN"
        echo >&4  # release semaphore slot
    ) &
done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)

wait

# Tally results from status file
generated=$(grep -c '^generated$' "$STATUS_FILE" 2>/dev/null || echo 0)
failed=$(grep -c '^failed$' "$STATUS_FILE" 2>/dev/null || echo 0)
total=$((generated + failed))

echo ""
echo "========================================="
echo "  Golden Reference Generation Complete"
echo "  Total: $total | Generated: $generated | Failed: $failed"
echo "========================================="
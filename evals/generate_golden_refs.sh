#!/usr/bin/env bash
set -uo pipefail

# Environment config
NODE="tw053"
CONTAINER="tracelens_evals"

# Paths
REPO_ROOT="$(pwd)"
STANDALONE_DIR="TraceLens/AgenticMode/Standalone"
EVALS_DIR="$REPO_ROOT/evals"
TEST_TRACES_CSV="$EVALS_DIR/unit_test_traces.csv"
PREFIX="docker exec -w $REPO_ROOT $CONTAINER"

echo "========================================="
echo "  Golden Reference Generation"
echo "========================================="
echo ""

total=0
skipped=0
generated=0
failed=0

while IFS=, read -r id sub_category trace_path reference_dir platform <&3; do
    [[ -z "$id" ]] && continue
    total=$((total + 1))

    REF_DIR="$REPO_ROOT/$reference_dir"
    CASE_DIR="$(dirname "$REF_DIR")"
    OUTPUT_DIR="$CASE_DIR/analysis_output"


    # Verify trace file exists
    if [ ! -f "$REPO_ROOT/$trace_path" ]; then
        echo "  [$id] ERROR: Trace file not found: $trace_path — skipping."
        failed=$((failed + 1))
        continue
    fi

    echo "-----------------------------------------"
    echo "  [$id] Generating golden reference..."
    echo "    Trace:    $trace_path"
    echo "    Platform: $platform"
    echo "    Output:   $OUTPUT_DIR"
    echo "-----------------------------------------"

    ssh -n "$NODE" "$PREFIX bash -c 'mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR'"

    # Run standalone analysis with retry + backoff
    agent_attempts=0
    agent_success=false
    while [ "$agent_success" = false ] && [ "$agent_attempts" -lt 3 ]; do
        agent_attempts=$((agent_attempts + 1))
        (
            cd "$STANDALONE_DIR"
            agent --print --force --trust \
                "Run standalone analysis on $trace_path with platform $platform, node $NODE, container $CONTAINER, output to $OUTPUT_DIR"
        ) < /dev/null 2>&1 | tee "$CASE_DIR/analysis.log"

        if head -c 2048 "$CASE_DIR/analysis.log" | grep -qiE 'Error:.*unavailable|Service Unavailable'; then
            echo "  [$id] Attempt $agent_attempts/3 failed (agent unavailable). Backing off 30s..."
            sleep 30
        else
            agent_success=true
        fi
    done

    if [ "$agent_success" = false ]; then
        echo "  [$id] FAILED after 3 attempts."
        failed=$((failed + 1))
        continue
    fi

    # Verify output was generated
    if [ ! -f "$OUTPUT_DIR/standalone_analysis.md" ]; then
        echo "  [$id] WARNING: standalone_analysis.md not found in output."
        failed=$((failed + 1))
        continue
    fi

    # Copy output as reference
    cp -r "$OUTPUT_DIR" "$REF_DIR"
    echo "  [$id] Reference saved to $reference_dir"
    generated=$((generated + 1))

    sleep 30

done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)

echo ""
echo "========================================="
echo "  Golden Reference Generation Complete"
echo "  Total: $total | Generated: $generated | Skipped: $skipped | Failed: $failed"
echo "========================================="
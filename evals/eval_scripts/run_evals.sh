#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
set -uo pipefail

# ---------------------------------------------------------------------------
# Usage: bash run_evals.sh [standalone|comparative]
#
# MODE can also be set via the MODE environment variable.
# Defaults to standalone.d
#
# CONTAINER is optional. If set (e.g. CONTAINER=my_container), python and
# setup commands run via: docker exec -w $REPO_ROOT $CONTAINER ...
# If unset or empty, those commands run on the host in the current shell's
# environment (still expects cwd to be REPO_ROOT).
# ---------------------------------------------------------------------------
MODE="${1:-${MODE:-standalone}}"

if [[ "$MODE" != "standalone" && "$MODE" != "comparative" ]]; then
    echo "ERROR: Unknown mode '$MODE'. Use 'standalone' or 'comparative'." >&2
    exit 1
fi

CONTAINER="${CONTAINER:-}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-30}"
TEST_IDS="${TEST_IDS:-}"

REPO_ROOT="$(pwd)"
STANDALONE_DIR="TraceLens/AgenticMode/Standalone"
EVALS_DIR="$REPO_ROOT/evals"

if [[ "$MODE" == "comparative" ]]; then
    TEST_TRACES_CSV="${TEST_TRACES_CSV:-$EVALS_DIR/unit_test_traces_comparative.csv}"
    RESULTS_ROOT="${RESULTS_ROOT:-$EVALS_DIR/results_comparative}"
else
    TEST_TRACES_CSV="${TEST_TRACES_CSV:-$EVALS_DIR/unit_test_traces.csv}"
    RESULTS_ROOT="${RESULTS_ROOT:-$EVALS_DIR/results}"
fi

if [[ -n "$CONTAINER" ]]; then
    DEXEC=(docker exec -w "$REPO_ROOT" "$CONTAINER")
    RUNTIME_LABEL="container $CONTAINER"
    NODE_LABEL="node $(hostname)"
else
    DEXEC=()
    RUNTIME_LABEL="host (no container)"
    NODE_LABEL="local"
fi

mkdir -p "$RESULTS_ROOT"
"${DEXEC[@]}" bash -c "mkdir -p $RESULTS_ROOT && chmod -R 777 $RESULTS_ROOT"

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

if [[ "$MODE" == "comparative" ]]; then
    expand_archive unit_tests_comparative
    expand_archive e2e_tests_comparative
else
    expand_archive unit_tests
    expand_archive e2e_tests
fi

echo "========================================="
echo "  Standalone Analysis Evaluation"
echo "  Mode:      $MODE"
echo "  Node:      $NODE_LABEL"
echo "  Runtime:   $RUNTIME_LABEL"
echo "  CSV:       $TEST_TRACES_CSV"
if [[ -n "$TEST_IDS" ]]; then
    echo "  Test filter: $TEST_IDS"
fi
echo "========================================="
echo ""

_run_case() {
    local id="$1" trace1_path="$2" trace2_path="$3" reference_dir="$4" platform="$5"

    CASE_RESULTS="$RESULTS_ROOT/$id"
    OUTPUT_DIR="$CASE_RESULTS/analysis_output"
    rm -rf "$CASE_RESULTS"
    mkdir -p "$OUTPUT_DIR"
    "${DEXEC[@]}" bash -c "mkdir -p $OUTPUT_DIR && chmod -R 777 $OUTPUT_DIR"
    "${DEXEC[@]}" bash -c "mkdir -p $CASE_RESULTS && chmod -R 777 $CASE_RESULTS"

    # Phase 1: Analysis (with retry + backoff)
    if [[ "$MODE" == "comparative" ]]; then
        if [[ ! -f "$trace1_path" ]] || [[ ! -f "$trace2_path" ]]; then
            echo "  [$id] SKIP: missing trace files"
            echo "           expected: $trace1_path and $trace2_path"
            return
        fi
        echo "  [$id] Comparative analysis: $trace1_path vs $trace2_path ($platform)..."
    else
        echo "  [$id] Analyzing $trace1_path on $platform..."
    fi

    local agent_success=false
    local agent_attempts=0
    while [ "$agent_success" = false ] && [ "$agent_attempts" -lt 3 ]; do
        agent_attempts=$((agent_attempts + 1))
        (
            if [[ "$MODE" == "comparative" ]]; then
                agent --model claude-opus-4-7-high --print --force --trust --output-format stream-json \
                    "Run comparative analysis following the orchestrator skill (use standalone-analysis-orchestrator) on $trace1_path and $trace2_path with platform $platform (baseline is trace1), $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR"
            else
                agent --model claude-opus-4-7-high --print --force --trust --output-format stream-json \
                    "Run standalone analysis following the orchestrator skill on $trace1_path with platform $platform, $NODE_LABEL, $RUNTIME_LABEL, output to $OUTPUT_DIR"
            fi
        ) < /dev/null > "$CASE_RESULTS/analysis_stream.ndjson" 2>&1

        python3 "$EVALS_DIR/eval_scripts/process_stream.py" "$CASE_RESULTS/analysis_stream.ndjson" 2>&1 || true
        if head -c 2048 "$CASE_RESULTS/analysis_stream.ndjson" | grep -qiE 'Error:.*unavailable|Service Unavailable'; then
            echo "  [$id] Attempt $agent_attempts/3 failed (agent unavailable). Backing off 30s..."
            sleep 30
        else
            agent_success=true
        fi
    done

    if [ "$agent_success" = false ]; then
        echo "  [$id] Analysis FAILED after 3 attempts (agent unavailable). Skipping evals."
        return
    fi
    echo "  [$id] Analysis complete."

    sleep "$SLEEP_BETWEEN"

    # Phase 2: Evals (4 parallel tasks: 2 scripted + 2 LLM)
    echo "  [$id] Running evals in parallel..."
    local eval_pids=()

    echo "    -> Scripted workflow evals"
    "${DEXEC[@]}" python3 "$EVALS_DIR/eval_utils/workflow_scripted_evals.py" \
        --output-dir "$OUTPUT_DIR" \
        --results "$CASE_RESULTS/workflow_scripted_results.csv" \
        --comparison-scope "$MODE" \
        > "$CASE_RESULTS/workflow_scripted_eval.log" 2>&1 &
    eval_pids+=($!)

    echo "    -> LLM workflow evals"
    (
        cd "$EVALS_DIR"
        agent --model claude-opus-4-7-high --print --force --trust --output-format stream-json \
            "Run workflow LLM eval skill on $OUTPUT_DIR for test case $id with comparison_scope=$MODE. Write results to $CASE_RESULTS/workflow_llm_results.csv"
    ) < /dev/null > "$CASE_RESULTS/workflow_llm_eval.ndjson" 2>&1 &
    eval_pids+=($!)

    echo "    -> Scripted quality evals"
    "${DEXEC[@]}" python3 "$EVALS_DIR/eval_utils/quality_scripted_evals.py" \
        --output-dir "$OUTPUT_DIR" --reference-dir "$reference_dir" \
        --results "$CASE_RESULTS/quality_scripted_results.csv" \
        --comparison-scope "$MODE" \
        > "$CASE_RESULTS/quality_scripted_eval.log" 2>&1 &
    eval_pids+=($!)

    echo "    -> LLM quality evals"
    (
        cd "$EVALS_DIR"
        agent --model claude-opus-4-7-high --print --force --trust --output-format stream-json \
            "Run quality LLM eval skill on $OUTPUT_DIR with reference $reference_dir for test case $id with comparison_scope=$MODE. Write results to $CASE_RESULTS/quality_llm_results.csv"
    ) < /dev/null > "$CASE_RESULTS/quality_llm_eval.ndjson" 2>&1 &
    eval_pids+=($!)

    for pid in "${eval_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
    echo "  [$id] Evals complete."

    # Aggregate Results
    "${DEXEC[@]}" python3 "$EVALS_DIR/eval_utils/merge_results.py" \
        --results-dir "$CASE_RESULTS" \
        --output "$CASE_RESULTS/eval_summary.csv" || true
    echo "  [$id] Summary written to $CASE_RESULTS/eval_summary.csv"
    echo ""
}

if [[ "$MODE" == "comparative" ]]; then
    # comparative CSV: id,sub_category,trace1_path,trace2_path,reference_dir,platform
    while IFS=, read -r id sub_category trace1_path trace2_path reference_dir platform <&3; do
        [[ -z "$id" ]] && continue
        if [[ -n "$TEST_IDS" ]]; then
            case " $TEST_IDS " in
                *" $id "*) ;;
                *) continue ;;
            esac
        fi
        _run_case "$id" "$REPO_ROOT/$trace1_path" "$REPO_ROOT/$trace2_path" "$REPO_ROOT/$reference_dir" "$platform"
    done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
else
    # standalone CSV: id,sub_category,trace_path,reference_dir,platform
    while IFS=, read -r id sub_category trace_path reference_dir platform <&3; do
        [[ -z "$id" ]] && continue
        if [[ -n "$TEST_IDS" ]]; then
            case " $TEST_IDS " in
                *" $id "*) ;;
                *) continue ;;
            esac
        fi
        _run_case "$id" "$REPO_ROOT/$trace_path" "" "$REPO_ROOT/$reference_dir" "$platform"
    done 3< <(tail -n +2 "$TEST_TRACES_CSV"; echo)
fi

echo ""
echo "========================================="
echo "  Eval run finished."
echo "  Results in: $RESULTS_ROOT"
echo "========================================="

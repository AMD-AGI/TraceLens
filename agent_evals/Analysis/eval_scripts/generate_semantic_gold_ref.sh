#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

# ---------------------------------------------------------------------------
# One-time gold-reference generation for the semantic-purity comparative test
# cases (semantic_purity_deepseek_r1, semantic_purity_qwen3_30b_a3b).
#
# For each test case this:
#   1. Expands the with-capture-only fixture tarball if not already expanded.
#   2. Derives a sibling no-capture trace pair (copy + strip
#      torch_trace/capture_traces/) -- this is what Phase 1 of
#      run_repeatability_parallel.sh feeds to the candidate agent runs.
#   3. Runs the full analysis orchestrator once on the WITH-CAPTURE trace pair
#      (same agent invocation as run_repeatability_parallel.sh/generate_ref.sh)
#      and copies the resulting perf_report_trace1_csvs/diff_stats.csv into
#      the test case's reference_dir as semantic_purity_gold_diff_stats.csv.
#      This file's presence is what self-gates semantic_partition_scripted_evals.py
#      onto these two test cases (see that script and the "Adding a semantic-purity
#      test case" section of README.md).
#
# This is a manual, one-time step (like generate_ref.sh) -- not part of the
# repeated Phase 1/2 test loop, since gold has no LLM involved and doesn't
# vary across repeats.
#
# Usage: bash generate_semantic_gold_ref.sh [test_id ...]
#   With no arguments, generates gold for both semantic-purity test cases.
# ---------------------------------------------------------------------------
set -uo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
ANALYSIS_DIR="$REPO_ROOT/TraceLens/Agent/Analysis"
EVALS_DIR="$REPO_ROOT/agent_evals/Analysis"
TESTS_DIR="$EVALS_DIR/analysis_tests"
GOLD_FILENAME="semantic_purity_gold_diff_stats.csv"

# id -> "with_capture_subdir platform1 platform2"
declare -A CASES=(
    [semantic_purity_deepseek_r1]="with_capture MI300 B300"
    [semantic_purity_qwen3_30b_a3b]="with_capture MI300 B300"
)

ts() { date "+%H:%M:%S"; }

expand_archive() {
    local name="$1"
    local archive="$TESTS_DIR/${name}.tar.gz"
    local target="$TESTS_DIR/$name"
    if [[ -f "$archive" ]] && [[ ! -d "$target" ]]; then
        echo "Expanding ${name}.tar.gz..."
        tar xzf "$archive" -C "$REPO_ROOT"
    fi
}

# Derive a no-capture trace tree from a with-capture one: identical file tree,
# minus torch_trace/capture_traces/ under each device folder.
derive_no_capture() {
    local with_capture_dir="$1"
    local no_capture_dir="$2"
    if [[ -d "$no_capture_dir" ]]; then
        echo "  no-capture dir already exists, skipping derivation: $no_capture_dir"
        return
    fi
    echo "  Deriving no-capture trace tree: $no_capture_dir"
    cp -r "$with_capture_dir" "$no_capture_dir"
    find "$no_capture_dir" -type d -name capture_traces -prune -exec rm -rf {} \;
}

generate_one() {
    local id="$1"
    local subdir platform1 platform2
    read -r subdir platform1 platform2 <<< "${CASES[$id]}"

    local case_dir="$TESTS_DIR/$id"
    local with_capture_dir="$case_dir/$subdir"
    local no_capture_dir="$case_dir/no_capture"
    local reference_dir="$case_dir/analysis_output_ref"

    expand_archive "$id"

    if [[ ! -d "$with_capture_dir" ]]; then
        echo "ERROR [$id]: with-capture trace dir not found: $with_capture_dir" >&2
        return 1
    fi

    local trace1="$with_capture_dir/MI300"
    local trace2="$with_capture_dir/B300"
    derive_no_capture "$with_capture_dir" "$no_capture_dir"

    local scratch_dir="$case_dir/_gold_generation_scratch"
    local output_dir="$scratch_dir/analysis_output"
    rm -rf "$scratch_dir"
    mkdir -p "$output_dir"

    echo "[$id] [$(ts)] Running full orchestrator on WITH-CAPTURE trace pair (gold generation)..."
    local agent_success=false
    local attempts=0
    while [ "$agent_success" = false ] && [ "$attempts" -lt 3 ]; do
        attempts=$((attempts + 1))
        (
            cd "$ANALYSIS_DIR" || exit
            agent --model claude-opus-4-8-thinking-medium --print --force --trust --output-format stream-json \
                "Follow the analysis orchestrator installed with the TraceLens pip package (look under TraceLens/Agent/Analysis/.cursor/skills/ in the package installation directory) and run the full agentic analysis workflow on $trace1 and $trace2 with platform $platform1 (trace1) and $platform2 (trace2), analysis mode default, output to $output_dir"
        ) < /dev/null > "$scratch_dir/gold_generation_stream.ndjson" 2>&1

        if head -c 2048 "$scratch_dir/gold_generation_stream.ndjson" | grep -qiE 'Error:.*unavailable|Service Unavailable'; then
            echo "[$id] Attempt $attempts/3 failed (agent unavailable). Backing off 30s..."
            sleep 30
        else
            agent_success=true
        fi
    done

    if [ "$agent_success" = false ]; then
        echo "ERROR [$id]: FAILED after 3 attempts (agent unavailable)." >&2
        return 1
    fi

    local gold_src="$output_dir/perf_report_trace1_csvs/diff_stats.csv"
    if [[ ! -f "$gold_src" ]]; then
        echo "ERROR [$id]: expected gold source not found: $gold_src" >&2
        return 1
    fi

    mkdir -p "$reference_dir"
    cp "$gold_src" "$reference_dir/$GOLD_FILENAME"
    rm -rf "$scratch_dir"

    echo "[$id] [$(ts)] Gold reference saved: $reference_dir/$GOLD_FILENAME"
}

if [[ $# -gt 0 ]]; then
    ids=("$@")
else
    ids=("${!CASES[@]}")
fi

status=0
for id in "${ids[@]}"; do
    if [[ -z "${CASES[$id]:-}" ]]; then
        echo "ERROR: unknown test id '$id'" >&2
        status=1
        continue
    fi
    generate_one "$id" || status=1
done

exit $status

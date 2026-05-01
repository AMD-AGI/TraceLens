#!/usr/bin/env bash
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Run comparative analysis for every trace pair in unit_tests_comparative/.
# All analyses are launched in parallel.
#
# Usage:
#   ./run_analyses.sh
#
# The agent CLI must be available on PATH. Each analysis output is written to:
#   <test_subdir>/analysis_output_ref/

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STANDALONE_DIR="$(cd "$SCRIPT_DIR/../../TraceLens/AgenticMode/Standalone" && pwd)"

pids=()
logs=()

for test_dir in "$SCRIPT_DIR"/test*/; do
    trace1="$test_dir/trace1.json"
    trace2="$test_dir/trace2.json"
    output_dir="$test_dir/analysis_output_ref"
    test_name="$(basename "$test_dir")"
    log="$test_dir/agent.log"

    if [[ ! -f "$trace1" || ! -f "$trace2" ]]; then
        echo "[$test_name] Skipping — trace1.json or trace2.json not found"
        continue
    fi

    mkdir -p "$output_dir"

    echo "[$test_name] Launching agent..."
    (
        cd "$STANDALONE_DIR"
        agent --print --force --trust --output-format stream-json \
            "Run comparative analysis on $trace1 and $trace2. Platform for trace1 is MI300X, for trace2 is H100. Run default, local. output dir is $output_dir"
    ) < /dev/null > "$log" 2>&1 &

    pids+=($!)
    logs+=("$log")
    echo "[$test_name] PID $! — log: $log"
done

echo ""
echo "All ${#pids[@]} analyses running in parallel. Waiting for completion..."
echo ""

failed=0
for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    log="${logs[$i]}"
    if wait "$pid"; then
        echo "[PID $pid] Done"
    else
        echo "[PID $pid] FAILED (exit $?) — see $log"
        failed=$((failed + 1))
    fi
done

echo ""
if [[ "$failed" -eq 0 ]]; then
    echo "All analyses complete."
else
    echo "$failed analysis/analyses failed."
fi

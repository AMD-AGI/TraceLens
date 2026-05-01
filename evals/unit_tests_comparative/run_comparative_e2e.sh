#!/usr/bin/env bash
# Run profilelens validate-comparative on every test case in this directory.
# Each subdirectory must contain trace1.json, trace2.json, and an
# analysis_output_ref/comparative_analysis.md report.
# Results are saved to results/<name>/ inside this script's directory.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
E2E_DIR="${SCRIPT_DIR}"
RESULTS_DIR="${SCRIPT_DIR}/results"
if [[ -d "${RESULTS_DIR}" && -n "$(ls -A "${RESULTS_DIR}" 2>/dev/null)" ]]; then
  n=2
  while [[ -d "${SCRIPT_DIR}/results_${n}" && -n "$(ls -A "${SCRIPT_DIR}/results_${n}" 2>/dev/null)" ]]; do
    (( n++ ))
  done
  RESULTS_DIR="${SCRIPT_DIR}/results_${n}"
fi
PARALLEL="${PARALLEL:-16}"   # validation workers per case
JOBS="${JOBS:-0}"            # cases to run concurrently; 0 = all at once

usage() {
  echo "Usage: $0 [--parallel N] [--jobs N] [test_name ...]"
  echo ""
  echo "  --parallel N   Validation workers per case (default: ${PARALLEL})"
  echo "  --jobs N       Cases to run concurrently; 0 = all at once (default: ${JOBS})"
  echo "  test_name ...  Optional list of subdirectory names to run (default: all)"
  echo ""
  echo "Environment:"
  echo "  PARALLEL=N   Alternative way to set --parallel"
  echo "  JOBS=N       Alternative way to set --jobs"
  exit 0
}

# Parse args
FILTER=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel|-p) PARALLEL="$2"; shift 2 ;;
    --jobs|-j)     JOBS="$2";     shift 2 ;;
    --help|-h)     usage ;;
    *)             FILTER+=("$1"); shift ;;
  esac
done

mkdir -p "${RESULTS_DIR}"

echo "========================================"
echo "  ProfileLens Comparative E2E Runner"
echo "========================================"
echo "  E2E dir:   ${E2E_DIR}"
echo "  Results:   ${RESULTS_DIR}"
echo "  Workers:   ${PARALLEL} per case"
[[ ${#FILTER[@]} -gt 0 ]] && echo "  Filter:    ${FILTER[*]}"
echo ""

run_case() {
  local name="$1"
  local case_dir="${E2E_DIR}/${name}"
  local out_dir="${RESULTS_DIR}/${name}"
  local report="${case_dir}/analysis_output_ref/comparative_analysis.md"
  local log="${out_dir}/profilelens.log"

  mkdir -p "${out_dir}"

  local baseline="${case_dir}/trace1.json"
  local candidate="${case_dir}/trace2.json"

  if [[ ! -f "${baseline}" || ! -f "${candidate}" ]]; then
    echo "  [SKIP] ${name}: trace1.json or trace2.json not found"
    return 2
  fi

  if [[ ! -f "${report}" ]]; then
    echo "  [SKIP] ${name}: report not found"
    return 2
  fi

  echo "  [RUN ] ${name} → ${log}"

  profilelens validate-comparative \
    "${baseline}" \
    "${candidate}" \
    "${report}" \
    --parallel "${PARALLEL}" \
    --output rich \
    --save "${out_dir}/results.md" \
    >"${log}" 2>&1

  echo "  [DONE] ${name}"
}

# Collect test cases
ALL_CASES=()
for d in "${E2E_DIR}"/*/; do
  name="$(basename "${d}")"
  [[ "${name}" == "agent_logs" || "${name}" == "results" ]] && continue
  ALL_CASES+=("${name}")
done

if [[ ${#FILTER[@]} -gt 0 ]]; then
  CASES=()
  for f in "${FILTER[@]}"; do
    if [[ " ${ALL_CASES[*]} " == *" ${f} "* ]]; then
      CASES+=("${f}")
    else
      echo "  [WARN] Unknown test case: ${f}"
    fi
  done
else
  CASES=("${ALL_CASES[@]}")
fi

if [[ ${#CASES[@]} -eq 0 ]]; then
  echo "No test cases to run."
  exit 1
fi

echo "Running ${#CASES[@]} test case(s): ${CASES[*]}"
echo ""

# --jobs 0 means run all at once
[[ "${JOBS}" -eq 0 ]] && JOBS="${#CASES[@]}"

echo "  Jobs:      ${JOBS} concurrent cases"
echo ""

FAILED=()
SKIPPED=()
PASSED=()
declare -A pids=()  # pid → case name

record_outcome() {
  local name="$1" rc="$2"
  if [[ $rc -eq 0 ]]; then
    PASSED+=("${name}")
    echo "  [PASS] ${name}"
  elif [[ $rc -eq 2 ]]; then
    SKIPPED+=("${name}")
    echo "  [SKIP] ${name}"
  else
    FAILED+=("${name}")
    echo "  [FAIL] ${name} (rc=${rc}) — see results/${name}/profilelens.log"
  fi
}

reap_one() {
  # Wait for any one child to finish (bash 4.3+ supports wait -n).
  local pid rc name
  wait -n -p pid || rc=$?   # capture non-zero without triggering set -e
  rc=${rc:-0}
  name="${pids[$pid]:-unknown}"
  unset "pids[$pid]"
  record_outcome "${name}" "${rc}"
}

for name in "${CASES[@]}"; do
  # Throttle: block until a slot is free
  while [[ ${#pids[@]} -ge ${JOBS} ]]; do
    reap_one
  done

  run_case "${name}" &
  pids[$!]="${name}"
done

# Drain remaining jobs
while [[ ${#pids[@]} -gt 0 ]]; do
  reap_one
done

echo ""
echo "========================================"
echo "  Summary"
echo "========================================"
echo "  Passed:  ${#PASSED[@]}  ${PASSED[*]:-}"
echo "  Skipped: ${#SKIPPED[@]}  ${SKIPPED[*]:-}"
echo "  Failed:  ${#FAILED[@]}  ${FAILED[*]:-}"
echo ""
echo "  Logs:    ${RESULTS_DIR}/<name>/profilelens.log"
echo "  Results: ${RESULTS_DIR}/<name>/results.md"
echo "========================================"

[[ ${#FAILED[@]} -gt 0 ]] && exit 1
exit 0

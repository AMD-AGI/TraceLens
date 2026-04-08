#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-modular_evals}"
PR_NUMBER="${PR_NUMBER:-}"
ISSUE_NUMBER="${ISSUE_NUMBER:-}"

if [[ -z "$PR_NUMBER" ]]; then
  echo "Error: PR_NUMBER is required."
  echo "Example: PR_NUMBER=189 ISSUE_NUMBER=200 bash evals/eval_scripts/run_format_test_publish.sh"
  exit 1
fi

if [[ -z "$ISSUE_NUMBER" ]]; then
  echo "Error: ISSUE_NUMBER is required for direct fix-ticket comment publishing."
  echo "Example: PR_NUMBER=189 ISSUE_NUMBER=200 bash evals/eval_scripts/run_format_test_publish.sh"
  exit 1
fi

CONTAINER="$CONTAINER" \
python3 evals/eval_scripts/run_eval_pipeline.py \
  --suite e2e \
  --num-repeats 5 \
  --max-parallel 5 \
  --sleep-between 30 \
  --run-id e2e_fullscale_publish \
  --publish \
  --pr-number "$PR_NUMBER" \
  --issue-number "$ISSUE_NUMBER"

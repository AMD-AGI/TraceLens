#!/usr/bin/env bash
set -euo pipefail

CONTAINER="${CONTAINER:-modular_evals}"

CONTAINER="$CONTAINER" \
python3 evals/eval_scripts/run_eval_pipeline.py \
  --suite unit \
  --num-repeats 2 \
  --max-parallel 2 \
  --sleep-between 30 \
  --run-id format_test_local \
  --no-validate

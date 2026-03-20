<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Standalone Analysis Evals

Evaluation harness for the TraceLens Standalone Analysis pipeline. Validates both **workflow correctness** (directory structure, file existence, report formatting) and **output quality** (comparison against reference reports).

## Prerequisites / Setup

### 1. Clone TraceLens-internal

```bash
git clone https://github.com/AMD-AGI/TraceLens-internal.git
cd TraceLens-internal
```

### 2. Install TraceLens inside your container

SSH into your node and exec into the container:

```bash
ssh <node>
docker exec -it <container> bash
```

Install TraceLens:

```bash
cd /path/to/TraceLens-internal
pip install -e .
```

### 3. Install Cursor CLI

The eval scripts invoke the Cursor agent via the `agent` CLI command. All scripts are designed to run **directly on the node** (not the head node). The `agent` CLI must be installed on the node.

```bash
curl https://cursor.com/install -fsS | bash
```
Verify with `agent --version`

### 4. Set Model to Claude Opus 4.6

The `agent` invocations require setting the default model to **Claude Opus 4.6**.

## Running Scripts

All scripts run **on the node** from the repo root. They use `docker exec` to run Python scripts inside the container, while `agent` commands run directly on the node. The `CONTAINER` environment variable is required for all scripts.

### Full Eval Run (single pass)

```bash
CONTAINER=my_container bash evals/eval_scripts/run_evals.sh
```

Runs standalone analysis on each test case, then runs workflow + quality evals, and merges results. Output goes to `evals/results/`.

### Repeatability Study (serial)

```bash
CONTAINER=my_container bash evals/eval_scripts/run_repeatability.sh
```

Runs each test case `NUM_REPEATS` times (default: 5) serially. Results go to `evals/repeatability_results/`.

### Repeatability Study (parallel)

Recommended for repeatability testing. Dispatches all `(test_case, repeat)` jobs concurrently with a configurable concurrency limit.

```bash
CONTAINER=my_container bash evals/eval_scripts/run_repeatability_parallel.sh
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `CONTAINER` | (required) | Docker container name |
| `MAX_PARALLEL` | 3 | Max concurrent jobs |
| `NUM_REPEATS` | 5 | Repeats per test case |
| `SLEEP_BETWEEN` | 30 | Seconds between Phase 1 and Phase 2 |

Full example:

```bash
MAX_PARALLEL=5 NUM_REPEATS=3 CONTAINER=my_container \
    bash evals/eval_scripts/run_repeatability_parallel.sh
```

Use a `screen` session to prevent disconnects during long runs. Each job cleans its output directory before starting to prevent stale results.

### Generate Golden References

Generates `analysis_output_ref/` for test cases listed in `unit_test_traces.csv`. Runs standalone analysis, copies the output as the reference, and strips intermediate files (keeps only `standalone_analysis.md` and `perf_report_csvs/`). Runs in parallel with `MAX_PARALLEL`. Skips test cases with missing trace files.

```bash
CONTAINER=my_container bash evals/eval_scripts/generate_golden_refs.sh
```

Or with more parallelism:

```bash
MAX_PARALLEL=5 CONTAINER=my_container bash evals/eval_scripts/generate_golden_refs.sh
```

### Individual Manual Runs

You can run each stage independently using the `agent` CLI. Examples:

**Standalone Analysis:**

```bash
cd TraceLens/AgenticMode/Standalone
agent --force "Run standalone analysis on <trace_path> with platform <platform>, node <node>, container <container>, output to <output_dir>"
```

**Workflow Eval:**

```bash
cd evals
agent --force "Run the workflow eval skill on <output_dir> for test case <id>. Write results to <results_path>"
```

**Quality Eval:**

```bash
cd evals
agent --force "Run the quality eval skill on <output_dir> with reference <reference_dir> for test case <id>. Write results to <results_path>"
```

## Adding New Test Cases

### 1. Create the test case directory

Each test case lives under `evals/unit_tests/<category>/` and must contain:

- The profiling trace JSON file.
- `analysis_output_ref/` -- a reference analysis output to compare against (used by quality evals). Should include `standalone_analysis.md` and `perf_report_csvs/`. Generate this with `generate_golden_refs.sh`.

### 2. Add a row to `unit_test_traces.csv`

The CSV has the following columns:

| Column | Description |
|---|---|
| `id` | Unique test case ID (e.g. `gemm_01_compute_few_tiles`) |
| `sub_category` | Category label (e.g. `gemm`) |
| `trace_path` | Relative path to the trace JSON |
| `reference_dir` | Relative path to the reference output (`analysis_output_ref`) |
| `platform` | Hardware platform (e.g. `MI300X`) |

Example row:

```
gemm_01_compute_few_tiles,gemm,evals/unit_tests/gemm/gemm_01_compute_few_tiles_analysis_output/gemm_01_compute_few_tiles.json,evals/unit_tests/gemm/gemm_01_compute_few_tiles_analysis_output/analysis_output_ref,MI300X
```

## Eval Pipeline Summary

### Shell Scripts

For each test case in `unit_test_traces.csv`, the scripts run two phases:

1. **Phase 1 -- Standalone Analysis:** Invokes the Standalone Analysis agent on the trace file. Output is written to `analysis_output/`. Agent output is logged as stream JSON (`.ndjson`).
2. **Phase 2 -- Workflow + Quality Evals (4 parallel tasks):** Launches four tasks concurrently:
   - Scripted workflow evals (via `docker exec`)
   - LLM workflow evals (via `agent`)
   - Scripted quality evals (via `docker exec`)
   - LLM quality evals (via `agent`)
3. **Merge Results:** Runs `eval_utils/merge_results.py` to combine per-eval CSVs into a single `eval_summary.csv`.

### Eval Skills

Two Cursor agent skills in `.cursor/skills/` define the eval logic:

**workflow-eval** -- 12 evals total:
- Evals 1-7 (scripted): directory structure, required files, plot output. Run via `eval_utils/workflow_scripted_evals.py`.
- Evals 8-12 (LLM-based): report template rendering, executive summary metrics, issue template formatting, hardware reference in appendix, sub-agent findings structure, and Impact Summary type validation (only `kernel_tuning` for compute, zero rows for system).

**quality-eval** -- 3 evals total:
- Eval 1 (scripted): CSV value alignment between generated and reference outputs. Run via `eval_utils/quality_scripted_evals.py`.
- Evals 2-3 (LLM-based): semantic comparison of compute issue titles and content (performance numbers, shapes, efficiency, pre-computed kernel_tuning gains) against the reference report. System-level P-items are skipped (no Impact field to compare).

## Results

Results are written to `evals/results/<id>/` (single run) or `evals/repeatability_results/<id>/run_<n>/` (repeatability) and include:

- `analysis_stream.ndjson` -- stream JSON output from the standalone analysis agent.
- `workflow_scripted_eval.log` / `workflow_scripted_results.csv` -- scripted workflow eval output.
- `workflow_llm_eval.ndjson` / `workflow_llm_results.csv` -- LLM workflow eval output.
- `quality_scripted_eval.log` / `quality_scripted_results.csv` -- scripted quality eval output.
- `quality_llm_eval.ndjson` / `quality_llm_results.csv` -- LLM quality eval output.
- `eval_summary.csv` -- merged results for the test case.

### Aggregating Repeatability Results

After repeatability runs complete, aggregate and analyze:

```bash
python evals/eval_utils/aggregate_repeatability.py
```

This produces three CSVs in `evals/eval_utils/output/`:

- `pass_rate_summary.csv` -- per-trace, per-eval pass rates across all runs
- `aggregated_results.csv` -- full eval results for every run
- `stream_diagnostics.csv` -- per-run metadata (outcome, token usage, last step reached, whether the report was written)

**What to look for:**

- **Overall pass rate** should not regress compared to the baseline on `main`.
- **Per-eval consistency** -- an eval that passes in 3/5 runs indicates flaky behavior that needs investigation.
- **Stream diagnostics regressions** -- runs where the report was not written or the last step reached regressed (e.g., stuck at Step 7 instead of reaching Step 11) signal a reliability problem.

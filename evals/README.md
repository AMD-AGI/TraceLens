<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Standalone Analysis Evals

Evaluation harness for the TraceLens Standalone Analysis pipeline. Validates both **workflow correctness** (directory structure, file existence, report formatting) and **output quality** (comparison against reference reports).

The framework uses a hybrid approach:
- **Deterministic scripted evals** (Python) for structural and factual checks — pre-check gates, per-item sub-scoring, root-cause metadata
- **LLM-judged evals** with multi-dimensional weighted scoring for semantic checks that require reasoning
- **Stability classification** (STABLE_PASS / FLAKY_PASS / FLAKY_FAIL / STABLE_FAIL) across repeated runs to separate deterministic bugs from LLM variability

All eval results use a 7-column CSV schema: `index, category, issue_summary, result, details, root_cause, recommended_fix`. See [EVAL_RUBRICS.md](EVAL_RUBRICS.md) for the full rubric reference.

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

Recommended for repeatability testing. Dispatches all `(test_case, repeat)` jobs concurrently with a configurable concurrency limit. After all jobs finish, the harness automatically invokes a Cursor agent to aggregate results and generate reports (see [Post-Processing Skill](#post-processing-skill)).

```bash
CONTAINER=my_container bash evals/eval_scripts/run_repeatability_parallel.sh
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `CONTAINER` | (required) | Docker container name |
| `MAX_PARALLEL` | 5 | Max concurrent jobs |
| `NUM_REPEATS` | 5 | Repeats per test case |
| `SLEEP_BETWEEN` | 30 | Seconds between Phase 1 and Phase 2 |
| `TEST_TRACES_CSV` | `evals/unit_test_traces.csv` | Path to the trace CSV to use |
| `RESULTS_ROOT` | `evals/repeatability_results` | Where per-run results are written |
| `REPORT_DIR` | `<RESULTS_ROOT>/../reports` | Where reports and reproducers are written |
| `SUITE_NAME` | `eval` | Suite label used in reports (e.g. `unit`, `e2e`) |
| `TEST_IDS` | (empty = all) | Space-separated trace IDs to run (filter) |
| `SKIP_POST_PROCESSING` | (empty) | Set to `1` to skip report generation after the eval loop |

Full example (e2e suite, 3 repeats, 2 parallel):

```bash
CONTAINER=my_container \
SUITE_NAME=e2e \
TEST_TRACES_CSV=evals/e2e_test_traces.csv \
RESULTS_ROOT=evals/eval_reports/my_run/results/e2e_repeatability_results \
REPORT_DIR=evals/eval_reports/my_run/reports \
NUM_REPEATS=3 MAX_PARALLEL=2 \
    bash evals/eval_scripts/run_repeatability_parallel.sh
```

Use a `screen` session to prevent disconnects during long runs. Each job cleans its output directory before starting to prevent stale results.

To run evals without generating reports (e.g. to re-process later):

```bash
SKIP_POST_PROCESSING=1 CONTAINER=my_container \
    bash evals/eval_scripts/run_repeatability_parallel.sh
```

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

## Post-Processing Skill

After the repeatability harness finishes, a Cursor agent is automatically invoked to aggregate results and generate reports. This is defined in `.cursor/skills/eval-post-processing.md`.

The skill performs four steps:

1. **Aggregate** -- Runs `aggregate_repeatability.py` to merge all per-run `eval_summary.csv` files into `aggregated_results.csv`, `pass_rate_summary.csv`, `stability_summary.csv`, and `stream_diagnostics.csv`. The stability summary classifies each (trace, eval) pair as `STABLE_PASS`, `FLAKY_PASS`, `FLAKY_FAIL`, or `STABLE_FAIL` based on pass rates across repeated runs.
2. **Classify** -- Reads the aggregate CSVs and `report_section_rules.yaml` to classify failures into report sections and failure modes. Uses `root_cause` and `recommended_fix` columns from eval results.
3. **Write reports** -- Generates `pr_report.md` (concise summary for a PR comment) and `fix_ticket_report.md` (detailed failure modes with root causes, suggested fixes, and reproducer commands).
4. **Build reproducer packages** -- Creates a self-contained folder per failure issue with sanitized stream JSONs, eval summaries, and a README. Each folder is compressed into a `.tar.gz` that can be assigned to a developer for debugging with Cursor.

### Re-running post-processing on existing results

You can re-run the post-processing skill on any previous results without re-running the eval loop:

```bash
cd evals
agent --print --force --trust \
    "Run eval post processing on results_root=<results_root> suite=<suite> test_traces_csv=<csv_path> report_dir=<report_dir> container=<container>"
```

Example:

```bash
cd evals
agent --print --force --trust \
    "Run eval post processing on results_root=evals/eval_reports/my_run/results/e2e_repeatability_results suite=e2e test_traces_csv=evals/e2e_test_traces.csv report_dir=evals/eval_reports/my_run/reports container=modular_evals"
```

### Error handling

- Each step logs its exit code. Failures are non-fatal — the pipeline continues and reports all failures in the summary.
- Root-owned files from Docker are cleaned using `docker exec rm` before host `rm`.
- The underlying scripts have built-in retry with backoff (3 attempts, 30s) for agent unavailability.

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

For each test case in the traces CSV, the scripts run two phases:

1. **Phase 1 -- Standalone Analysis:** Invokes the Standalone Analysis agent on the trace file. Output is written to `analysis_output/`. Agent output is logged as stream JSON (`.ndjson`). The orchestrator includes targeted subagent retry (1 retry per failed subagent in Steps 6, 7, and 9) to reduce flaky pipeline failures.
2. **Phase 2 -- Workflow + Quality Evals (4 parallel tasks):** Launches four tasks concurrently:
   - Scripted workflow evals — 13 evals with per-item sub-scoring (via `docker exec`)
   - LLM workflow eval — 1 eval with multi-dimensional weighted scoring (via `agent`)
   - Scripted quality evals — 1 eval (via `docker exec`)
   - LLM quality evals — 2 evals with multi-dimensional weighted scoring (via `agent`)
3. **Merge Results:** Runs `eval_utils/merge_results.py` to combine per-eval CSVs into a single `eval_summary.csv`.

### Eval Skills

Four Cursor agent skills in `.cursor/skills/` define the eval and pipeline logic:

**full-eval-pipeline** -- End-to-end orchestrator that chains all steps (extract, golden refs, ProfileLens, repeatability, re-archive) with interactive input prompts and progress logging. See [Full Eval Pipeline](#full-eval-pipeline-end-to-end) above.

**eval-post-processing** -- Aggregates repeatability results, classifies failures using `report_section_rules.yaml`, and generates PR + fix-ticket reports with reproducer packages. Invoked automatically by `run_repeatability_parallel.sh` or manually on existing results.

The remaining two skills define the per-case eval logic:

**workflow-eval** -- 14 evals (13 scripted + 1 LLM):

| Eval | Type | What it checks |
|------|------|---------------|
| 1–8 | Scripted | Directory structure, metadata files, model info, perf report, tree data, findings existence, plot |
| 9 | Scripted | Report template rendering — all required `##` headers present. Per-header sub-indices (`_executive_summary`, `_compute`, `_system`, `_detailed`, `_appendix`, `_metrics_table`) |
| 10 | Scripted | Executive Summary metrics — values match `gpu_timeline.csv` within 1%. Per-metric sub-indices (`_compute_pct`, `_idle_pct`, `_comm_pct`, `_total_time`, `_bottleneck`) |
| 11 | Scripted | Issue template rendering — each P-item has Insight/Action/Impact fields. Per-P-item sub-indices (`_compute_P1`, `_system_P2`, etc.) |
| 12 | LLM | Hardware Reference in Appendix — platform, HBM BW, MAF values. Multi-dimensional weighted scoring: correctness (50%) + completeness (50%), pass threshold ≥ 7.0 |
| 13 | Scripted | Sub-agent findings structure and Impact Summary types. Per-category sub-indices (`_gemm`, `_cpu_idle`, `_kernel_fusion`, etc.) |
| 14 | Scripted | Model identification in Appendix — all 4 `model_info.json` fields present. Per-field sub-indices (`_model`, `_architecture`, `_scale`, `_precision`) |

Scripted evals (1–11, 13–14) run via `eval_utils/workflow_scripted_evals.py`. LLM eval (12) runs via `.cursor/skills/workflow-llm-eval.md`.

All scripted evals include **pre-check gates** that immediately FAIL all evals with a clear message if the output directory is missing, `standalone_analysis.md` is absent/garbled, or other fundamental prerequisites are unmet.

**quality-eval** -- 3 evals (1 scripted + 2 LLM):

| Eval | Type | What it checks |
|------|------|---------------|
| 1 | Scripted | CSV value alignment between generated and reference `perf_report_csvs/`. Run via `eval_utils/quality_scripted_evals.py` |
| 2 | LLM | Compute Issue Title Alignment — semantic comparison of P-item titles against reference. Multi-dimensional scoring: correctness (40%) + completeness (30%) + precision (30%), pass threshold ≥ 7.0 |
| 3 | LLM | Compute Issue Content Alignment — performance numbers, shapes, efficiency, gains against reference. Same 3-dimension scoring as eval 2 |

LLM evals (2–3) run via `.cursor/skills/quality-llm-eval.md`. System-level P-items are skipped (no Impact field to compare).

## Results

Results are written to `evals/results/<id>/` (single run) or `evals/repeatability_results/<id>/run_<n>/` (repeatability) and include:

- `analysis_stream.ndjson` -- stream JSON output from the standalone analysis agent.
- `workflow_scripted_eval.log` / `workflow_scripted_results.csv` -- scripted workflow eval output.
- `workflow_llm_eval.ndjson` / `workflow_llm_results.csv` -- LLM workflow eval output.
- `quality_scripted_eval.log` / `quality_scripted_results.csv` -- scripted quality eval output.
- `quality_llm_eval.ndjson` / `quality_llm_results.csv` -- LLM quality eval output.
- `eval_summary.csv` -- merged results for the test case.

### CSV Schema

All eval result CSVs use the same 7-column schema:

```
index,category,issue_summary,result,details,root_cause,recommended_fix
```

| Column | Description |
|--------|-------------|
| `index` | Eval identifier (e.g., `workflow_eval_9_compute`, `quality_eval_2`) |
| `category` | `Workflow` or `Quality` |
| `issue_summary` | Human-readable name of the check |
| `result` | `PASS` or `FAIL` |
| `details` | Failure specifics, scoring breakdown, or match confirmation |
| `root_cause` | `pipeline`, `template`, or `data` (empty if PASS) |
| `recommended_fix` | Actionable fix suggestion (empty if PASS) |

### Aggregated Results and Reports

After a repeatability run, the post-processing skill generates reports in `<REPORT_DIR>/`:

- `aggregates/aggregated_results.csv` -- full eval results for every (trace, run) pair
- `aggregates/pass_rate_summary.csv` -- per-trace, per-eval pass rates across all runs
- `aggregates/stability_summary.csv` -- per-(trace, eval) stability classification (see below)
- `aggregates/stream_diagnostics.csv` -- per-run metadata (outcome, token usage, last step reached, whether the report was written)
- `pr_report.md` -- concise summary for posting as a PR comment
- `fix_ticket_report.md` -- detailed failure analysis with root causes, suggested fixes, and reproducer commands
- `reproducers/` -- per-issue reproducer packages (`.tar.gz` files with sanitized stream JSONs for debugging)

### Stability Classification

The `stability_summary.csv` classifies each (trace, eval) pair across repeated runs:

| Classification | Meaning | Action |
|---|---|---|
| `STABLE_PASS` | Passed 100% of runs | No action needed |
| `FLAKY_PASS` | Passed >50% but not all | Investigate LLM variability or edge cases |
| `FLAKY_FAIL` | Passed >0% but ≤50% | Likely a significant issue with intermittent masking |
| `STABLE_FAIL` | Failed 100% of runs | Deterministic bug — fix the pipeline code |

**What to look for:**

- **Overall pass rate** should not regress compared to the baseline on `main`.
- **STABLE_FAIL entries** in `stability_summary.csv` — these are deterministic bugs that need pipeline code fixes. Check the `root_cause` column to understand whether it's a `pipeline`, `template`, or `data` issue.
- **FLAKY_PASS / FLAKY_FAIL entries** — indicate LLM scoring variability or edge-case inconsistency. Check if the LLM eval scores hover near the 7.0 threshold (visible in the `details` column scoring breakdown).
- **Pre-check gate failures** — if many evals show `"Pre-check gate: ..."` in the details column, the standalone analysis pipeline itself failed to produce output. Focus on fixing the pipeline before investigating individual eval failures.
- **Stream diagnostics regressions** -- runs where the report was not written or the last step reached regressed (e.g., stuck at Step 7 instead of reaching Step 11) signal a reliability problem.
- **Reproducer packages** -- assign the `.tar.gz` for a specific issue to a developer. They can extract it, load the stream JSON into Cursor, and debug the root cause.

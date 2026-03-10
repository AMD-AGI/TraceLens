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

The eval scripts invoke the Cursor agent via the `agent` CLI command. 

Install on the head node:
```bash
curl https://cursor.com/install -fsS | bash
```
Verify with `agent --version`

### 4. Set Model to Claude Opus 4.6

The `agent` invocations require setting the default model to **Claude Opus 4.6**.

## Running Evals

### Full Automated Run

From the repo root:

```bash
bash evals/run_evals.sh
```

Before running, edit the two variables at the top of `run_evals.sh` to match your environment.

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

Each test case lives in `evals/<id>/` and must contain:

- `trace.json.jz` -- the profiling trace to analyze.
- `analysis_output_ref/` -- a reference analysis output to compare against (used by quality evals). Should include `standalone_analysis.md` and `perf_report_csvs/`.

### 2. Add a row to `unit_test_traces.csv`

The CSV has the following columns:

| Column | Description |
|---|---|
| `id` | Unique test case ID (e.g. `GEMM_0`) |
| `sub_category` | Category label (e.g. `GEMM`) |
| `trace_path` | Relative path to the trace JSON (e.g. `evals/GEMM_0/trace.json`) |
| `reference_dir` | Relative path to the reference output (e.g. `evals/GEMM_0/analysis_output_ref`) |
| `platform` | Hardware platform (e.g. `MI300X`) |

Example row:

```
GEMM_1,GEMM,evals/GEMM_1/trace.json.jz,evals/GEMM_1/analysis_output_ref,MI300X
```

## Eval Pipeline Summary

### Shell Script (`run_evals.sh`)

For each test case in `unit_test_traces.csv`, the script runs three phases:

1. **Phase 1 -- Standalone Analysis (serial):** Invokes the Standalone Analysis agent on the trace file. Output is written to `<trace_dir>/analysis_output/`.
2. **Phase 2 -- Workflow + Quality Evals (parallel):** Launches two `agent` sub-processes concurrently:
   - Workflow eval -- validates the analysis output structure and formatting.
   - Quality eval -- compares the analysis output against the reference.
3. **Phase 3 -- Merge Results:** Runs `eval_scripts/merge_results.py` to combine per-eval CSVs into a single `eval_summary.csv` for the test case.

### Eval Skills

Two Cursor agent skills in `.cursor/skills/` define the eval logic:

**workflow-eval** -- 12 evals total:
- Evals 1-7 (scripted): directory structure, required files, plot output. Run via `eval_scripts/workflow_scripted_evals.py`.
- Evals 8-12 (LLM-based): report template rendering, executive summary metrics, issue template formatting, hardware reference in appendix, and compute sub-agent findings structure.

**quality-eval** -- 3 evals total:
- Eval 1 (scripted): CSV value alignment between generated and reference outputs. Run via `eval_scripts/quality_scripted_evals.py`.
- Evals 2-3 (LLM-based): semantic comparison of compute issue titles and content (performance numbers, shapes, efficiency) against the reference report.

## Results

Results are written to `evals/results/<id>/` and include:

- `analysis.log` -- stdout/stderr from the standalone analysis run.
- `workflow_eval.log` / `workflow_eval_results.csv` -- workflow eval output.
- `quality_eval.log` / `quality_eval_results.csv` -- quality eval output.
- `eval_summary.csv` -- merged results for the test case.

The aggregated summary across all test cases is at `evals/results/eval_summary.csv`.

## Repeatability Study

Because the pipeline uses LLMs, outputs are non-deterministic. The repeatability study runs each test case multiple times (default: 5) and aggregates pass rates across runs to surface flaky or inconsistent behavior. This **should be run** (ideally on a 'screen' to prevent disconnects) before merging changes to the agent. 

```bash
cd TraceLens-internal
bash evals/run_repeatability.sh
```

After the runs complete, analyze and aggregate the results by prompting Cursor:

```bash
python evals/utils/aggregate_repeatability.py
```

This produces three CSVs in `evals/utils/output/`:

- `pass_rate_summary.csv` -- per-trace, per-eval pass rates across all runs
- `aggregated_results.csv` -- full eval results for every run
- `stream_diagnostics.csv` -- per-run metadata (outcome, token usage, last step reached, whether the report was written)

**What to look for using Cursor:**

- **Overall pass rate** should not regress compared to the baseline on `main`.
- **Per-eval consistency** -- an eval that passes in 3/5 runs indicates flaky behavior that needs investigation.
- **Stream diagnostics regressions** -- runs where the report was not written or the last step reached regressed (e.g., stuck at Step 7 instead of reaching Step 11) signal a reliability problem.
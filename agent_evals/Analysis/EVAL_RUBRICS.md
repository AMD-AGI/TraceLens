<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# TraceLens Eval Rubrics

Single-source reference for every eval check, its pass/fail criteria, tolerances,
root-cause categories, and where the implementation lives.

> **Canonical location:** `agent_evals/Analysis/EVAL_RUBRICS.md`
> Update this file whenever an eval is added, removed, or its criteria change.

---

## Overview

| Range | Type | Count | Engine |
|-------|------|:-----:|--------|
| workflow\_eval\_1–8 | Scripted (file/dir existence) | 8 | `workflow_scripted_evals.py` |
| workflow\_eval\_9 | Scripted (per-header) | 6 sub-indices | `workflow_scripted_evals.py` |
| workflow\_eval\_10 | Scripted (per-row + CSV cross-check) | 5 sub-indices | `workflow_scripted_evals.py` |
| workflow\_eval\_11 | Scripted (per-P-item) | Dynamic | `workflow_scripted_evals.py` |
| workflow\_eval\_12 | LLM (multi-dimensional scoring) | 1 | `workflow-llm-eval.md` |
| workflow\_eval\_13 | Scripted (per-category) | Dynamic | `workflow_scripted_evals.py` |
| workflow\_eval\_14 | Scripted (per-field) | 4 sub-indices | `workflow_scripted_evals.py` |
| quality\_eval\_1 | Scripted (CSV alignment) | 1 | `quality_scripted_evals.py` |
| quality\_eval\_2 | LLM (multi-dimensional scoring) | 1 | `quality-llm-eval.md` |
| quality\_eval\_3 | LLM (multi-dimensional scoring) | 1 | `quality-llm-eval.md` |

**Total:** 13 scripted evals (expanding to ~30+ sub-indices) + 3 LLM evals = 16 logical evals.

---

## Pre-Check Gates

Before any eval runs, hard gates are checked in order. If any trips, **all**
workflow evals are set to FAIL with `root_cause=pipeline`.

| Gate | Condition | Implemented in |
|------|-----------|----------------|
| 1 | `output_dir` does not exist | `workflow_scripted_evals.py` |
| 2 | `analysis.md` missing or < 100 bytes | `workflow_scripted_evals.py` |
| 3 | `analysis.md` is garbled (> 50% non-ASCII) | `workflow_scripted_evals.py` |
| 4 | `analysis.md` contains raw JSON instead of markdown | `workflow_scripted_evals.py` |
| 5 | `output_dir` or `reference_dir` missing | `quality_scripted_evals.py` |
| 6 | `perf_report_csvs/` directory missing (generated or reference) | `quality_scripted_evals.py` |

---

## Workflow Evals 1–8: File & Directory Existence

All binary PASS/FAIL. Root cause is always `pipeline`.

| Index | Summary | Pass Criteria | Key Details |
|-------|---------|---------------|-------------|
| `workflow_eval_1` | Directory structure created | `metadata/`, `category_data/`, `system_findings/`, `category_findings/` all exist | — |
| `workflow_eval_2` | Metadata files exist on disk | Every `metadata_file` listed in `category_manifest.json` exists on disk | Requires manifest |
| `workflow_eval_3` | Model info JSON exists and valid | `metadata/model_info.json` exists, is valid JSON, has keys `{model, architecture, scale, precision}`, none empty | — |
| `workflow_eval_4` | Unified perf report exists | `perf_report_csvs/unified_perf_summary.csv` exists | — |
| `workflow_eval_5` | Tree data files exist on disk | Every `tree_data_file` listed in `category_manifest.json` exists on disk | Requires manifest |
| `workflow_eval_6` | Categorical findings .md files exist | For every category in manifest (except `cpu_idle` when idle ≤ 15%), the corresponding `*_findings.md` exists in the correct tier directory | — |
| `workflow_eval_7` | All findings correctly placed | No findings file is in the wrong tier directory (`system_findings/` vs `category_findings/`) | — |
| `workflow_eval_8` | Plot generated on disk | `perf_improvement.png` exists **OR** no kernel tuning recommendations exist (legitimately skipped) | Checks `plot_data.json` and `*_metrics.json` for zero recommendations |

---

## Workflow Eval 9: Report Template Rendering

**Type:** Scripted, per-header. **Root cause on fail:** `template`.

Checks `analysis.md` for required `##` section headers and a metrics table.

| Sub-index | Pass Criteria |
|-----------|---------------|
| `workflow_eval_9_executive_summary` | `## Executive Summary` header present (regex: `^## Executive Summary`) |
| `workflow_eval_9_compute` | `## Compute Kernel Optimizations` header present |
| `workflow_eval_9_system` | `## System-Level Optimizations` header present |
| `workflow_eval_9_detailed` | `## Detailed Analysis` header present |
| `workflow_eval_9_appendix` | `## Appendix` header present |
| `workflow_eval_9_metrics_table` | Executive Summary section contains at least one markdown table row (`\|.*\|`) |

---

## Workflow Eval 10: Executive Summary Metrics Table

**Type:** Scripted, per-row + CSV cross-check. **Root cause on fail:** `template`.

Parses the metrics table in `## Executive Summary` and optionally cross-checks
numeric values against `perf_report_csvs/gpu_timeline.csv`.

| Sub-index | Row Labels (synonyms) | CSV Cross-Check | Tolerance |
|-----------|----------------------|-----------------|-----------|
| `workflow_eval_10_total_time` | "Total Compute Time", "Total Time" | — | — |
| `workflow_eval_10_compute_pct` | "Computation", "Compute %", "Compute" | `computation_time` percent | ±1.0% absolute |
| `workflow_eval_10_idle_pct` | "Idle Time", "Idle %", "Idle" | `idle_time` percent | ±1.0% absolute |
| `workflow_eval_10_comm_pct` | "Exposed Communication", "Exposed Communication %" | — | — |
| `workflow_eval_10_bottleneck` | "Top Bottleneck Category", "Top Bottleneck" | — | — |

**Pass:** Row found in table. If CSV cross-check applies, report value within tolerance of CSV value.
**Fail:** Row not found, or numeric value exceeds tolerance.

---

## Workflow Eval 11: Issue Template Rendering

**Type:** Scripted, per-P-item. **Root cause on fail:** `template`.

Finds every priority item (`### ...P{N}:` headers) and checks for required bold fields.

| Section | Required Fields | Example Sub-index |
|---------|----------------|-------------------|
| `## Compute Kernel Optimizations` | `**Insight**` or `**Issue**`, `**Action**`, `**Impact**` | `workflow_eval_11_compute_P1` |
| `## System-Level Optimizations` | `**Insight**` or `**Issue**`, `**Action**` (no Impact) | `workflow_eval_11_system_P1` |

**Pass:** All required bold fields present in the P-item block.
**Fail:** Any required field missing. Details list which fields are absent.

**Note:** Either `**Insight**` or `**Issue**` is acceptable — both are valid as the first field.

---

## Workflow Eval 12: Hardware Reference in Appendix

**Type:** LLM, multi-dimensional weighted scoring. **Root cause on fail:** `template`.

**Implementation:** `agent_evals/Analysis/.cursor/skills/workflow-llm-eval.md`

Checks the `## Appendix` section for hardware reference values:
- Platform name (e.g., MI300X)
- Peak HBM BW value (e.g., 5.3 TB/s)
- At least one Peak MAF value (e.g., 708 TFLOPS)

### Scoring Dimensions

| Dimension | Weight | Scale |
|-----------|--------|-------|
| **correctness** | 50% | 10 = all values correct and plausible, 7 = present but imprecise, 0 = wrong/invented |
| **completeness** | 50% | 10 = all 3 items present, 7 = 2 of 3, 4 = 1 of 3, 0 = none |

### Pass/Fail

```
overall_score = correctness × 0.50 + completeness × 0.50
```

- **FAIL** if correctness = 0
- **FAIL** if overall\_score < 7.0
- **PASS** otherwise

**Details format:** `correctness=N/10 completeness=N/10 overall=N.N | <explanation>`

---

## Workflow Eval 13: Model Identification in Report

**Type:** Scripted, per-field. **Root cause on fail:** `template`.

Reads `metadata/model_info.json` and checks that each field value appears
(case-insensitive substring match) in the `## Appendix` section of the report.

| Sub-index | JSON Field | Pass Criteria |
|-----------|-----------|---------------|
| `workflow_eval_13_model` | `model` | Value appears in Appendix text |
| `workflow_eval_13_architecture` | `architecture` | Value appears in Appendix text |
| `workflow_eval_13_scale` | `scale` | Value appears in Appendix text |
| `workflow_eval_13_precision` | `precision` | Value appears in Appendix text |

**Special case:** If a field value is empty or `"Cannot be inferred from trace"`,
the check is **skipped (PASS)** with a note in details.

---

## Quality Eval 1: Perf Report CSV Alignment

**Type:** Scripted. **Root cause on fail:** `data`.

**Implementation:** `quality_scripted_evals.py`

Compares every CSV in `perf_report_csvs/` against the reference directory.

| Check | Criteria |
|-------|----------|
| File presence | Every reference CSV must exist in generated output |
| Column presence | All non-optional reference columns must exist (optional prefixes: `Pct Roofline`, `Roofline Time`) |
| Row count | Generated row count must match reference |
| Numeric columns | Relative diff ≤ 1% **AND** absolute diff ≤ 0.05 (both must exceed to fail) |
| String columns | Exact match after stripping numpy type wrappers (e.g., `np.int64(135)` → `135`) |

**Pass:** All reference CSVs match within tolerances.
**Fail:** Details list up to 5 mismatches.

---

## Quality Eval 2: Compute Issue Title Alignment

**Type:** LLM, multi-dimensional weighted scoring. **Root cause on fail:** `data`.

**Implementation:** `agent_evals/Analysis/.cursor/skills/quality-llm-eval.md`

Compares P-item titles in the generated report against the reference report.
This is a **semantic** comparison — not a string match.

### Scoring Dimensions

| Dimension | Weight | Scale |
|-----------|--------|-------|
| **correctness** | 40% | 10 = same bottleneck identified (e.g., both say "GEMM low CU occupancy"), 7 = same category and root cause but framed differently (e.g., ref says "low CU occupancy due to small tiles" vs generated says "insufficient tile coverage on CUs"), 4 = same category but different root cause (e.g., ref says "occupancy" vs generated says "memory bandwidth"), 0 = wrong category or fabricated bottleneck |
| **completeness** | 30% | 10 = all reference P-items matched, 7 = one miss, 4 = several misses, 0 = most unmatched |
| **precision** | 30% | 10 = same priority level, 7 = off by 1, 4 = off by 2+, 0 = completely different |

### Pass/Fail

```
overall_score = correctness × 0.40 + completeness × 0.30 + precision × 0.30
```

- **FAIL** if correctness = 0
- **FAIL** if overall\_score < 7.0
- **PASS** otherwise

**Details format:** `correctness=N/10 completeness=N/10 precision=N/10 overall=N.N | <explanation>`

---

## Quality Eval 3: Compute Issue Content Alignment

**Type:** LLM, multi-dimensional weighted scoring. **Root cause on fail:** `data`.

**Implementation:** `agent_evals/Analysis/.cursor/skills/quality-llm-eval.md`

For each matched compute P-item pair (from eval 2), compares content values.
**Only compares Compute Kernel P-items** — System-Level P-items are skipped entirely.

### Value Tolerances

| Value Type | Tolerance | Special Rules |
|-----------|-----------|---------------|
| Performance numbers (kernel time, efficiency %) | 5% relative | — |
| Shapes (matrix dimensions, batch sizes) | Exact match | — |
| Gap to roofline (efficiency %) | 5% relative | — |
| Estimated savings (ms) | 5% relative | If savings < 5 ms, accept ≤ 1 ms absolute diff |
| Non-numeric gains ("Not quantifiable") | Semantic match | Mismatch only if one side numeric, other not |

### Scoring Dimensions

| Dimension | Weight | Scale |
|-----------|--------|-------|
| **correctness** | 40% | 10 = all values match, 7 = minor discrepancies, 0 = wrong values |
| **completeness** | 30% | 10 = all fields present (time, efficiency, shapes, gains), 7 = one missing, 0 = most missing |
| **precision** | 30% | See precision scale below |

**Precision scale** (based on the [Value Tolerances](#value-tolerances) above):

| Score | Meaning |
|-------|---------|
| 10 | All numeric values within 5% relative tolerance |
| 7 | Most values within 5%; 1–2 values between 5–10% off |
| 4 | Several values 5–10% off, or 1–2 values >10% off |
| 0 | Most values >10% off, or systematically wrong |

### Pass/Fail

```
overall_score = correctness × 0.40 + completeness × 0.30 + precision × 0.30
```

- **FAIL** if correctness = 0
- **FAIL** if overall\_score < 7.0
- **PASS** otherwise

**Details format:** `correctness=N/10 completeness=N/10 precision=N/10 overall=N.N | <explanation>`

---

## Stability Classification

After multiple repeated runs, the aggregation script (`aggregate_repeatability.py`)
classifies each (trace, eval\_index) pair:

| Classification | Criteria |
|----------------|----------|
| `STABLE_PASS` | 100% of runs passed |
| `FLAKY_PASS` | > 50% passed but not all |
| `FLAKY_FAIL` | > 0% passed but ≤ 50% |
| `STABLE_FAIL` | 0% of runs passed |

---

## Root Cause Categories

Every FAIL row includes a `root_cause` field for triage:

| Root Cause | Meaning | Typical Fix |
|-----------|---------|-------------|
| `pipeline` | Analysis pipeline didn't produce expected output | Re-run pipeline step or fix subagent |
| `template` | Report formatting / structure issue | Fix report assembly logic |
| `data` | Generated data doesn't match reference | Regenerate golden refs or fix data pipeline |

---

## CSV Output Schema

All eval results use the same 7-column schema:

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
| `root_cause` | `pipeline`, `template`, `data`, or empty if PASS |
| `recommended_fix` | Actionable fix suggestion, or empty if PASS |

---
name: Quality Eval
description: Run quality evals (pythonic + LLM) comparing standalone analysis output against a reference and produce a results CSV.
triggers:
  - quality eval
  - run quality eval
tools:
  - terminal
  - file_read
  - file_write
---

# Quality Eval

Evaluate the standalone analysis output for quality by comparing against a reference report. Checks both deterministic CSV alignment and semantic content alignment.

**MANDATORY: This skill has 3 steps. ALL steps must be completed. The final CSV must contain exactly 3 rows (evals 13-15). Do NOT stop after Step 1.**

## Inputs

When triggered, the prompt will specify:
- **output_dir**: path to the generated `analysis_output/` directory
- **reference_dir**: path to the `analysis_output_ref/` directory
- **results_path**: path to write the results CSV

## Step 1: Run Pythonic Evals (13) -- produces 1 row

Execute the pythonic quality evals script:

```bash
ssh <node> "docker exec -w <repo_root> tracelens_evals python3 eval_scripts/quality_evals.py \
    --output-dir <output_dir> \
    --reference-dir <reference_dir> \
    --results <results_path>"
```

Read the resulting CSV. This covers eval 13 only -- 1 of 3 rows. You MUST continue to Step 2.

## Step 2: Run LLM-Based Evals (14-15) -- produces 2 more rows

Read the files listed below, then evaluate EACH of the 2 LLM evals below. Append BOTH results to the CSV from Step 1.

### Files to Read

- `<output_dir>/standalone_analysis.md` (generated report)
- `<reference_dir>/standalone_analysis.md` (reference report)
- `<output_dir>/category_findings/*_findings.md` (generated findings)
- `<reference_dir>/category_findings/*_findings.md` (reference findings)

### quality_eval_2: Compute Issue Title Alignment

**Category:** Quality
**Issue Summary:** Compute Issue Title Alignment

Compare the P-item titles in the generated `standalone_analysis.md` against the reference report. For each P-item in the reference (lines matching `### ... P1:`, `### ... P2:`, etc.):

- Check if the generated report identifies the **same bottleneck** at the same or similar priority level
- This is a **semantic** comparison, not a string match.
- A P-item in the reference that has no semantic match in the generated report is a miss

**PASS** if every reference P-item has a semantic match in the generated report. **FAIL** listing unmatched reference P-items.

### quality_eval_3: Compute Issue Content Alignment

**Category:** Quality
**Issue Summary:** Compute Issue Content Alignment

For each matched P-item pair (from eval 14), compare the content values:

- **Performance numbers**: kernel time, efficiency percentage, achieved bandwidth/TFLOPS. Numeric tolerance: 5% relative difference.
- **Shapes**: matrix dimensions, batch sizes. Must match exactly.
- **Gap to roofline**: the efficiency percentage or fraction of peak. Tolerance: 5%.
- **Estimated gain**: savings in ms or percentage improvement. Tolerance: 5%.

Check the category findings files for detailed values if the top-level report summarizes them.

**PASS** if all matched P-items have aligned content within tolerance. **FAIL** listing specific mismatches with expected vs actual values.

## Step 3: Write Merged Results -- final CSV must have 3 rows

Read the CSV from Step 1 (1 row). Append the 2 LLM eval results (quality_eval_2 and quality_eval_3) as new rows. Write the merged CSV back to the same path. Use `quality_eval_2`, `quality_eval_3` as the `index` values.

**Output CSV must have exactly these 5 columns:** `index,category,issue_summary,result,details`

Do not add any other columns (no test_case_id, no priority, no eval_type). Each LLM eval row must use the exact index, category, and issue_summary specified above.

**Verify the final CSV has exactly 3 data rows (quality_eval_1 through quality_eval_3) before finishing.**

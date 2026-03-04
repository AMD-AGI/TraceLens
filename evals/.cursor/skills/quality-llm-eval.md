---
name: Quality LLM Eval
description: Run LLM-based quality evals comparing standalone analysis output against a reference and produce a results CSV.
triggers:
  - quality LLM eval
  - run quality LLM eval
tools:
  - file_read
  - file_write
---

# Quality LLM Eval

Evaluate the standalone analysis output for semantic quality by comparing against a reference report.

## Inputs

When triggered, the prompt will specify:
- **output_dir**: path to the generated `analysis_output/` directory
- **reference_dir**: path to the `analysis_output_ref/` directory
- **results_path**: path to write the results CSV

## Files to Read

Read ALL of these before evaluating:

- `<output_dir>/standalone_analysis.md` (generated report)
- `<reference_dir>/standalone_analysis.md` (reference report)
- `<output_dir>/category_findings/*_findings.md` (generated findings)
- `<reference_dir>/category_findings/*_findings.md` (reference findings)

## Evals

Evaluate BOTH checks below. Write BOTH rows to the results CSV.

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

For each matched P-item pair (from eval 2), compare the content values:

- **Performance numbers**: kernel time, efficiency percentage, achieved bandwidth/TFLOPS. Numeric tolerance: 5% relative difference.
- **Shapes**: matrix dimensions, batch sizes. Must match exactly.
- **Gap to roofline**: the efficiency percentage or fraction of peak. Tolerance: 5%.
- **Estimated gain**: savings in ms or percentage improvement. Tolerance: 5%.

Check the category findings files for detailed values if the top-level report summarizes them.

**PASS** if all matched P-items have aligned content within tolerance. **FAIL** listing specific mismatches with expected vs actual values.

## Output

Write a CSV to the specified `results_path` with exactly these 5 columns and 2 data rows:

`index,category,issue_summary,result,details`

Use `quality_eval_2` and `quality_eval_3` as the `index` values. Do not add any other columns.

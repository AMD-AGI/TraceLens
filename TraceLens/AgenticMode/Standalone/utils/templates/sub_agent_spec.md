<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Sub-Agent Findings Specification

Canonical reference for the output that sub-agents write into their findings
files. The orchestrator extracts these sections when composing the final
`standalone_analysis.md` report.

> **Usage:** Link here from every `*-analyzer.md` instead of duplicating the
> schema. Replace `<category>` with the actual category name
---

## Orchestrator-consumed sections

Every findings file must end with these two sections, in this order:

1. `## Recommendations`
2. `## Detailed Analysis`

Applies to both tiers (compute → `category_findings/`, system → `system_findings/`). Agents may include any other sections (Overview, Operations Breakdown, Key Bottlenecks, …) before them — those are agent-internal and not parsed by the orchestrator.

---

## Recommendations

Each P-item maps 1:1 to a `## Detailed Analysis` reasoning candidate at the same rank.

```markdown
### P1: <Brief Title> (<Library>)            <!-- (<Library>) only on compute tier -->
**Insight**: [1 sentence — what's wrong]
**Action**: [1-2 sentences — what to do]
**Impact**: [~X.X–Y.Y ms savings (X.X–Y.Y% of E2E), OR "Not quantifiable from trace data"]   <!-- compute tier only -->
```

- **Compute tier**: include all three fields. Pull `**Impact**` from `impact_estimates` in the metrics JSON.
- **System tier**: omit `**Impact**` and the `(<Library>)` title suffix.
- **Field labels are exact** — always `**Insight**`, `**Action**`, `**Impact**`.
- **`(<Library>)` suffix**: comma-separated list of unique non-null `library` values across the bottleneck operations. If all are `null`, omit the parenthetical entirely.

---

## Detailed Analysis block schema

Each candidate block lives inside a `## Detailed Analysis` section. It starts
with an HTML comment and an `####` heading:

```markdown
## Detailed Analysis

<!-- reasoning-candidate tier=<compute|system> rank=<N> -->
#### <insight_title>
**Identification:** …

**Data:** …

**Reasoning for Slowdown:** …

**Resolution:** …

**Impact estimate:** …
```

### HTML comment fields

| Field | Values | Meaning |
|-------|--------|---------|
| `tier` | `compute` \| `system` | Must match the findings directory (`category_findings/` → compute, `system_findings/` → system). |
| `rank` | Integer ≥ 1 | Agent-local priority within this file (1 = highest). |

### Required labels

The five labels below must appear **in this order**, each on its own line with a
blank line between them. The validator checks for these as substring matches.

| Label | Purpose |
|-------|---------|
| `**Identification:**` | Why these operations were flagged. Body text must be plain language — JSON keys, dotted paths, and internal variable names belong **only** in the closing `(source: \`artifact\` → \`keys\`)` parenthetical (artifact + keys backticked, e.g. `(source: \`gemm_metrics.json\` → \`operations[].efficiency.efficiency_percent\` < 70)`). When any flagged op has a non-null `library` (e.g. `Tensile`, `CK`, `AITER`, `Triton`, `rocBLAS`), state the backend in prose (e.g. "These operations use the **Tensile** backend.") and include `operations[].library` in the `(source:)` parenthetical. |
| `**Data:**` | **Compute** (`tier=compute`): trace-grounded kernel breakdown table (see § Operations Table Schema). Omit columns that have no data. **System** (`tier=system`): **must not** include kernel breakdown tables. Default columns: `Metric \| Value \| Flagged`. |
| `**Reasoning for Slowdown:**` | Why the workload is slow *as the trace shows*: low % of roofline, low arithmetic intensity, unfused patterns, etc. **Forbidden:** micro-architecture speculation (bank conflicts, L1 miss rates, etc.). |
| `**Resolution:**` | **Why** the suggested optimization helps — not merely restating *what* to do. Must align with the P-item **Action** on the card. **Forbidden tautologies:** Do not restate the roofline definition (e.g. "raising bandwidth toward the roofline reduces kernel time"). Instead, explain the **mechanism** (e.g. "fusion eliminates the intermediate write-back, cutting bytes moved per invocation in half"). If the mechanism is not inferable from the trace, state only the action. |
| `**Impact estimate:**` | Rendered from `metadata/*.json → impact_estimates[]`. Quantifiable entries use the three-bullet format (see § Impact estimate rendering); non-quantifiable entries use: `Impact estimate is not quantifiable from trace data.` |

### Sentence quality

- Each sentence should convey **one main idea**. Do not chain independent
  observations with em-dashes, semicolons, or "while" bridges. Avoid run-on
  sentences.

---

## Operations Table Schema (compute tier)

Standard column schema for operations breakdown tables and the `**Data:**` table
inside `## Detailed Analysis` blocks.

```markdown
| Operation | Args | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|-----------|------|-----------|------|-------|------------|------------|-------|
```

**Column mappings** (source: `metrics['operations']`):
- **Operation**: `operations[i].name`. Bare op name only — shape/dtype go in Args. Allowed suffix: `(decode)`/`(prefill)` to disambiguate the same op at multiple shapes.
- **Args**: `operations[i].args`. Pre-rendered shape/dtype string, already joined with `<br>` — paste verbatim, do not reformat or re-join. Omit the column if every row is missing this field.
- **Time (ms)**: `operations[i].time_ms` — kernel time in milliseconds.
- **%E2E**: `operations[i].percent_of_total` — kernel time as % of E2E GPU time. `null` ⇒ omit the column. (`percent_of_category` is still in the JSON for screening thresholds but no longer rendered.)
- **Count**: `operations[i].count` — total invocations, not unique signatures.
- **FLOPS/Byte**: `operations[i].efficiency.flops_per_byte`
- **Efficiency**: `operations[i].efficiency.efficiency_percent`, formatted by `bound_type`:
  - `compute-bound`: `X.XX% of Y TFLOPS` (Y = `resolved_peak_maf`)
  - `memory-bound`: `X.XX% of Y TB/s` (Y = `resolved_peak_hbm_bw`)
- **Bound**: `operations[i].efficiency.bound_type` + `-bound` suffix (e.g., `memory-bound`). Must reflect compute/memory bound type — never use `classification.gemm_type` or similar.

Agents may add extra columns when needed (e.g. `Sub-Category` in the generic-op analyzer).

---

## Peak Reference (compute tier)

When citing peak performance for a bottleneck, select the correct peak based on
`operations[i].efficiency.bound_type`:
- **compute-bound**: Use `operations[i].efficiency.resolved_peak_maf` (TFLOPS).
  Report achieved TFLOPS/s vs peak TFLOPS.
- **memory-bound**: Use `operations[i].efficiency.resolved_peak_hbm_bw` (TB/s).
  Report achieved TB/s vs peak TB/s.

Do not look up peaks independently from the metadata dict.

---

## Impact Estimates

Sub-agents write an **`impact_estimates` array** into
`metadata/<category>_metadata.json`, then render impact bullets in their
`## Detailed Analysis` block.

### Rules

- Exactly **one array element per `<!-- reasoning-candidate -->` block** in the
  same findings file, ordered by `rank` ascending.
- Each entry sums only the operations that its candidate covers (not the entire
  category).
- No insights → empty array `[]`.
- **Compute tier only:** use `kernel_tuning` estimates from the pre-computed
  metrics JSON (`savings_ms_low`–`savings_ms_high`, `e2e_pct_low`–`e2e_pct_high`).
  Do NOT manually estimate algorithmic, fusion, or system savings.
- **Confidence:** `high` = clear, measurable gap to peak; `medium` = likely
  opportunity but outcome depends on implementation; `low` = rough estimate.

### JSON schema

Element shape (quantifiable):

```json
{
  "impact_estimates": [
    {
      "low_e2e_ms": <number>,
      "high_e2e_ms": <number>,
      "low_e2e_percent": <number>,
      "high_e2e_percent": <number>,
      "quantifiable": true
    }
  ]
}
```

Non-quantifiable entries use `null` values with `"quantifiable": false`:

```json
{
  "impact_estimates": [
    {
      "low_e2e_ms": null,
      "high_e2e_ms": null,
      "low_e2e_percent": null,
      "high_e2e_percent": null,
      "quantifiable": false
    }
  ]
}
```

### Rendering in `## Detailed Analysis`

**Quantifiable:**

```markdown
- Low end (75% roofline target): <low_e2e_ms> ms savings (<low_e2e_percent>% E2E)
- High end (100% roofline target): <high_e2e_ms> ms savings (<high_e2e_percent>% E2E)
- Range: <low_e2e_ms>–<high_e2e_ms> ms (<low_e2e_percent>–<high_e2e_percent>% E2E)
```

**Non-quantifiable:** `Impact estimate is not quantifiable from trace data.`

### Write impact estimates to metadata

```bash
# Compute tier
<prefix> python3 -c "from TraceLens.AgenticMode.Standalone.utils.report_utils import write_impact_estimates; write_impact_estimates('<output_dir>', '<category>', 'compute')"

# System tier
<prefix> python3 -c "from TraceLens.AgenticMode.Standalone.utils.report_utils import write_impact_estimates; write_impact_estimates('<output_dir>', '<category>', 'system')"
```

---

## Validate findings (required before returning)

After writing the findings file and impact estimates, run the programmatic
validator. This replaces the previous manual self-check.

```bash
<prefix> python3 -c "
import sys
from TraceLens.AgenticMode.Standalone.utils.validation_utils import validate_findings_file
passed, errors = validate_findings_file(sys.argv[1], sys.argv[2])
if not passed:
    print('FAIL:')
    for e in errors:
        print('  - ' + e)
    sys.exit(1)
print('PASS: Findings file is valid')
" '<output_dir>/<subdir>/<category>_findings.md' '<tier>'
```

Where `<tier>` is `compute` or `system` and `<subdir>` is `category_findings`
or `system_findings` respectively.

**If validation fails (exit code 1):**

1. Read the FAIL output to identify structural issues
2. Fix the findings file — add missing sections, correct P-item labels, etc.
3. Re-run validation
4. Maximum 2 retry attempts. If still failing, return with a warning

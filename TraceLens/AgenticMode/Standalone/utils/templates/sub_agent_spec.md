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

The sections below must appear **at the end** of every findings file, in the
order shown. Agents may include any additional sections before these (Overview,
Operations Breakdown, Key Bottlenecks, etc.) — those are agent-internal and not
parsed by the orchestrator.

**Compute tier** (`category_findings/`):

1. `## Recommendations`
2. `## Detailed Analysis`

**System tier** (`system_findings/`):

1. `## Recommendations`
2. `## Detailed Analysis`

---

## Recommendations

Each `## Recommendations` P-item must map 1:1 to a `## Detailed Analysis`
reasoning candidate at the same rank.

**Compute tier:**

```markdown
### P1: <Brief Title> (<Library>)
**Insight**: [1 sentence — what's wrong]
**Action**: [1-2 sentences — what to do]
**Impact**: [~X.X–Y.Y ms savings (X.X–Y.Y% of E2E) from impact_estimates in metrics JSON, OR "Not quantifiable from trace data"]
```

**Library parenthetical:** Collect the unique non-null `library` values from
the bottleneck operations in the metrics JSON. Append them comma-separated
in parentheses after the title. All `library: null`: omit the parenthetical
entirely.

**System tier:**

```markdown
### P1: <Brief Title>
**Insight**: [1 sentence — what's wrong]
**Action**: [1-2 sentences — what to do]
```

System-tier recommendations have no **Impact** field. Always use exactly `**Insight**`, `**Action**`, `**Impact**` as field labels.

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
| `**Identification:**` | How these operations were deemed an optimization opportunity. Body text must use **plain language only** — no JSON keys, dotted paths, or internal variable names. **Must** end with a `(source: <artifact> → <keys>)` parenthetical as the final text. Artifact names and keys must be wrapped in backticks (e.g. `(source: \`gemm_metrics.json\` → \`operations[].efficiency.efficiency_percent\` < 70)`). All JSON keys and internal variable names belong **exclusively** inside this parenthetical. When the metrics JSON includes a non-null `library` field for an operation (e.g. `"Tensile"`, `"CK"`, `"AITER"`, `"Triton"`, `"rocBLAS"`), **always** state which library the operations use (e.g. "These operations use the **Tensile** backend.") and include `operations[].library` in the `(source:)` parenthetical. |
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
| Operation | Kernel time (ms) | % of category | Count | FLOPS/Byte | Efficiency | Bound |
|-----------|-----------------|---------------|-------|------------|------------|-------|
```

**Column mappings** (source: `metrics['operations']`):
- **Kernel time (ms)**: `operations[i].time_ms`
- **% of category**: `operations[i].percent_of_category`
- **Count**: `operations[i].count` (total invocations, not unique signatures)
- **FLOPS/Byte**: `operations[i].efficiency.flops_per_byte`
- **Efficiency**: `operations[i].efficiency.efficiency_percent` formatted by bound type:
  - `compute-bound`: `X.XX% of Y TFLOPS` (Y = `resolved_peak_maf`)
  - `memory-bound`: `X.XX% of Y TB/s` (Y = `resolved_peak_hbm_bw`)
- **Bound**: `operations[i].efficiency.bound_type` with a `-bound` suffix (e.g., `memory-bound`, `compute-bound`)

Agents may extend the table with additional columns (e.g., `Sub-Category` for
the generic-op analyzer). Do NOT use `classification.gemm_type` or similar
internal fields for the Bound column — it must reflect compute/memory bound type.

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

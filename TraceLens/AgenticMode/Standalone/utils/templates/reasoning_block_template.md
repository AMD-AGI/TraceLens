<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Reasoning Block Specification

Canonical reference for the `## Detailed Analysis` candidate blocks that sub-agents
write into their findings files. The orchestrator pastes these blocks into the final
`## Detailed Analysis` section of `standalone_analysis.md`, renaming headings with
P-numbers, priority icons, and HTML anchors.

**Link here from every `*-analyzer.md`** instead of duplicating the schema.

---

## Block schema

Each candidate block lives inside a `## Detailed Analysis` section at the end of the
findings file. It starts with an HTML comment and an `####` heading:

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

## Required labels

The five labels below must appear **in this order**, each on its own line with a blank line between them. The validator checks for these as substring matches.

| Label | Purpose |
|-------|---------|
| `**Identification:**` | How these operations were deemed an optimization opportunity. Body text must use **plain language only** — no JSON keys, dotted paths, or internal variable names. **Must** end with a `(source: <artifact> → <keys>)` parenthetical as the final text. Artifact names and keys must be wrapped in backticks (e.g. `(source: \`gemm_metrics.json\` → \`operations[].efficiency.efficiency_percent\` < 70)`). All JSON keys and internal variable names belong **exclusively** inside this parenthetical. When the metrics JSON includes a non-null `library` field for an operation (e.g. `"Tensile"`, `"CK"`, `"AITER"`, `"Triton"`, `"rocBLAS"`), **always** state which library the operations use (e.g. "These operations use the **Tensile** backend.") and include `operations[].library` in the `(source:)` parenthetical. |
| `**Data:**` | **Compute** (`tier=compute`): trace-grounded kernel breakdown table. Default columns: `Operation \| Kernel time (ms) \| % of category \| Count \| FLOPS/Byte \| Efficiency \| Bound`. Omit columns that have no data. **System** (`tier=system`): **must not** include kernel breakdown tables. Default columns: `Metric \| Value \| Flagged`. |
| `**Reasoning for Slowdown:**` | Why the workload is slow *as the trace shows*: low % of roofline, low arithmetic intensity, unfused patterns, etc. **Forbidden:** micro-architecture speculation (bank conflicts, L1 miss rates, etc.). |
| `**Resolution:**` | **Why** the suggested optimization helps — not merely restating *what* to do. Must align with the P-item **Action** on the card. **Forbidden tautologies:** Do not restate the roofline definition (e.g. "raising bandwidth toward the roofline reduces kernel time"). Instead, explain the **mechanism** (e.g. "fusion eliminates the intermediate write-back, cutting bytes moved per invocation in half"). If the mechanism is not inferable from the trace, state only the action. |
| `**Impact estimate:**` | Rendered from `metadata/*.json → impact_estimates[]`. Quantifiable entries use the three-bullet format (see below); non-quantifiable entries use: `Impact estimate is not quantifiable from trace data.` |

### Sentence quality

- Each sentence should convey **one main idea**. Do not chain independent observations with em-dashes, semicolons, or "while" bridges. Avoid run-on sentences.

---

## Impact estimates in metadata JSON

Sub-agents write an **`impact_estimates` array** into `metadata/<category>_metadata.json`.

**Rules:**
- Exactly **one array element per `<!-- reasoning-candidate -->` block** in the same findings file, ordered by `rank` ascending.
- Each entry sums only the operations that its candidate covers (not the entire category).
- No insights → empty array `[]`.

Element shape:

```json
{
  "impact_estimates": [
    {
      "low_e2e_ms": <low_e2e_ms>,
      "high_e2e_ms": <high_e2e_ms>,
      "low_e2e_percent": <low_e2e_percent>,
      "high_e2e_percent": <high_e2e_percent>,
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

### Rendering format

Quantifiable:

```markdown
- Low end (75% roofline target): <low_e2e_ms> ms savings (<low_e2e_percent>% E2E)
- High end (100% roofline target): <high_e2e_ms> ms savings (<high_e2e_percent>% E2E)
- Range: <low_e2e_ms>–<high_e2e_ms> ms (<low_e2e_percent>–<high_e2e_percent>% E2E)
```

Non-quantifiable: `Impact estimate is not quantifiable from trace data.`

Each sub-agent renders the impact bullets directly in its `## Detailed Analysis` block after writing `impact_estimates` to metadata.

---

## Kernel fusion variant

Kernel fusion blocks use **three** labels: **Identification**, **Data**, **Impact estimate**.

```markdown
<!-- reasoning-candidate tier=system rank=<N> -->
#### <candidate_title>
**Identification:** …

**Data:** …

**Impact estimate:** …
```

### Required labels (fusion)

| Label | Purpose |
|-------|---------|
| `**Identification:**` | How the fusion candidate was surfaced. Plain language body ending with `(source: \`fusion_candidates.json\` → <keys>)`. |
| `**Data:**` | Kernels table with columns: `Kernel \| Type \| Duration (us) \| Perf model`. |
| `**Impact estimate:**` | Rendered from `kernel_fusion_metrics.json → impact_estimates[]`. Uses four-bullet format (see below); non-quantifiable entries use the standard single-line form. |

### Fusion impact rendering format

Quantifiable:

```markdown
- Low end (75% roofline): <savings_ms_low> ms savings (<e2e_pct_low>% E2E)
- High end (100% roofline): <savings_ms_high> ms savings (<e2e_pct_high>% E2E)
- Coverage: <modeled_kernel_count> of <kernel_count> kernels modelled
- Fusion pattern: <bound_type>-bound, <fusion_type>
```

When partial coverage, append to Coverage: `(<unmodeled_count> kernel(s) use measured trace time)`.

Non-quantifiable: `Impact estimate is not quantifiable from trace data.`

---

## Self-check (for sub-agents)

Before writing findings, verify these items:

1. **Card–Detailed Analysis consistency:** Every claim, number, and operation in the P-item card (Insight / Action / Impact) must be consistent with the corresponding Detailed Analysis block. Do not introduce numbers or claims in one that are absent from the other.
2. **Data table columns** match the tier defaults (compute: `Operation | Kernel time (ms) | % of category | Count | FLOPS/Byte | Efficiency | Bound`; fusion: `Kernel | Type | Duration (us) | Perf model`; system: `Metric | Value | Flagged`).
3. **Final report slice** From `## Detailed Analysis` through the next `##`, include `### Compute Kernel Insights` and `### System-Level Insights`.
4. **P-block body:** five required labels in order; each label starts at the beginning of a line.
5. **Identification** before `(source:`: no JSON-path-shaped backticks except op names used as prose.
6. Outside Identification’s `(source: …)` and in **Data** / **Reasoning** / **Resolution**: no bare internal/tooling field names; use plain language or keep paths inside `(source: …)`.
7. **Impact:** use the exact Low / High / Range line prefixes from [Rendering format](#rendering-format), or include `not quantifiable`.

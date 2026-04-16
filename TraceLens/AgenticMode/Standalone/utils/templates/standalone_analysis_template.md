<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

<!--
=== FORMATTING RULES (for the agent filling in this template) ===

=== MODE SELECTION ===
This template supports two modes determined by `comparison_scope`:
  - **standalone**: Single-trace roofline analysis (default). Use sections marked STANDALONE.
  - **comparative**: Two-trace analysis (Trace 1 =  primary, Trace 2 = target). Use sections marked COMPARATIVE.
When filling in this template, select the block matching the active `comparison_scope` for each
section that has STANDALONE / COMPARATIVE variants. Delete the unused variant.

=== COMPARATIVE TERMINOLOGY ===
  - **Trace 1** =  trace (primary). **Trace 2** = trace (target/comparison).
  - Impact semantics: standalone uses roofline gap (75–100% of peak); comparative uses
    trace 2 kernel time as the optimization target (gap = trace1 time − trace2 time).
  - Efficiency semantics: standalone = % of roofline; comparative = 100 × (trace2 kernel time) / (trace1 kernel time).

=== GENERAL RULES ===
1. Warnings section: Only include if there were errors or high-variance operations; omit entirely if all succeeded and no variance flags.
2. Executive Summary: Max ~20 lines.
3. Performance plot: The {{PERF_PLOT}} placeholder is replaced by Step 10.2 with a base64-embedded
   PNG data URI (![Performance Improvement](data:image/png;base64,...)). This makes the report
   fully portable.
   - Standalone: plot shows kernel tuning potential with 75–100% roofline potential on both panes
     (left: E2E latency error bars from savings_ms_low/savings_ms_high, baseline bar has no error
     bar; right: throughput uncertainty band from same range, no uncertainty at baseline).
   - Comparative: plot shows gap-to-target per category (trace 1 vs trace 2 kernel times).
   If the plot was not generated (Step 9.5 failed), the placeholder is removed.
   - `{{COMPARATIVE_CUMULATIVE_PLOT}}` (comparative only): Step 10.3 embeds a cumulative kernel-time figure from TraceDiff-enriched `unified_perf_summary.csv`; removed if generation is skipped.
4. Compute Kernel Optimizations: P1-P3+ from category subagent findings.
   - Standalone: Impact estimates show a range (75–100% of roofline target), e.g. "~X.X–Y.Y ms savings (X.X–Y.Y% of E2E)".
   - Comparative: Impact estimates show gap to target trace, e.g. "~X.X ms gap to target (Y.Y% of E2E)".
5. System-Level Optimizations: If all system-level analyses report no actionable issues
   (NONE/N/A severity), use a single "✅ No system-level bottlenecks detected" summary instead of
   P1/P2/P3 recommendations. Only generate numbered priorities when at least one actionable issue
   exists (number sequentially from P1, including CPU/Idle first if invoked).
6. Each section is independently composable -- can be shared standalone.
7. All three tiers (Compute, Kernel Fusion, System) use separate sequential P1/P2/P3 numbering (no gaps).
8. Priority icons are assigned by PRIORITY NUMBER, not severity:
   - Compute Kernel: 🔴 P1 → 🟡 P2 → 🟢 P3 → 🟢 P4 ...
   - Kernel Fusion: icon by confidence (🔴 high → 🟡 medium → 🟢 low), not priority number
   - System-Level: 🔴 P1 → 🟡 P2 → 🟢 P3 → 🟢 P4 ... (only when actionable issues exist)
9. Field labels — each section uses EXACTLY these labels:

   OPTIMIZATION CARDS (§Compute Kernel Optimizations, §Kernel Fusion, §System-Level):
   - Compute Kernel P-items: **Insight** / **Action** / **Impact**
   - Kernel Fusion P-items:  **Insight** / **Action** / **Impact** / **Confidence**
   - System-Level P-items:   **Insight** / **Action**

   DETAILED ANALYSIS (§Detailed Analysis only):
   - Compute / System blocks: **Identification:** / **Data:** / **Reasoning for Slowdown:** / **Resolution:** / **Impact estimate:**
   - Kernel Fusion blocks:    **Identification:** / **Data:** / **Impact estimate:**

10. Detailed Analysis: three subsections (`### Compute Kernel Insights`, `### Kernel Fusion Insights`, `### System-Level Insights`) with `#### 🔴/🟡/🟢 Pn: <Brief Title>` blocks matching card titles and order.
11. Model and appendix: Use `model_info["model"]` from `metadata/model_info.json` for the
    report title (fall back to "Workload" if "Cannot be inferred from trace"). Fill Appendix
    **Model Architecture** with the raw `model`, `architecture`, `scale`, `precision` values.
12. Library parenthetical: Compute Kernel card titles and Detailed Analysis headings must include
    the library name(s) in parentheses when present in the sub-agent findings. Omit when no
    library is identified. System-Level and Kernel Fusion titles do NOT include a library
    parenthetical.
-->

<!-- === STANDALONE title === -->
# <Model> - <Platform> Standalone Analysis

<!-- === COMPARATIVE title === -->
# <Model> - Comparative Analysis: <Platform1> vs <Platform2>

## Executive Summary

<!-- === STANDALONE Executive Summary === -->
[1 paragraph overview + key metrics table]

<!-- MANDATORY: This table must contain exactly these 5 rows:
     Total Time | Compute % | Idle % | Exposed Communication % | Top Bottleneck Category -->
| Metric | Value |
|--------|-------|
| Total Time | X ms |
| Compute % | Y% |
| Idle % | Z% |
| Exposed Communication % | W% |
| Top Bottleneck Category | Category (V%) |

<!-- === COMPARATIVE Executive Summary === -->
[1 paragraph comparative overview: summarize which trace is faster overall, by how much, and the dominant gap categories]

| Metric | Trace 1 - (<Platform1>) | Trace 2 - (<Platform2>) | Difference |
|--------|----------------------------|-------------------------------|------------|
| Total Time | X ms | Y ms | +/-Z ms (+/-W%) |
| Compute % | X% | Y% | +/-Z% |
| Idle % | X% | Y% | +/-Z% |
| Exposed Communication % | X% | Y% | +/-Z% |
| Top Bottleneck Category | Category (X%) | Category (Y%) | — |

{{PERF_PLOT}}

{{COMPARATIVE_CUMULATIVE_PLOT}}

## Warnings

**Include this section ONLY if any subagent failed OR any operation has high_variance: true in *_metrics.json:**

<!-- Subagent failures (if any): -->
The following analyses could not be completed due to script failures:

| Analysis | Tier | Error Summary |
|----------|------|---------------|
| <name> | System / Compute Kernel | <brief error description> |

These are excluded from the recommendations below.

<!-- Data quality warnings (if any operation has high_variance: true in *_metrics.json): -->
**Data Quality:** The following operations have unreliable kernel time measurements (CoV > 1.0, indicating extreme variance across instances — likely a profiler timing artifact):

| Operation | Category | CoV | Reported Time (ms) |
|-----------|----------|-----|-------------------|
| <name> | <category> | X.X | Y.Y |

---

## Compute Kernel Optimizations

Findings from per-category kernel analysis (GEMM, SDPA, elementwise, etc.).
Summaries of recommendations from Step 7 sub-agents, focused on individual kernel efficiency.

### Top Operations

<!-- === STANDALONE Top Operations === -->
Use **% of computation time** (not % of total trace time) so readers can see each category's share of the GPU compute budget. Compute the denominator as `total_time_ms * computation_time_percent / 100` from the manifest `gpu_utilization`. For **Ops** column use `operation_count` from `category_data/<category>_metrics.json` (total invocations). The table is category-level with columns: Rank | Category | Time (ms) | % of Compute Time | Ops | Potential improvement (time, E2E %). The last column shows both the time range and E2E % range when kernel_tuning estimates exist (e.g. "~770–9801 ms (1.4–17.3%)"); use "—" when no estimates.


| Rank | Category | Time (ms) | % of Compute Time | Ops | Potential improvement (time, E2E %) |
|------|----------|-----------|-------------------|-----|-------------------------------------|
| 1 | ... | ... | ... | ... | ~X–Y ms (X–Y%) or — |

<!-- === COMPARATIVE Top Operations === -->
<!-- Use **% of computation time** based on Trace 1. "Difference" = Trace 2 category time − Trace 1 category time; negative means Trace 2 is faster. Use "—" when trace 2 has no matching category data. -->

| Rank | Category | Trace 1 Time (ms) | Trace 2 Time (ms) | % of Compute Time | Ops | Difference (ms) |
|------|----------|-------------------|-------------------|-------------------|-----|-----------------|
| 1 | ... | ... | ... | ... | ... | +/-X.X or — |

<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->
<!-- Use category-specific Action text: SDPA (fwd/bwd) → tile/block tuning, Flash Attention backend; GEMM → fusion with adjacent ops, tile sizes, library; elementwise → fuse with adjacent ops; other → fusion where applicable, tile sizes. Do NOT suggest "kernel fusion" for SDPA (already fused). -->

### 🔴 P1: <Brief Title> (<Library>)

**Insight**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - category-appropriate: GEMM fusion/tile/library; SDPA tile/backend; elementwise fusion; etc.]

<!-- Standalone Impact -->
**Impact**: [~X.X–Y.Y ms savings (X.X–Y.Y% of E2E) from closing efficiency gaps to 75–100% of roofline (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]
<!-- Comparative Impact -->
**Impact**: [~X.X ms gap to target (Y.Y% of E2E), OR "Gap not quantifiable from trace data"]

→ *See [Detailed Analysis: Compute kernel insights > P1](#detailed-analysis-compute-p1) for details*

---

### 🟡 P2: <Brief Title> (<Library>)

**Insight**: [1 sentence]

**Action**: [1-2 sentences]
<!-- Standalone Impact -->
**Impact**: [~X.X–Y.Y ms savings (X.X–Y.Y% of E2E) from closing efficiency gaps to 75–100% of roofline (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]
<!-- Comparative Impact -->
**Impact**: [~X.X ms gap to target (Y.Y% of E2E), OR "Gap not quantifiable from trace data"]

→ *See [Detailed Analysis: Compute kernel insights > P2](#detailed-analysis-compute-p2) for details*

---

### 🟢 P3: <Brief Title> (<Library>)

**Insight**: [1 sentence]

**Action**: [1-2 sentences]
<!-- Standalone Impact -->
**Impact**: [~X.X–Y.Y ms savings (X.X–Y.Y% of E2E) from closing efficiency gaps to 75–100% of roofline (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]
<!-- Comparative Impact -->
**Impact**: [~X.X ms gap to target (Y.Y% of E2E), OR "Gap not quantifiable from trace data"]

→ *See [Detailed Analysis: Compute kernel insights > P3](#detailed-analysis-compute-p3) for details*

<!-- All additional P-items (P4, P5, ...) follow the same pattern with Detailed Analysis links: → *See [Detailed Analysis: Compute kernel insights > PN](#detailed-analysis-compute-pN) for details* -->

---

## Kernel Fusion Opportunities (Experimental)
<!-- === STANDALONE Kernel Fusion === -->
> **Note:** Kernel fusion analysis is experimental. Savings estimates use a roofline projection model (75-100% of peak) with 85% memory/compute pipeline overlap. Kernels without perf models use their measured trace time as-is. Candidates where fewer than 75% of kernels have perf models are not reported. Each finding shows both a **Confidence** (fusion pattern quality) and perf model coverage in the **Impact** line. Actual savings depend on implementation feasibility and interaction effects.
<!-- === COMPARATIVE Kernel Fusion === -->
> **Note:** Kernel fusion analysis is experimental. In comparative mode, fusion candidates are identified from Trace 1 only. Cross-trace fusion mapping is not yet supported. Savings estimates use a roofline model.

<!-- Populate from category_findings/kernel_fusion_findings.md if kernel_fusion category exists in manifest. -->
<!-- Each finding uses Insight / Action / Impact format, with Impact from kernel_fusion_metrics.json. -->
<!-- P1/P2/P3+ ordered by confidence then kernel time. -->
<!-- If no findings or kernel_fusion category not in manifest, show the message below. -->

No kernel fusion opportunities detected.

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

<!-- === COMPARATIVE system-level note === -->
<!-- In comparative mode, add this note immediately after the blockquote above: -->
<!-- > **Comparative note:** System-level analysis is performed on the primary trace (Trace 1) only. Cross-trace system-level comparison is not yet supported. -->

Findings from system-level analysis (GPU utilization, memory transfer patterns,
communication/compute overlap). These affect the GPU pipeline as a whole.

<!-- CONDITIONAL: If NO actionable system-level issues found (idle <= 15% and all multi-kernel assessments flagged: false), use Template A. -->
<!-- Otherwise, number priorities sequentially: CPU/Idle first (if idle > 15%), then multi-kernel issues by severity. -->
<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->
<!-- Title format: Descriptive name only. -->
<!-- System-level recommendations have NO **Impact** field -- impact is not quantifiable for system-level issues. -->

<!-- === TEMPLATE A: No actionable system-level issues === -->
<!-- Use this when idle <= 15% and all multi-kernel assessments have flagged: false -->

✅ No system-level bottlenecks detected. GPU activity breakdown shows X% computation, with negligible memcpy and communication overhead.

<!-- === TEMPLATE B: Actionable issues found === -->
<!-- Use this when idle > 15% or at least one multi-kernel assessment has flagged: true -->

### 🔴 P1: <CPU/Idle Title OR Multi-Kernel Issue Title>

**Insight**: [1-2 sentences - what's wrong]

**Action**: [1-2 sentences - what to do]

→ *See [Detailed Analysis: System-level insights > P1](#detailed-analysis-system-p1) for details*

---

### 🟡 P2: <Multi-Kernel Issue Title>

**Insight**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - what to do]

→ *See [Detailed Analysis: System-level insights > P2](#detailed-analysis-system-p2) for details*

---

### 🟢 P3: <Next Multi-Kernel Issue>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

→ *See [Detailed Analysis: System-level insights > P3](#detailed-analysis-system-p3) for details*

<!-- All additional system P-items follow the same pattern with Detailed Analysis links -->

---

## Detailed Analysis

<!-- Paste reasoning blocks from sub-agent findings, augment headings with P-numbers, icons, and HTML anchors. Everything else should be copied verbatim-->
<!-- Detailed Analysis labels per rule 9 — do not use these labels in optimization cards above -->
<!-- Impact estimate bullets are rendered by each sub-agent from metadata/*.json → impact_estimates (same source as card Impact). -->

### Compute Kernel Insights

<!-- One #### 🔴/🟡/🟢 Pn: <title> block per promoted compute P-item, in priority order. -->
<!-- Each block has an HTML anchor: <a id="detailed-analysis-compute-pN"></a> -->

<!-- === STANDALONE Compute Kernel Data table === -->

<a id="detailed-analysis-compute-p1"></a>
#### 🔴 P1: <Brief Title> (<Library>)
**Identification:**
**Data:**

| Operation | Kernel time (ms) | % of category | Count | FLOPS/Byte | Efficiency | Bound |
|-----------|-----------------|---------------|-------|------------|------------|-------|
| ...       | ...             | ...           | ...   | ...        | ...        | ...   |

**Reasoning for Slowdown:**
**Resolution:**
**Impact estimate:**

<!-- === COMPARATIVE Compute Kernel Data table === -->
<!-- Trace 1 ms = Kernel Time (µs)_sum / 1000. Trace 2 ms = Kernel Time (µs)_trace2_sum / 1000 when
     present; else delta_us + t1, or —. Count T1/T2 = operation_count / operation_count_trace2 when
     present. Difference (ms) = Trace 2 Time − Trace 1 Time (positive ⇒ more time on Trace 2), or —. -->

<a id="detailed-analysis-compute-p1"></a>
#### 🔴 P1: <Brief Title>
**Identification:** [1-2 sentences - How this opportunity was surfaced relative to the target trace. Must end with (source: <artifact> → <keys>).]
**Data:** [1 sentence summary of table]

| Operation | Trace 1 Time (ms) | Trace 2 Time (ms) | Count (T1/T2) | Difference (ms) | FLOPS/Byte (T1) | Bound (T1) |
|-----------|-------------------|-------------------|---------------|-----------------|-----------------|------------|
| ...       | ...               | ...               | .../...       | ...             | ...             | ...        |

**Reasoning for Slowdown:** [2-3 sentences - Why Trace 1 is slower than Trace 2 for these operations as the traces show. No micro-architecture speculation.]
**Resolution:** [1-2 sentences - Why the suggested optimization helps close the gap — not merely restating what to do.]
**Impact estimate:** [Rendered from metadata → impact_estimates]

### Kernel Fusion Insights

> **Note:** Kernel fusion analysis is experimental. Savings estimates use a roofline projection model (75-100% of peak) with 85% memory/compute pipeline overlap. Kernels without perf models use their measured trace time as-is. Actual savings depend on implementation feasibility and interaction effects.

<!-- Paste reasoning blocks from kernel_fusion_findings.md, ordered by confidence then kernel time (matching card order). -->
<!-- Each block uses three required labels: **Identification:**, **Data:**, **Impact estimate:** -->
<!-- If kernel_fusion category is not in the manifest or findings are empty, show "No fusion savings estimates available." -->

<a id="detailed-analysis-fusion-P1"></a>
#### 🔴/🟡/🟢 P1: <Candidate Name> (<time_ms> ms, <instance_count> instances)

**Identification:**

**Data:**

| Kernel | Type | Duration (us) | Perf model |
|--------|------|--------------|------------|
| <kernel name (truncated to ~60 chars)> | <type> | X.X | Yes/No |

**Impact estimate:**

<a id="detailed-analysis-fusion-P2"></a>
#### 🔴/🟡/🟢 P2: <Candidate Name> (<time_ms> ms, <instance_count> instances)

*Repeat the same Identification + Data + Impact estimate format for each candidate, with anchors `detailed-analysis-fusion-PN`.*

### System-Level Insights

<!-- One #### 🔴/🟡/🟢 Pn: <title> block per promoted system P-item, in priority order. -->
<!-- Each block has an HTML anchor: <a id="detailed-analysis-system-pN"></a> -->
<!-- System-level detailed analysis uses the same format for both standalone and comparative modes.
     In comparative mode, system-level analysis covers Trace 1 () only. -->

<a id="detailed-analysis-system-p1"></a>
#### 🔴 P1: <Brief Title>
**Identification:**
**Data:**
**Reasoning for Slowdown:**
**Resolution:**
**Impact estimate:**

---

## Appendix

### Model Architecture
- **Model**: <model>
- **Architecture**: <architecture>
- **Scale**: <scale>
- **Precision**: <precision>

### Hardware Reference

<!-- === STANDALONE Hardware Reference === -->
- **Platform**: <platform>
- **Peak HBM BW**: X TB/s
- **Peak MAF (BF16)**: Y TFLOPS
- **Peak MAF (FP8)**: Z TFLOPS (if supported)
- **Peak MAF (FP4)**: W TFLOPS (if supported)

<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

<!--
=== FORMATTING RULES (for the agent filling in this template) ===

1. Warnings section: Only include if there were errors or high-variance operations; omit entirely if all succeeded and no variance flags.
2. Executive Summary: Max ~20 lines.
3. Performance plot: The {{PERF_PLOT}} placeholder is replaced by Step 10.2 with a base64-embedded
   PNG data URI (![Performance Improvement](data:image/png;base64,...)). This makes the report
   fully portable. The plot shows kernel tuning potential only with 75–100% roofline potential on
   both panes (left: E2E latency error bars from savings_ms_low/savings_ms_high, baseline bar has
   no error bar; right: throughput uncertainty band from same range, no uncertainty at baseline).
   If the plot was not generated (Step 9.5 failed), the placeholder is removed.
4. Compute Kernel Optimizations: P1-P3+ from category subagent findings. Impact estimates show a
   range (75–100% of roofline target), e.g. "~X.X–Y.Y ms savings (X.X–Y.Y% of E2E)".
5. System-Level Optimizations: If all system-level analyses report no actionable issues
   (NONE/N/A severity), use a single "✅ No system-level bottlenecks detected" summary instead of
   P1/P2/P3 recommendations. Only generate numbered priorities when at least one actionable issue
   exists (number sequentially from P1, including CPU/Idle first if invoked).
6. Each section is independently composable -- can be shared standalone.
7. Compute and System tiers use separate sequential P1/P2/P3 numbering (no gaps).
8. Priority icons are assigned by PRIORITY NUMBER, not severity:
   - Compute Kernel: 🔴 P1 → 🟡 P2 → 🟢 P3 → 🟢 P4 ...
   - System-Level: 🔴 P1 → 🟡 P2 → 🟢 P3 → 🟢 P4 ... (only when actionable issues exist)
9. Detailed Analysis: Split into Compute Kernels and System-Level subsections. For compute
   categories with metrics, use a per-op table from *_metrics.json with columns:
   Operation | Kernel time (ms) | % of category | Count | FLOPS/Byte | Efficiency |
   Potential improvement (time, E2E %) (FLOPS/Byte from efficiency.flops_per_byte, "—" when null;
   improvement from impact_estimates, "—" when none). For categories with a CSV but no metrics
   (e.g. multi_tensor_apply), include a "Most expensive instances" table from the category CSV
   (top ops by kernel time). In System-Level, use explicit HTML anchors
   <a id="cpu-idle-time-analysis"></a> and <a id="multi-kernel-issues"></a> before the subsection
   headings so in-report links work in all renderers. Always include the Detailed Analysis:
   System-Level section with full metrics even when no actionable issues exist.
10. Model and appendix: Read `metadata/model_info.json`. For the report title and any **&lt;Model&gt;** placeholder used for display: use `model_info["model"]` when it is not "Cannot be inferred from trace"; otherwise use **"Workload"**. Fill the Appendix **Model Architecture** section with the raw `model`, `architecture`, `scale`, and `precision` values from that file (they may be "Cannot be inferred from trace").
11. No redundancy: Information appears in ONE place only.
12. Recommendations: Max ~10 lines PER recommendation. Use category-specific Action text
    (SDPA: tile/block, backend; GEMM: fusion, tile, library; elementwise: fuse with adjacent;
    do not suggest kernel fusion for SDPA).
-->

# <Model> - <Platform> Standalone Analysis

## Executive Summary
[1 paragraph overview + key metrics table]

| Metric | Value |
|--------|-------|
| Total Time | X ms |
| Compute % | Y% |
| Idle % | Z% |
| Exposed Communication % | W% |
| Top Bottleneck Category | Category (V%) |

{{PERF_PLOT}}

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

Use **% of computation time** (not % of total trace time) so readers can see each category's share of the GPU compute budget. Compute the denominator as `total_time_ms * computation_time_percent / 100` from the manifest `gpu_utilization`. The table is category-level with columns: Rank | Category | Time (ms) | % of Compute Time | Ops | Potential improvement (time, E2E %). The last column shows both the time range and E2E % range when kernel_tuning estimates exist (e.g. "~770–9801 ms (1.4–17.3%)"); use "—" when no estimates.

| Rank | Category | Time (ms) | % of Compute Time | Ops | Potential improvement (time, E2E %) |
|------|----------|-----------|-------------------|-----|-------------------------------------|
| 1 | ... | ... | ... | ... | ~X–Y ms (X–Y%) or — |

<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->
<!-- Use category-specific Action text: SDPA (fwd/bwd) → tile/block tuning, Flash Attention backend; GEMM → fusion with adjacent ops, tile sizes, library; elementwise → fuse with adjacent ops; other → fusion where applicable, tile sizes. Do NOT suggest "kernel fusion" for SDPA (already fused). -->

### 🔴 P1: <Brief Title>

**Insight**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - category-appropriate: GEMM fusion/tile/library; SDPA tile/backend; elementwise fusion; etc.]

**Impact**: [~X.X–Y.Y ms savings (X.X–Y.Y% of E2E) from closing efficiency gaps to 75–100% of roofline (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]

→ *See [Detailed Analysis: Compute Kernels > Section](#section-link) for details*

---

### 🟡 P2: <Brief Title>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [~X.X–Y.Y ms savings (X.X–Y.Y% of E2E) from closing efficiency gaps to 75–100% of roofline (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]

→ *See [Detailed Analysis: Compute Kernels > Section](#section-link) for details*

---

### 🟢 P3: <Brief Title>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [~X.X–Y.Y ms savings (X.X–Y.Y% of E2E) from closing efficiency gaps to 75–100% of roofline (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]

---

## Kernel Fusion Opportunities (Experimental)

> **Note:** Kernel fusion analysis is experimental. Savings estimates use a roofline projection model (75-100% of peak). Kernels without perf models use their measured trace time as-is. Candidates where fewer than 75% of kernels have perf models are not reported. Each finding shows both a **Confidence** (fusion pattern quality) and perf model coverage in the **Impact** line. Actual savings depend on implementation feasibility and interaction effects.

<!-- Populate from category_findings/kernel_fusion_findings.md if kernel_fusion category exists in manifest. -->
<!-- Each finding uses Insight / Action / Impact format, with Impact from kernel_fusion_metrics.json. -->
<!-- P1/P2/P3+ ordered by confidence then kernel time. -->
<!-- If no findings or kernel_fusion category not in manifest, show the message below. -->

No kernel fusion opportunities detected.

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

Findings from system-level analysis (GPU utilization, memory transfer patterns,
communication/compute overlap). These affect the GPU pipeline as a whole.

<!-- CONDITIONAL: If NO actionable system-level issues found (idle <= 15% and all multi-kernel assessments flagged: false), use Template A. -->
<!-- Otherwise, number priorities sequentially starting from P1. Include CPU/Idle only if idle > 15%. -->
<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->
<!-- Title format: Descriptive name only. -->
<!-- System-level recommendations have NO **Impact** field -- impact is not quantifiable for system-level issues. -->

<!-- === TEMPLATE A: No actionable system-level issues === -->
<!-- Use this when idle <= 15% and all multi-kernel assessments have flagged: false -->

✅ No system-level bottlenecks detected. GPU activity breakdown shows X% computation, with negligible memcpy and communication overhead. See [Detailed Analysis: System-Level](#detailed-analysis-system-level) for full metrics.

<!-- === TEMPLATE B: Actionable issues found === -->
<!-- Use this when idle > 15% or at least one multi-kernel assessment has flagged: true -->

### 🔴 P1: <CPU/Idle Title OR Multi-Kernel Issue Title>

**Insight**: [1-2 sentences - what's wrong]

**Action**: [1-2 sentences - what to do]

→ *See [Detailed Analysis: System-Level > CPU/Idle Time](#cpu-idle-time-analysis) for details* OR → *See [Detailed Analysis: System-Level > Multi-Kernel Issues](#multi-kernel-issues) for details*

<!-- Use explicit HTML anchors in Detailed Analysis: System-Level so links work in all renderers: <a id="cpu-idle-time-analysis"></a> before "### 1. CPU/Idle Time Analysis", <a id="multi-kernel-issues"></a> before "### 2. Multi-Kernel Issues". Link targets: #cpu-idle-time-analysis, #multi-kernel-issues. -->

---

### 🟡 P2: <Multi-Kernel Issue Title>

**Insight**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - what to do]

→ *See [Detailed Analysis: System-Level > Multi-Kernel Issues](#multi-kernel-issues) for details*

---

### 🟢 P3: <Next Multi-Kernel Issue>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

---

## Detailed Analysis: Compute Kernels

For each category, include total time, % of compute, average efficiency (if from metrics), and either:

- **Per-op table from `*_metrics.json`**: columns **Operation | Kernel time (ms) | % of category | Count | FLOPS/Byte | Efficiency | Potential improvement (time, E2E %)**. The FLOPS/Byte column shows arithmetic intensity from `operations[i].efficiency.flops_per_byte` (use "—" when null); this grounds the compute-bound vs memory-bound classification against the platform's ridge point (peak MAF / peak HBM BW). The Efficiency column shows `efficiency_percent` from the metrics JSON, formatted as `X.XX% of Y TFLOPS` for compute-bound ops (Y = `resolved_peak_maf`) or `X.XX% of Y TB/s` for memory-bound ops (Y = `resolved_peak_hbm_bw`). The last column shows both time range and E2E % range from `impact_estimates` when kernel_tuning estimates exist (e.g. "~635–2378 ms (1.12–4.19% E2E)"); use "—" when no estimates. Match impact rows to ops by `time_ms` (and operation name) from the same metrics file.

- **For categories with a CSV but no metrics** (e.g. **multi_tensor_apply**): a **Most expensive instances** table from the category CSV: top N rows by `Kernel Time (µs)_sum`, columns Operation | Kernel time (ms) | % of category | Count. (No Efficiency or Potential improvement columns when metrics are absent.)

### 1. <Operation Category> (X% of compute)
[Kernel breakdowns, per-op table with Efficiency and Potential improvement (time, E2E %) where available, or most expensive instances from CSV]

### 2. <Operation Category> (X% of compute)
[...]

---

## Detailed Analysis: Kernel Fusion

**REQUIRED: For each candidate in kernel_fusion_findings.md, include BOTH the Kernels table AND the Projection table below, sorted by savings descending. Do NOT summarize into a single table. If kernel_fusion category is not in the manifest or findings are empty, show "No fusion savings estimates available."**

### 1. <Candidate Name> (<time_ms> ms, <instance_count> instances)

**Kernels:**

| Kernel | Type | Duration (us) | Perf model |
|--------|------|--------------|------------|
| <kernel name (truncated to ~60 chars)> | <type> | X.X | Yes/No |

**Projection:**

| Metric | Value |
|--------|-------|
| Bound type | compute / memory |
| Fusion type | matrix_compute / memory_bound |
| Kernels modelled | M of N |
| Savings (low-mid-high) | X.XX - Y.YY - Z.ZZ ms |
| E2E impact | X.XX - Z.ZZ% |

### 2. <Candidate Name> (<time_ms> ms, <instance_count> instances)

*Repeat the same Kernels + Projection format for each candidate.*

---

## Detailed Analysis: System-Level

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

<a id="cpu-idle-time-analysis"></a>
### 1. CPU/Idle Time Analysis
[Full cpu_idle_findings.md content from system_findings/ or metrics table: total GPU time, computation %, exposed comm/memcpy, idle %]

<a id="multi-kernel-issues"></a>
### 2. Multi-Kernel Issues
[Full multi_kernel_findings.md content from system_findings/ or reference to category_data/multi_kernel_data.json]

---

## Appendix

### Model Architecture
- **Model**: <model>
- **Architecture**: <architecture>
- **Scale**: <scale>
- **Precision**: <precision>

### Hardware Reference
- **Platform**: <platform>
- **Peak HBM BW**: X TB/s
- **Peak MAF (BF16)**: Y TFLOPS
- **Peak MAF (FP8)**: Z TFLOPS (if supported)
- **Peak MAF (FP4)**: W TFLOPS (if supported)

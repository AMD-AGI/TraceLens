<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

<!--
=== FORMATTING RULES (for the agent filling in this template) ===

1. Warnings section: Only include if there were errors or high-variance operations; omit entirely if all succeeded and no variance flags.
2. Executive Summary: Max ~20 lines.
3. Performance plot: The {{PERF_PLOT}} placeholder is replaced by Step 11.3 with a base64-embedded
   PNG data URI (![Performance Breakdown](data:image/png;base64,...)) of a single horizontal stacked
   bar showing the run's compute-time breakdown by kernel category. The plot is purely descriptive
   (no error bars, no throughput cone, no savings estimates). If the plot was not generated
   (Step 10.3 / Step 11.2 failed), the placeholder is removed.
4. Compute Kernel Optimizations: One P-item per entry in `priority_data.json::findings[]`,
   numbered P1, P2, ... in `findings[]` order (already globally sorted by `impact_score`). The
   P-item Impact line uses the canonical mid `impact_score` value; low/high values appear only
   in Detailed Analysis. Cards join the corresponding sub-agent's Detailed Analysis block by
   `(findings[i].category, findings[i].category_rank)`.
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

# <Model> - <Platform> Standalone Analysis

## Executive Summary
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

One row per entry in `priority_data.json::priorities[]`, in array order (no manifest-sort, no extra rows). For row N (= `priorities[N-1]`): `Rank`/`Category` = `rank`/`display_name`; `Time (ms)` = matching `manifest.categories[].gpu_kernel_time_ms` (verbatim); `Ops` = matching `manifest.categories[].ops_count`; `% of Compute Time` = `Time (ms) / (gpu_utilization.total_time_ms * computation_time_percent / 100)`; trailer `low`/`high` = `priorities[N-1].impact_score_low`/`impact_score_high` (use `null` for `source: "manifest_fallback"`). Wrap the whole block (header + separator + rows) in the `kind=top_ops` marker.

<!-- impact-begin kind=top_ops -->
| Rank | Category | Time (ms) | % of Compute Time | Ops |
|------|----------|-----------|-------------------|-----|
| 1 | ... | ... | ... | ... | <!-- top-ops-row low=<impact_score_low> high=<impact_score_high> -->
<!-- impact-end -->

<!-- Icon mapping by PRIORITY NUMBER (not severity): P1=🔴, P2=🟡, P3+=🟢 -->
<!-- One card per entry in priority_data.findings[] in array order. Title uses the entry's category and library; Action text is category-appropriate. Do NOT recommend "fuse the SDPA kernel" (already fused — defer upstream/downstream fusion to Kernel Fusion section). -->

### 🔴 P1: <Brief Title> (<Library>)

**Insight**: [1 sentence - what's wrong]

**Action**: [1-2 sentences - category-appropriate: GEMM fusion/tile/library; SDPA tile/backend; elementwise fusion; etc.]

<!-- impact-begin kind=p_item category=<priority_data.findings[0].category> low=<priority_data.findings[0].impact_score_low> mid=<priority_data.findings[0].impact_score> high=<priority_data.findings[0].impact_score_high> -->
**Impact**: [impact_score: X.X, OR "Not quantifiable from trace data"]
<!-- impact-end -->

→ *See [Detailed Analysis: Compute kernel insights > P1](#detailed-analysis-compute-p1) for details*

---

### 🟡 P2: <Brief Title> (<Library>)

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

<!-- impact-begin kind=p_item category=<priority_data.findings[1].category> low=<priority_data.findings[1].impact_score_low> mid=<priority_data.findings[1].impact_score> high=<priority_data.findings[1].impact_score_high> -->
**Impact**: [impact_score: X.X, OR "Not quantifiable from trace data"]
<!-- impact-end -->

→ *See [Detailed Analysis: Compute kernel insights > P2](#detailed-analysis-compute-p2) for details*

---

### 🟢 P3: <Brief Title> (<Library>)

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

<!-- impact-begin kind=p_item category=<priority_data.findings[2].category> low=<priority_data.findings[2].impact_score_low> mid=<priority_data.findings[2].impact_score> high=<priority_data.findings[2].impact_score_high> -->
**Impact**: [impact_score: X.X, OR "Not quantifiable from trace data"]
<!-- impact-end -->

→ *See [Detailed Analysis: Compute kernel insights > P3](#detailed-analysis-compute-p3) for details*

<!-- All additional P-items (P4, P5, ...) follow the same pattern, sourcing markers from priority_data.findings[N-1]. Detailed Analysis links: → *See [Detailed Analysis: Compute kernel insights > PN](#detailed-analysis-compute-pN) for details* -->

---

## Kernel Fusion Opportunities (Experimental)

> **Note:** Kernel fusion analysis is experimental. impact_score projections estimate the recoverable fraction of E2E with 85% memory/compute pipeline overlap. Kernels without perf models use their measured trace time as-is. Candidates where fewer than 75% of kernels have perf models are not reported. Actual recoverable time depends on implementation feasibility and interaction effects.

<!-- Populate from system_findings/kernel_fusion_findings.md if kernel_fusion category exists in manifest. -->
<!-- Each finding uses Insight / Action / Impact / Confidence format, with Impact from kernel_fusion_metrics.json. -->
<!-- P1/P2/P3+ ordered by confidence then kernel time. -->
<!-- Icon mapping by CONFIDENCE (not priority number): 🔴 high → 🟡 medium → 🟢 low. -->
<!-- If no findings or kernel_fusion category not in manifest, replace the cards below with: "No kernel fusion opportunities detected." -->

### 🔴 P1: <Candidate Name>

**Insight**: [1 sentence - what fusion pattern was detected]

**Action**: [1-2 sentences - which kernels to fuse and how]

<!-- impact-begin kind=p_item low=<kernel_fusion_metrics.impact_estimates[0].impact_score_low> mid=<kernel_fusion_metrics.impact_estimates[0].impact_score> high=<kernel_fusion_metrics.impact_estimates[0].impact_score_high> -->
**Impact**: [impact_score: X.X (perf-model coverage Y/Z kernels)]
<!-- impact-end -->

**Confidence**: [high / medium / low - fusion pattern quality]

→ *See [Detailed Analysis: Kernel fusion insights > P1](#detailed-analysis-fusion-P1) for details*

---

### 🟡 P2: <Candidate Name>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

<!-- impact-begin kind=p_item low=<kernel_fusion_metrics.impact_estimates[1].impact_score_low> mid=<kernel_fusion_metrics.impact_estimates[1].impact_score> high=<kernel_fusion_metrics.impact_estimates[1].impact_score_high> -->
**Impact**: [impact_score: X.X (perf-model coverage Y/Z kernels)]
<!-- impact-end -->

**Confidence**: [high / medium / low]

→ *See [Detailed Analysis: Kernel fusion insights > P2](#detailed-analysis-fusion-P2) for details*

---

### 🟢 P3: <Candidate Name>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

<!-- impact-begin kind=p_item low=<kernel_fusion_metrics.impact_estimates[2].impact_score_low> mid=<kernel_fusion_metrics.impact_estimates[2].impact_score> high=<kernel_fusion_metrics.impact_estimates[2].impact_score_high> -->
**Impact**: [impact_score: X.X (perf-model coverage Y/Z kernels)]
<!-- impact-end -->

**Confidence**: [high / medium / low]

→ *See [Detailed Analysis: Kernel fusion insights > P3](#detailed-analysis-fusion-P3) for details*

<!-- All additional fusion P-items (P4, P5, ...) follow the same pattern with Detailed Analysis links: → *See [Detailed Analysis: Kernel fusion insights > PN](#detailed-analysis-fusion-PN) for details* -->

---

## System-Level Optimizations

> **Note:** System-level analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

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

<!-- One #### 🔴/🟡/🟢 Pn: <title> block per entry in priority_data.findings[], in array order. -->
<!-- Source the body block from the sub-agent's findings.md by joining on (findings[i].category, findings[i].category_rank): the sub-agent emits its P-items ordered by intra-category rank, so its rank-N block becomes this report's PN where N matches the position in priority_data.findings[]. -->
<!-- Each block has an HTML anchor: <a id="detailed-analysis-compute-pN"></a> -->

<a id="detailed-analysis-compute-p1"></a>
#### 🔴 P1: <Brief Title> (<Library>)
**Identification:**
**Data:**

| Operation | Args |                 Kernel Path                          | Time (ms) | %E2E | Count | FLOPS/Byte | Efficiency | Bound |
|-----------|------|------------------------------------------------------|-----------|------|-------|------------|------------|-------|
| ...       | ...  | ...                                                  | ...       | ...  | ...   | ...        | ...        | ...   |

**Reasoning for Slowdown:**
**Resolution:**
**Impact estimate:**

### Kernel Fusion Insights

> **Note:** Kernel fusion analysis is experimental. impact_score projections estimate the recoverable fraction of E2E with 85% memory/compute pipeline overlap. Kernels without perf models use their measured trace time as-is. Actual recoverable time depends on implementation feasibility and interaction effects.

<!-- Paste reasoning blocks from kernel_fusion_findings.md, ordered by confidence then kernel time (matching card order). -->
<!-- Each block uses three required labels: **Identification:**, **Data:**, **Impact estimate:** -->
<!-- If kernel_fusion category is not in the manifest or findings are empty, show "No fusion impact estimates available." -->

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
- **Platform**: <platform>
- **Peak HBM BW**: X TB/s
- **Peak MAF (BF16)**: Y TFLOPS
- **Peak MAF (FP8)**: Z TFLOPS (if supported)
- **Peak MAF (FP4)**: W TFLOPS (if supported)

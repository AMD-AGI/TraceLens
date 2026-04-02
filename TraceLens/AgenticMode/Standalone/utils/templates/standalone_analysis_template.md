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
9. Detailed Analysis: A single `## Detailed Analysis` section contains
   `### Compute Kernel Insights` then `### System-Level Insights`, each with
   `#### 🔴/🟡/🟢 Pn: <Brief Title>` blocks matching the optimization card titles and order.
   Each block has five required labels in order: **Identification:**, **Data:**,
   **Reasoning for Slowdown:**, **Resolution:**, **Impact estimate:**. Compute Data uses
   trace-grounded kernel tables (FLOPS/Byte, Efficiency, Bound); System Data uses system
   evidence only (no kernel breakdown tables). Impact estimate is rendered from
   `metadata/*.json → impact_estimates`. P-item cards link to Detailed Analysis anchors
   (e.g. `#detailed-analysis-compute-p1`).
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

→ *See [Detailed Analysis: Compute kernel insights > P1](#detailed-analysis-compute-p1) for details*

---

### 🟡 P2: <Brief Title>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [~X.X–Y.Y ms savings (X.X–Y.Y% of E2E) from closing efficiency gaps to 75–100% of roofline (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]

→ *See [Detailed Analysis: Compute kernel insights > P2](#detailed-analysis-compute-p2) for details*

---

### 🟢 P3: <Brief Title>

**Insight**: [1 sentence]

**Action**: [1-2 sentences]

**Impact**: [~X.X–Y.Y ms savings (X.X–Y.Y% of E2E) from closing efficiency gaps to 75–100% of roofline (pre-computed), OR "Not quantifiable from trace data" if no kernel_tuning estimates]

→ *See [Detailed Analysis: Compute kernel insights > P3](#detailed-analysis-compute-p3) for details*

<!-- All additional P-items (P4, P5, ...) follow the same pattern with Detailed Analysis links: → *See [Detailed Analysis: Compute kernel insights > PN](#detailed-analysis-compute-pN) for details* -->

---

## Kernel Fusion Opportunities (Experimental)

> **Note:** Kernel fusion analysis is experimental. The Compute Kernel section above analyzes the individual constituent kernels of these fusion candidates independently. Joint analysis, where fusion candidate's performance is also estimated, is planned for a future release. Opportunities are ranked by confidence level. Actual savings depend on implementation feasibility and interaction effects.

<!-- Populate from system_findings/kernel_fusion_findings.md if kernel_fusion category exists in manifest. -->
<!-- Each finding uses Insight / Action format (no Impact -- detection only). -->
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

<!-- Paste reasoning blocks from sub-agent findings, renaming headings with P-numbers, icons, and HTML anchors. Everything else should be copied verbatim-->
<!-- Each P-block uses five required labels: **Identification:**, **Data:**, **Reasoning for Slowdown:**, **Resolution:**, **Impact estimate:** -->
<!-- Impact estimate bullets are rendered by each sub-agent from metadata/*.json → impact_estimates (same source as card Impact). -->

### Compute Kernel Insights

<!-- One #### 🔴/🟡/🟢 Pn: <title> block per promoted compute P-item, in priority order. -->
<!-- Each block has an HTML anchor: <a id="detailed-analysis-compute-pN"></a> -->

<a id="detailed-analysis-compute-p1"></a>
#### 🔴 P1: <Brief Title>
**Identification:** [1-2 sentences - How this opportunity was surfaced. Must end with (source: <artifact> → <keys>).]
**Data:** [1 sentence summary of table]

| Operation | Kernel time (ms) | % of category | Count | FLOPS/Byte | Efficiency | Bound |
|-----------|-----------------|---------------|-------|------------|------------|-------|
| ...       | ...             | ...           | ...   | ...        | ...        | ...   |

**Reasoning for Slowdown:** [2-3 sentences - Why the workload is slow as the trace shows. No micro-architecture speculation.]
**Resolution:** [1-2 sentences - Why the suggested optimization helps — not merely restating what to do.]
**Impact estimate:** [Rendered from metadata → impact_estimates]

### System-Level Insights

<!-- One #### 🔴/🟡/🟢 Pn: <title> block per promoted system P-item, in priority order. -->
<!-- Each block has an HTML anchor: <a id="detailed-analysis-system-pN"></a> -->

<a id="detailed-analysis-system-p1"></a>
#### 🔴 P1: <Brief Title>
**Identification:** [1-2 sentences - How this opportunity was surfaced. Must end with (source: <artifact> → <keys>).]
**Data:** [1-2 sentences - System-level trace evidence — no kernel breakdown tables.]
**Reasoning for Slowdown:** [2-3 sentences - Why the workload is slow as the trace shows. No micro-architecture speculation.]
**Resolution:** [1-2 sentences - Why the suggested optimization helps — not merely restating what to do.]
**Impact estimate:** [Rendered from metadata → impact_estimates]

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

<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Multi-Kernel Issue Analysis Findings

> **Note:** This analysis is exploratory. The patterns and recommendations below are under active development and may be refined as system-level analysis matures.

**Status**: SUCCESS  
**Analysis Tier**: System-Level  
**Comparison scope**: comparative

## Summary

| Metric | Value | Flagged |
|--------|-------|---------|
| Total GPU time (analysis) | 517.21 ms | — |
| Total memory-copy events | 273 | false |
| D2H transfers | 0 (0 ms) | false |
| H2D transfers | 0 (0 ms) | false |
| Exposed collective communication | 0% of total | false |
| Compute / collective overlap | N/A (no collective communication time) | false |

## Memory Copy Analysis

### D2H (device-to-host) transfers

- **Count**: 0 transfers  
- **Total time**: 0 ms (0% of total GPU time)  
- **Flagged**: false  
- **Interpretation**: No device-to-host memory copies appear in the multi-kernel memory-copy breakdown for this trace window, so there is no evidence here of repeated host readback or synchronization-driven staging visible to this categorization.

### H2D (host-to-device) transfers

- **Count**: 0 transfers  
- **Total time**: 0 ms (0% of total GPU time)  
- **Flagged**: false  
- **Interpretation**: No host-to-device copies are recorded in the categorized memory-copy summary for this capture.

### Device-internal copies (D2D)

- **Count**: 273 events  
- **Total time**: ~1.03 ms (~0.2% of total GPU time of ~517.21 ms)  
- **Flagged**: false  
- **Interpretation**: Short device-internal copies account for all memory-copy–class events in the summary. Their aggregate duration is a small fraction of total GPU span and does not meet system-level thresholds focused on host-visible directions (D2H/H2D).

## Communication Blocking Analysis

### Collective communication blocking compute

- **Exposed collective communication time**: 0 ms (0% of total)  
- **Total collective communication time**: 0 ms  
- **Flagged**: false  

No collective communication events appear in the multi-kernel summary for this trace segment, so exposed communication blocking compute is not observable in this view.

### Compute / collective communication overlap

- **Overlap ratio**: Not computed (insufficient collective communication data; `comm_overlap_ratio` is null)  
- **Flagged**: false  

With no collective communication time recorded, the overlap heuristic does not apply. This alone does not indicate poor pipelining between compute and communication.

## Detected Patterns

No cross-cutting multi-kernel patterns were detected (`patterns_detected` is empty).

## Recommendations

- **System P1**  
  - **Insight:** Memory-copy pressure in host-visible directions (D2H/H2D) is absent in this snapshot; device-internal copy time is negligible versus total GPU span.  
  - **Action:** No memcpy-focused remediation is indicated at the system level for this window; continue routine monitoring if workload or capture scope changes.

- **System P2**  
  - **Insight:** Collective communication is not present in the multi-kernel data, so blocking and overlap cannot be assessed.  
  - **Action:** For distributed or multi-device workloads, capture a profiling window that includes collective phases so exposed communication and compute/collective overlap can be evaluated.

- **System P3**  
  - **Insight:** Cross-validation against a unified GPU timeline CSV was skipped because `perf_report_csvs/gpu_timeline.csv` was missing from this output layout.  
  - **Action:** When timeline reconciliation is required, ensure the comparative pipeline emits or links a primary `gpu_timeline.csv` (or equivalent) under the expected path, or align the script’s path with the available exports (e.g. trace-specific CSV folders).

## Technical Details

### Top collective communication operations

| Rank | Operation | Duration (µs) | Stream |
|------|-----------|---------------|--------|
| — | (none) | — | — |

### Memory copy breakdown

| Direction | Count | Total Time (ms) | Avg Size (bytes) | Severity |
|-----------|-------|-----------------|------------------|----------|
| D2D | 273 | ~1.03 | ~1,878 | Below threshold |
| D2H | 0 | — | — | Not observed |
| H2D | 0 | — | — | Not observed |

**Cross-validation**: Skipped — the multi-kernel script expected `perf_report_csvs/gpu_timeline.csv`, which was not present in this output directory. Metrics could not be reconciled against a single unified timeline export from this layout.

**GPU utilization (manifest / metadata)**: Total GPU span ~517.21 ms; computation ~96.95%; exposed memory copy ~0.20%; exposed collective communication 0%; idle ~2.85% (per `category_manifest.json` / `multi_kernel_metadata.json`).

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Estimated Improvement (E2E %) | Confidence |
|---------------|------|----------------------|-------------------------------|------------|

## Detailed Analysis

#### Single-GPU window with no host-direction copy or collective pressure

**Identification:** System-level multi-kernel review of memory movement, collective communication exposure, and compute/collective overlap for the analyzed GPU window (pre-computed `multi_kernel_data.json` / `multi_kernel_metrics.json`).

**Data:**

| Metric | Value | Flagged |
|--------|-------|---------|
| Total GPU time (analysis) | 517.21 ms | — |
| Memory copy time (aggregate) | ~1.03 ms (~0.2% of total) | false |
| D2H / H2D (directional) | 0 / 0 | false |
| Collective communication time | 0 ms | false |
| Exposed collective communication | 0% of total | false |
| `patterns_detected` | (empty) | — |
| Cross-validation vs. GPU timeline | Skipped (missing CSV) | — |

**Reasoning:** The preponderance of GPU time is attributed to computation. Memory-copy events are exclusively device-internal (D2D), short in aggregate, and below host-direction concern thresholds. No collective communication is present, so blocking of compute by communication and compute/collective overlap cannot be evaluated beyond “not applicable.” No automated cross-cutting patterns were emitted.

**Resolution:** No multi-kernel remediation is required from this slice. Regenerate or point the pipeline at a `gpu_timeline.csv` (or equivalent) when cross-validation against the perf report timeline is needed. For distributed workloads, include collective activity in the capture to enable overlap and blocking assessments.

**Impact estimate:** System-level multi-kernel analysis does not produce per-issue impact rows; the Impact Summary table above has no data rows by policy.

Per system-level analysis policy, no `reasoning-candidate` blocks are emitted when there are zero flagged system-level multi-kernel issues.

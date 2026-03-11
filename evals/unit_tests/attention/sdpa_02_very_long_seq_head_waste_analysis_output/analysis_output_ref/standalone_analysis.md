# TraceLens Standalone Analysis Report

**Trace:** `sdpa_02_very_long_seq_head_waste.json`
**Platform:** MI300X
**Total GPU Time:** 1,123.2 ms
**Analysis Date:** 2026-03-09

---

## Executive Summary

This trace profiles a **Flash Attention forward pass** with a very long sequence length (65,536) and a non-power-of-2 head dimension (d=192). The GPU is fully utilized (100% compute, ~0% idle), so the bottleneck is entirely within the attention kernel itself.

The single operation — `aten::_scaled_dot_product_flash_attention` — achieves **34.31% of peak BF16 compute** (242.93 TFLOPS vs 708 TFLOPS peak on MI300X). The primary cause is **tile waste from the non-power-of-2 head dimension**: d=192 is padded to 256 in the flash attention tiling scheme, wasting ~25% of compute on padding.

**Estimated potential savings: up to 737.8 ms per trace window** if head dimension waste is eliminated.

---

## GPU Utilization Breakdown

| Metric | Value |
|--------|-------|
| Computation | 100.0% |
| Idle | ~0.0% (0.0004 ms) |
| Communication | 0.0% |
| MemCpy | 0.0% |

**Assessment:** GPU utilization is optimal. No system-level bottlenecks were detected. There is no idle time, no memory copy overhead, and no communication overhead. All optimization opportunities are at the compute kernel level.

---

## Prioritized Recommendations

### P1: Use Power-of-2 Head Dimension

| Field | Value |
|-------|-------|
| Category | SDPA Forward (Flash Attention) |
| Operation | `aten::_scaled_dot_product_flash_attention` |
| Current Efficiency | 34.31% of peak BF16 MAF |
| Issue | Head dimension d=192 is not a power of 2; Flash Attention tiles to next power of 2 (256), wasting 64/256 = 25% of compute |
| Recommendation | Change head dimension to 128 or 256 (requires model architecture change) |
| Estimated Savings | Up to 737.8 ms |
| Confidence | High |

### P2: Kernel-Level Tuning

| Field | Value |
|-------|-------|
| Category | SDPA Forward (Flash Attention) |
| Operation | `aten::_scaled_dot_product_flash_attention` |
| Current Efficiency | 34.31% of peak BF16 MAF |
| Issue | `attn_fwd.kd` kernel shows 18.9% coefficient of variation across 10 invocations |
| Recommendation | Profile the kernel to identify tile size optimization opportunities beyond head dim padding |
| Estimated Savings | Included in P1 estimate |
| Confidence | Medium |

---

## System-Level Analysis

**Assessment:** GPU utilization is optimal. No system-level bottlenecks were detected. There is no idle time, no memory copy overhead, and no communication overhead. All optimization opportunities are at the compute kernel level.

---

## Compute Kernel Analysis

### SDPA Forward (Flash Attention) — 100% of GPU Time

| Metric | Value |
|--------|-------|
| Total Time | 1,123.2 ms |
| Operation | `aten::_scaled_dot_product_flash_attention` |
| Invocations | 10 |
| Kernel | `attn_fwd.kd` (stream 1) |
| Mean Kernel Time | 112.3 ms |
| Min / Max | 90.6 ms / 143.5 ms |
| TFLOPS Achieved | 242.93 |
| Peak BF16 TFLOPS | 708 |
| Efficiency | 34.31% |
| Bound Type | Compute |
| FLOPS/Byte | 32,768 |

#### Workload Profile

| Parameter | Value |
|-----------|-------|
| Batch Size (B) | 1 |
| Heads (H_Q = H_KV) | 8 |
| Sequence Length (N_Q = N_KV) | 65,536 |
| Head Dimension (d_h_qk = d_h_v) | 192 |
| Attention Pattern | MHA (Multi-Head Attention) |
| Causal | No |
| Dtype | BFloat16 |
| Approx FLOPs/call | ~26.4 TFLOPS |
| Approx Data/call | ~604 MB |

#### Bottleneck: Non-Power-of-2 Head Dimension

The head dimension **d=192** is not a power of 2. The nearest power-of-2 values are 128 and 256. Flash Attention kernels tile the head dimension to the next power of 2 (256), resulting in:

- **25% compute waste** on padding (64 out of 256 elements per tile are padding)
- **34.31% efficiency** vs the expected 50-70% for long-sequence attention
- The ~20-35 percentage point efficiency gap is consistent with tile padding overhead

#### Kernel Variability

The kernel shows notable time variability (CV = 18.9%):
- Mean: 112.3 ms
- Median: 108.9 ms
- Range: 90.6 ms to 143.5 ms

This may indicate memory subsystem contention or workload-dependent tile scheduling effects.

---

## Validation Summary

| Check | Status |
|-------|--------|
| Time Sanity (category sum vs computation time) | PASS |
| Efficiency Anomalies (>100%) | PASS |
| Coverage (all categories have findings) | PASS |
| Priority Consistency | INFO - Top by GPU time: sdpa_fwd |

---

## Impact Summary

| # | Recommendation | Type | Savings (ms) | Confidence |
|---|----------------|------|-------------|------------|
| 1 | Use power-of-2 head dimension | architecture | 737.8 | High |
| 2 | Kernel-level tuning | kernel_tuning | Included in P1 | Medium |
| | **Total Estimated Savings** | | **737.8** | |

---

## Appendix

### Analysis Metadata

| Field | Value |
|-------|-------|
| Analysis Date | 2026-03-09 |
| TraceLens Version | AgenticMode Standalone |
| Categories Analyzed | 2 (cpu_idle, sdpa_fwd) |
| System-Level Findings | 1 (cpu_idle: no issues) |
| Compute Kernel Findings | 1 (sdpa_fwd: head dim waste) |

### Output Files

```
analysis_output/
├── standalone_analysis.md          # This report
├── perf_report.xlsx                # Excel performance report
├── perf_report_csvs/               # CSV exports
│   ├── gpu_timeline.csv
│   ├── ops_summary.csv
│   ├── ops_summary_by_category.csv
│   ├── ops_unique_args.csv
│   ├── unified_perf_summary.csv
│   └── SDPA_fwd.csv
├── category_data/                  # Pre-computed analysis data
│   ├── category_manifest.json
│   ├── cpu_idle_metrics.json
│   ├── sdpa_fwd_metrics.json
│   ├── sdpa_fwd_ops.csv
│   ├── sdpa_fwd_tree_data.json
│   └── cpu_idle_ops.csv
├── system_findings/
│   └── cpu_idle_findings.md
├── category_findings/
│   └── sdpa_fwd_findings.md
└── metadata/
    ├── cpu_idle_metadata.json
    └── sdpa_fwd_metadata.json
```

---

*Report generated by TraceLens AgenticMode Standalone Analysis*

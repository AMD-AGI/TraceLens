# MoE Fused Operations Analysis

**Status:** SUCCESS  
**Platform:** MI300X  
**Peak HBM BW:** 5.3 TB/s  
**Resolved Peak MAF (matrix_bf16):** 708 TFLOPS  

---

## Summary

| Metric | Value |
|--------|-------|
| Operation count | 1 |
| Total GPU kernel time | 6.5 ms |
| % of compute | 100.0% |
| Average efficiency | N/A (no perf model) |

---

## Operations

| Operation | Type | Count | Time (ms) | % of Category | Efficiency | Peak (TFLOPS) |
|-----------|------|-------|-----------|---------------|------------|---------------|
| vllm::rocm_aiter_fused_moe | Fused | 1 | 6.5 | 100.0 | N/A | 708 |

**Classification:**
- **Fused:** `vllm::rocm_aiter_fused_moe` — End-to-end fused MoE kernel combining routing and expert compute (aiter_fused_moe_kernel_bf16).

---

## Bottleneck Assessment

**Time criteria:** > 100 ms OR > 5% of category time  
**Efficiency criteria:** < 70% of peak TFLOPS  

- **vllm::rocm_aiter_fused_moe:** 6.5 ms — Below 100 ms threshold; single operation in category.
- **Efficiency:** Not available — Perf model (`has_perf_model: False`) does not provide TFLOPS/s or TB/s for this fused MoE kernel. Cannot assess efficiency or flag anomalies.

**Result:** No bottlenecks identified. Trace is short (6.5 ms total compute); MoE is the sole compute category.

---

## Recommendations

### Algorithmic
- No actionable algorithmic recommendations — single fused MoE op with no routing/balance data.
- For longer traces with multiple MoE layers, consider validating token distribution across experts and routing balance.

### Kernel Optimization
- MoE kernels are specialized; limited kernel-level tuning opportunity.
- Efficiency could not be assessed; consider enabling perf modeling for fused MoE kernels if available in the profiling pipeline.

---

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| *(none)* | — | — | — |

*No kernel_tuning impact estimates — efficiency data unavailable for this fused MoE operation.*

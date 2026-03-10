# GEMM Analysis Summary

**Status:** SUCCESS

**Platform:** MI300X | Peak HBM BW: 5.3 TB/s | Peak FP16 MAF: 654 TFLOPS/s

**Time breakdown:** GPU kernel time: 0.135 ms | CPU duration: 0.207 ms | Sync bottleneck: No

## Operations Breakdown

| Operation | Count | Invocations | Time (ms) | % of Category | Efficiency | FLOPS/Byte | Type |
|-----------|-------|-------------|-----------|---------------|------------|------------|------|
| aten::mm | 1 | 10 | 0.135 | 100.0% | 23.51% | 16.0 | memory-bound |

## Bottleneck: aten::mm — (131072, 32) × (32, 32)

- **Time:** 0.135 ms (100% of GEMM compute)
- **Efficiency:** 23.51% of peak HBM bandwidth (1.25 TB/s achieved vs 5.3 TB/s peak)
- **Issue:** Memory-bound GEMM with tall-skinny shape (M=131072, N=32, K=32). Low FLOPS/Byte (16) limits reuse; achieved bandwidth is well below peak.
- **Kernel:** `Cijk_Ailk_Bljk_HHS_BH_MT64x128x32_MI16x16x16x1_SN_...` — tile size 64×128×32 appears suboptimal for this tall-skinny geometry where N=32 is smaller than the 128 tile dimension. Generate replay artifact for kernel team to investigate alternative tile configurations and memory access patterns.
- **Algorithmic:** Batch the 10 invocations together using `torch.bmm` or grouped GEMM to improve GPU parallelism and amortize memory overhead. Consider `torch.compile` to auto-batch if applicable.
- **Priority:** Critical (100% of compute time AND <30% efficiency)

## Additional Notes

- Missing perf models: 0
- Quantized GEMMs detected: 0
- Trace is very short (total GPU kernel time ~0.135 ms); absolute impact is small, but the efficiency gap is significant for this shape class and representative of how this kernel would perform at scale.

## Impact Summary
| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Kernel tuning for tall-skinny GEMM (M=131K, N=32, K=32) | kernel_tuning | 0.103 | medium |
| Batch 10 aten::mm invocations | algorithmic | ~0.02 | low |

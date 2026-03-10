# SDPA Forward Findings

## Attention Type Detected

**Flash Attention** — The trace uses the optimized Flash Attention implementation (`aten::_scaled_dot_product_flash_attention`). Paged Attention was not detected.

---

## Summary

| Metric | Value |
|--------|-------|
| Total SDPA Time | 1,123.2 ms |
| Operation Count | 10 |
| Average Efficiency | 34.31% of peak MAF |
| TFLOPS Achieved | 242.93 (vs 708 peak) |
| Bound Type | Compute |

---

## Operations Table

| Operation | Count | Time (ms) | % of Category | Efficiency | Input Dims |
|-----------|-------|-----------|---------------|------------|------------|
| aten::_scaled_dot_product_flash_attention | 10 | 1,123.2 | 100.0% | 34.31% | B=1, H=8, N=65536, D=192 |

**Workload Profile:** MHA (Multi-Head Attention), N_Q=65536, N_KV=65536, 8 heads

---

## Bottleneck Analysis

### Primary Bottleneck: Non-Power-of-2 Head Dimension (Tile Waste)

The head dimension **d=192** is not a power of 2. Flash Attention kernels typically tile the head dimension to the next power of 2 (256), resulting in:

- **~25% compute waste** — 64/256 of the tiled dimension is padding
- **34.31% efficiency** — below the 50–70% expected for N>4096 sequences
- **Compute-bound** — FLOPS/Byte = 32,768 (extremely compute-bound)

### Workload Characteristics

- **Sequence length:** 65,536 (very long context)
- **Shape:** B=1, H=8, N=65536, D=192
- **Approx FLOPs/call:** ~26.4 TFLOPS
- **Kernel:** `attn_fwd.kd` on stream 1
- **Kernel time variability:** Mean 112.3 ms, CV ~18.9% (notable variability)

### Efficiency Context

| Expected (N>4096) | Actual | Gap |
|-------------------|--------|-----|
| 50–70% | 34.31% | ~20–35 pp below expected |

The gap is consistent with tile padding overhead from the non-pow2 head dimension.

---

## Recommendations

### Algorithmic (Primary)

1. **Use power-of-2 head dimension** — Change d from 192 to 256 (or 128) to eliminate tile waste. This may require model architecture changes or fine-tuning.
2. **Consider head dimension 128** — If 256 is too large, 128 is the next smaller power of 2 and would reduce waste compared to 192.

### Kernel-Level

1. **Generate replay artifact** — Profile the `attn_fwd.kd` kernel to validate tile size behavior and identify any additional optimization opportunities.
2. **Kernel tuning** — The pre-computed impact estimate indicates significant savings potential from kernel-level optimization (see Impact Summary).

---

## Impact Summary

| Recommendation | Type | Estimated Savings (ms) | Confidence |
|---------------|------|----------------------|------------|
| Kernel tuning (attn_fwd.kd) | kernel_tuning | 737.8 | high |

---

## Platform Notes

- **GPU:** MI300X
- **Peak BF16 MAF:** 708 TFLOPS
- **Peak HBM BW:** 5.3 TB/s
- **Memory:** 192 GB

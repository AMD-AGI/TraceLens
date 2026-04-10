<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Kernel fusion findings

**Scope:** Module-level fusion candidates from `fusion_candidates.json` after deterministic analysis (`kernel_fusion_analysis.py`). **Platform context (metrics file):** MI355X, peak HBM 8.0 TB/s. **Trace-level fusion category time:** ~492 ms; **modeled mid-case savings (subset of candidates):** ~144.76 ms across five operations.

**Classification note:** Twenty-two candidates were scored for savings estimates; the on-disk `fusion_candidates.json` contains 35 rows (broader extraction). Below, **P1–P3** document actionable fusion angles; an appendix lists candidates **not pursued** as single fused units.

---

## P1 (high confidence)

### 1. Convolution + normalization + activation (`nn.Module: Conv2dNormActivation_0`)

**Insight:** Known **unfused convolution / normalization / activation** pattern: grouped convolution, three batch-norm forward passes, dtype/layout copies, and a bias-style add appear as separate kernels instead of one fused forward pass.

**Action:** Enable a **fused convolution–normalization–activation** path from the vendor math library (or a supported eager fusion) for this block shape; verify numerical parity on training.

**Impact:** Pre-computed (memory-bound / matrix-compute fusion model): **~45.74 ms** saved vs. unfused (mid), **~55.19 ms** current modeled kernel time, **~8.8%** of end-to-end window (low/high same in metrics).

**Confidence:** High — standard vision block pattern.

**Kernels (unique names, times aggregated per candidate slice):**

| Approx. time (µs) | Class (trace) | Kernel (abbrev.) |
|-------------------|---------------|-------------------|
| 38.92 | — | `MIOpenBatchNormFwdTrainSpatialNorm` |
| 36.20 | — | `MIOpenBatchNormFwdTrainSpatialMeanVariance` |
| 27.32 | — | `MIOpenBatchNormFwdTrainSpatialFinalMeanVariance` |
| 18.80 | Matrix | Grouped conv GEMM-style kernel (`ck`…`grouped_conv_fwd_multiple_abd`…) |
| 2.68 | Elementwise | `bfloat16_copy` vectorized kernel |
| 1.52 | Elementwise | `CUDAFunctorOnSelf_add` vectorized kernel |

**Projection (from `kernel_fusion_metrics.json`):**

| Field | Value |
|--------|--------|
| Estimated savings (mid) | 45.741 ms |
| Current modeled time | 55.191 ms |
| Instances | 440 |
| Kernels in scope | 6 |
| Bound | memory |
| E2E % (mid) | ~8.84% |

---

### 2. LayerNorm backward (`aten::native_layer_norm_backward`)

**Insight:** Known **unfused LayerNorm backward**: gradient w.r.t. input and gamma/beta reduction run as **two** separate kernels.

**Action:** Use a **fused LayerNorm backward** implementation (or compiler fusion) where available; align with forward LN fusion if the forward pass is also split.

**Impact:** No row in this run’s `impact_estimates`; qualitative gain is moderate per instance (39 instances) but worthwhile if LayerNorm is on the critical path.

**Confidence:** High — classic split backward pattern.

**Kernels:**

| Approx. time (µs) | Class (trace) | Kernel (abbrev.) |
|-------------------|---------------|-------------------|
| 79.96 | LayerNorm | `layer_norm_grad_input_kernel` |
| 10.48 | — | `cuComputeGradGammaBeta` |

**Projection:**

| Field | Value |
|--------|--------|
| Deterministic savings estimate | Not produced for this symbol in this run |
| Suggested follow-up | Re-run estimator after CSV/kernel mapping coverage for these kernel names |

---

### 3. BatchNorm2d module forward (`nn.Module: BatchNorm2d_0`)

**Insight:** Known **unfused batch normalization (forward)**: spatial mean/variance, final statistics, norm, plus a separate add — typical when a **library BN** path does not fuse all phases.

**Action:** Prefer **fused BN forward** APIs or fused conv–BN when legal for this model.

**Impact:** Pre-computed: **~38.77 ms** saved (mid), **~39.94 ms** current, **~7.5%** E2E (range ~7.46–7.52%).

**Confidence:** High — matches standard unfused BN forward structure.

**Kernels:**

| Approx. time (µs) | Class (trace) | Kernel (abbrev.) |
|-------------------|---------------|-------------------|
| 52.76 | — | `MIOpenBatchNormFwdTrainSpatialFinalMeanVariance` |
| 17.56 | — | `MIOpenBatchNormFwdTrainSpatialNorm` |
| 16.52 | — | `MIOpenBatchNormFwdTrainSpatialMeanVariance` |
| 2.32 | Elementwise | `CUDAFunctorOnSelf_add` vectorized kernel |

**Projection:**

| Field | Value |
|--------|--------|
| Estimated savings (mid) | 38.768 ms |
| Current modeled time | 39.942 ms |
| Instances | 448 |
| Kernels in scope | 4 |
| Bound | memory |
| E2E % | ~7.46–7.52% |

---

### 4. BatchNorm forward (ATen op) (`aten::miopen_batch_norm`)

**Insight:** Same **unfused BN forward** pattern as the `BatchNorm2d` module scope (three spatial BN kernels); annotation sits at the **ATen** boundary instead of `nn.Module`.

**Action:** Same as finding 3 — **fused BN** or **conv–BN fusion**; treat this as duplicate telemetry at a different stack frame.

**Impact:** Pre-computed: **~33.58 ms** saved (mid), **~38.90 ms** current, **~6.3–6.6%** E2E.

**Confidence:** High.

**Kernels:**

| Approx. time (µs) | Class (trace) | Kernel (abbrev.) |
|-------------------|---------------|-------------------|
| 52.76 | — | `MIOpenBatchNormFwdTrainSpatialFinalMeanVariance` |
| 17.56 | — | `MIOpenBatchNormFwdTrainSpatialNorm` |
| 16.52 | — | `MIOpenBatchNormFwdTrainSpatialMeanVariance` |

**Projection:**

| Field | Value |
|--------|--------|
| Estimated savings (mid) | 33.579 ms |
| Current modeled time | 38.903 ms |
| Instances | 448 |
| Kernels in scope | 3 |
| Bound | memory |
| E2E % | ~6.32–6.62% |

---

### 5. LayerNorm forward (`nn.Module: LayerNorm_0`)

**Insight:** Known **unfused LayerNorm forward**: explicit **dtype conversion** kernel plus a **vectorized LayerNorm** kernel instead of one fused cast+normalize path.

**Action:** Use a **fused LayerNorm** (including cast) or match input dtype to avoid the extra copy kernel.

**Impact:** Not in `impact_estimates`; single module instance but useful for latency-sensitive steps.

**Confidence:** High.

**Kernels:**

| Approx. time (µs) | Class (trace) | Kernel (abbrev.) |
|-------------------|---------------|-------------------|
| 31.76 | LayerNorm | `vectorized_layer_norm_kernel` |
| 16.44 | Elementwise | `bfloat16tofloat32_copy` vectorized kernel |

**Projection:**

| Field | Value |
|--------|--------|
| Deterministic savings estimate | Not produced in this run |

---

### 6. BatchNorm backward (`aten::miopen_batch_norm_backward`)

**Insight:** Known **unfused batch normalization backward**: dscale/dbias, dx, and a final combine phase as **three** kernels.

**Action:** Use a **fused BN backward** kernel from the vendor library or framework integration.

**Impact:** Pre-computed: **~10.72 ms** saved (mid), **~12.08 ms** current, **~2.0–2.1%** E2E.

**Confidence:** High.

**Kernels:**

| Approx. time (µs) | Class (trace) | Kernel (abbrev.) |
|-------------------|---------------|-------------------|
| 10.32 | — | `MIOpenBatchNormBwdSpatialDScaleDBias` |
| 9.60 | — | `MIOpenBatchNormBwdSpatialDX` |
| 7.04 | — | `MIOpenBatchNormBwdSpatialFinalDScaleDBias` |

**Projection:**

| Field | Value |
|--------|--------|
| Estimated savings (mid) | 10.718 ms |
| Current modeled time | 12.077 ms |
| Instances | 448 |
| Kernels in scope | 3 |
| Bound | memory |
| E2E % | ~2.03–2.11% |

---

## P2 (medium confidence)

### 1. Convolution backward (`aten::convolution_backward`)

**Insight:** **Novel / vendor-specific** backward graph: two large **inverse-GEMM / conv-bwd** kernels plus **copy/cast** and **subtensor** helpers — potential fusion is **framework- and version-dependent** (not a single documented “unfused LN” pattern).

**Action:** Profile with **vendor-backed conv backward** tuning; investigate whether epilogues can absorb adjacent elementwise/cast work without breaking autograd semantics.

**Impact:** No entry in `impact_estimates` for this symbol; summed candidate GPU time ~293 µs across listed launches (per-trace slice).

**Confidence:** Medium — real sequence, but fusion feasibility depends on backward API contracts.

**Kernels:**

| Approx. time (µs) | Class (trace) | Kernel (abbrev.) |
|-------------------|---------------|-------------------|
| 133.04 | Matrix | `igemm_bwd_gtcx35_nhwc_bf16` … |
| 94.60 | Matrix | `igemm_wrw_gtcx35_nhwc_bf16` … |
| 50.28 | Copy/cast | `direct_copy` elementwise path |
| 15.20 | — | `SubTensorOpWithScalar1d` |

**Projection:**

| Field | Value |
|--------|--------|
| Deterministic savings estimate | Not produced in this run |

---

### 2. Squeeze–excitation block (`nn.Module: SqueezeExcitation_0`)

**Insight:** **Novel compound pattern**: global reduce, **1×1**-style matrix multiply, **sigmoid**, **broadcast multiply**, and several **copy/add** kernels — a **channel attention** subgraph that is rarely one kernel in eager mode.

**Action:** Consider a **custom fused module** (or vendor primitive) that implements reduce + FC + activation + gating in fewer passes; validate against reference.

**Impact:** Pre-computed: **~15.95 ms** saved (mid), **~18.38 ms** current, **~3.0–3.1%** E2E.

**Confidence:** Medium — pattern is recognizable, fusion is custom.

**Kernels (abbrev.):** global `reduce_kernel` (~47.8 µs), `elementwise` mul (~47.4 µs), multiple `bfloat16_copy`, `CUDAFunctor_add`, small `grouped_conv` / `naive_conv` / `sigmoid` kernels — 8 unique names, 12 launch rows in candidate.

**Projection:**

| Field | Value |
|--------|--------|
| Estimated savings (mid) | 15.954 ms |
| Current modeled time | 18.380 ms |
| Instances | 136 |
| Kernels in scope | 12 |
| Bound | memory |
| E2E % | ~3.04–3.12% |

---

### 3. Fully connected (`nn.Module: Linear_0`) — GEMM + epilogue

**Insight:** **GEMM epilogue** pattern: one large **matrix multiply** kernel plus multiple **layout/dtype copy** kernels — epilogue fusion could remove memory round-trips.

**Action:** Enable **fused bias/epilogue** on the matrix multiply (where API allows) or align tensor layouts to avoid extra copies.

**Impact:** Not modeled in `impact_estimates`; candidate slice ~129 µs summed GPU time, **197** instances — cumulative effect may be meaningful.

**Confidence:** Medium — depends on layout and framework epilogue support.

**Kernels:**

| Approx. time (µs) | Class (trace) | Kernel (abbrev.) |
|-------------------|---------------|-------------------|
| 93.68 | Matrix | `Cijk_Alik_Bljk` … (GEMM) |
| 35.04 | Elementwise | `bfloat16_copy` (×3) |

**Projection:**

| Field | Value |
|--------|--------|
| Deterministic savings estimate | Not produced in this run |

---

### 4. Channel scaling helper (`torchvision/ops/misc.py(251): _scale`)

**Insight:** Same structural idea as squeeze–excitation **gating** (reduce → small conv/GEMM → nonlinear → broadcast): **medium** novelty, overlaps conceptually with finding P2-2.

**Action:** If hot, **inline into a fused custom op** or reuse the same remediation as the squeeze–excitation block.

**Impact:** Not in `impact_estimates`; ~88 µs summed in candidate slice.

**Confidence:** Medium.

**Kernels:** Closely mirrors `SqueezeExcitation_0` (shared reduce, copies, add, small GEMM, sigmoid, conv) — seven unique kernels in eleven launch rows.

**Projection:**

| Field | Value |
|--------|--------|
| Deterministic savings estimate | Not produced in this run |

---

## P3 (low confidence)

### 1. Plain convolution (`nn.Module: Conv2d_0`)

**Insight:** **Partial** fusion angle: **convolution** plus **duplicate-style copy** kernels — smaller per-launch footprint than full conv–BN–act blocks.

**Action:** Layout tuning and **fused conv** APIs if copies exist only for dtype/layout bridging.

**Impact:** Not modeled; ~69 µs summed, **729** instances.

**Confidence:** Low — gain per instance is small; bulk effect is from instance count.

**Kernels:** `naive_conv_ab_nonpacked_fwd_nhwc` (~51 µs), `bfloat16_copy` (~17.6 µs ×2).

**Projection:** Not in `impact_estimates`.

---

### 2. Convolution (ATen) with add epilogue (`aten::miopen_convolution`)

**Insight:** **GEMM epilogue**-like pair (matrix conv kernel + elementwise add) — already a tight sequence; further fusion is **incremental**.

**Action:** Low priority unless this op sits on a hot loop; consider **fused add** epilogue on the convolution API.

**Impact:** Not modeled; ~8 µs summed, **272** instances.

**Confidence:** Low.

**Kernels:** `ck` grouped conv forward (~4.5 µs), `CUDAFunctor_add` (~remainder).

**Projection:** Not in `impact_estimates`.

---

## Appendix: Candidates not pursued as single fusion targets

| Module / op (trace name) | Classification | Rationale |
|--------------------------|----------------|-----------|
| `torch/optim/adam.py`: `_multi_tensor_adam` | Not fusable (here) | Optimizer **foreach** over many independent tensors; launches are already batched **multi-tensor** kernels — not one logical op to fuse. |
| `nn.Module: Sequential_0` | Container | **Sequential** wrapper — not one kernel fusion unit. |
| `model/model.py(234): backbone_forward` | Container | Full **backbone** span — aggregate of many independent stages. |
| `nn.Module: AnyStage_0`, `ResBottleneckBlock_0`, `BottleneckTransform_0`, `SimpleStemIN_0` | Container / composite | Stage or **block** containers; prefer leaf modules (e.g. `Conv2dNormActivation`) for fusion. |
| `AddmmBackward0` / `autograd::…AddmmBackward0` | Not fusable | **All or nearly all matrix multiplies** in the slice — GEMM–GEMM fusion is generally **not** the same as epilogue fusion; reject per guidance. |
| `aten::_foreach_*` (add, mul, lerp, sqrt, div, addcdiv, addcmul) | Not fusable (here) | Already **vectorized foreach** kernels across parameter lists; not sequential subgraph fusion. |
| `autograd::…MeanBackward1`, `MulBackward0`, `ToCopyBackward0`, `MulBackward0` (module), `MSELoss_0`, `aten::mse_loss_backward` | Low value / independent | Small **elementwise** or **loss** chains; fusion possible but **low impact** vs. BN/conv blocks. |
| `model/model.py(152): forward` | Unclear / sparse | Candidate slice lacked resolved kernel metadata in extraction (`?` types) — treat as **non-actionable** from this file alone. |

---

## Summary

- **Dominant known patterns:** Unfused **convolution + batch norm + activation**, **batch norm forward/backward**, and **LayerNorm** forward/backward splits — these align with the **largest modeled savings** in `kernel_fusion_metrics.json` (~145 ms mid total across five estimated ops).
- **Novel / custom:** **Squeeze–excitation**-style blocks and **channel scaling** helpers are good **medium-priority** targets for **custom fused kernels** or library upgrades.
- **Deprioritize:** Optimizer **foreach** regions, **Sequential/backbone** spans, **GEMM-only** backward slices, and tiny **loss/grad** elementwise chains for **kernel fusion** specifically.

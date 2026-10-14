<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# Model op performance without a trace
```{meta}
:description: Learn how to call the TraceLens perf model directly as an analytical library - compute FLOPs, bytes, arithmetic intensity, and roofline-bounded time from op shapes alone, with no trace or kernel launch.
:keywords: TraceLens, perf model, roofline, FLOPS, arithmetic intensity, GEMM, SDPA, analytical model, sizing study, ceiling, no trace, ROCm, MI300X
```

In normal use, TraceLens fuses two sources per kernel: the *perf model* supplies
analytical FLOPs and bytes from an op's shapes and dtypes, and the *trace*
supplies the measured wall-clock time. Dividing one by the other gives the
achieved `TFLOPS/s` and `TB/s` in every report.

The perf model is a Python library — you can call the analytical half on its
own, with *no trace and no kernel launch*. Hand it shapes, get back FLOPs and
bytes; arithmetic intensity and roofline-bounded time follow. This is useful for
sizing studies, end-to-end ceilings, and regression bounds before you run
anything.

## Before you begin

Confirm you have the following before continuing.

- [TraceLens installed](../install/install.md).
- The shapes and dtypes of the ops you want to model (no trace required).

## Op coverage

The perf model ships curated formulas for the fundamental op classes in modern
workloads — `GEMM`, `SDPA`, `CONV`, `Softmax`, `Normalization`, `GroupedGemm`,
`MoEComm`, `MambaSSD`, `FusedRoPE`, `CrossEntropy`, `CausalConv1d`, plus the
elementwise and reduction primitives — over a dozen base classes, all maintained
and tested against real traces. Every base class in
`TraceLens.PerfModel.perf_model` exposes `flops_func` and `bytes_func` (and
`*_bwd_func` variants where a backward pass applies).

## Compute FLOPs, bytes, and arithmetic intensity

Import the op class and call its `flops_func` / `bytes_func` with the op's shapes.
For a GEMM of shape `M × N × K`:

```python
from TraceLens.PerfModel.perf_model import GEMM

M, N, K = 40960, 6144, 1536
bpe = 2  # bytes per element (bf16)

flops = GEMM.flops_func(M, N, K, bias=False)                 # 2 * M * N * K
bytes_ = GEMM.bytes_func(M, N, K, False, bpe, bpe, bpe, bpe)  # read A + read B + write C
arithmetic_intensity = flops / bytes_                         # FLOPs per byte
```

Attention works the same way, with forward and backward variants:

```python
from TraceLens.PerfModel.perf_model import SDPA

# B, S_q, H_Q, S_kv, H_KV, d_h_qk, d_h_v, causal
flops_fwd = SDPA.flops_func(1, 4096, 64, 4096, 8, 128, 128, True)
bytes_fwd = SDPA.bytes_func(1, 4096, 64, 4096, 8, 128, 128, True, bpe)
flops_bwd = SDPA.flops_bwd_func(1, 4096, 64, 4096, 8, 128, 128, True, flash_impl=True)
```

## Bound the time with a roofline

Arithmetic intensity tells you which side of the roofline an op sits on. Compare
it against the hardware's roofline knee (peak compute divided by peak bandwidth),
then take the larger of the compute-bound and bandwidth-bound times as the
achievable lower bound:

```python
PEAK_TFLOPS_BF16 = 708.0   # MI300X matrix bf16
HBM_BW_GBPS = 5300.0       # HBM3

ai_knee = (PEAK_TFLOPS_BF16 * 1e12) / (HBM_BW_GBPS * 1e9)   # FLOPs/byte crossover

def roofline_time_us(flops, bytes_):
    compute_us = flops / (PEAK_TFLOPS_BF16 * 1e12) * 1e6
    bandwidth_us = bytes_ / (HBM_BW_GBPS * 1e9) * 1e6
    return max(compute_us, bandwidth_us)
```

This is the same analytical model TraceLens's built-in roofline analysis uses —
here applied to shapes directly instead of to a trace. The roofline is an upper
bound on achievable performance (a real kernel won't reach it), but the *ratios*
between layers and regimes are solid planning numbers that fall out of pure
arithmetic.

## Worked example

The
[`perf_model_without_trace.ipynb`](https://github.com/AMD-AGI/TraceLens/blob/main/examples/perf_model_without_trace.ipynb)
notebook works a full example end to end: every linear in a Llama-3-70B decoder
layer plotted as arithmetic intensity, SDPA forward and backward across sequence
length, and a per-token roofline rollup (predicted ms per token for prefill and
decode, bf16, AMD Instinct™ MI300X) — all without a single trace.

## Related topics

- [Analyze traces with the TraceLens SDK](./sdk-analysis.md)
- [GEMM analysis in TraceLens](../conceptual/gemm-analysis.md)
- [Performance report columns](../reference/perf-report-columns.md)
- [API reference](../reference/api-reference.md)

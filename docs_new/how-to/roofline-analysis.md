<!--
Copyright (c) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
-->

# Add roofline analysis

This guide shows how to add roofline metrics to a report so you can tell whether
each operation is compute-bound or memory-bound on a target accelerator.

## Prerequisites

- TraceLens installed (see [Installation instructions](../install/installation.md)).
- A PyTorch profiler trace.
- A GPU architecture specification with peak FLOPS and peak bandwidth for your
  accelerator (for example, `mi300.json`). Bundled specifications live under
  `TraceLens/Agent/Analysis/utils/arch/`.

## Step 1: Generate the report with an architecture spec

The PyTorch report includes roofline columns by default; to use simulated
GEMM/SDPA timings from Origami, pass `--enable-origami` together with a GPU
architecture JSON:

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path traces/trace.json \
    --enable-origami
```

**Expected output:** the report's category sheets gain roofline columns. Each
operation gets a percentage indicating how close it runs to the theoretical
roofline and a classification of **compute-bound** or **memory-bound**.

## Step 2: Understand the roofline knee point

TraceLens classifies an operation by comparing its arithmetic intensity
(FLOPs/byte) against the accelerator's roofline knee point:

```text
Arithmetic Intensity Threshold = Peak FLOPS / Peak Bandwidth
```

For the MI300X (FP16), this is approximately 1300 TFLOPS / 5.3 TB/s ≈ **245
FLOPs/byte**. Operations *below* the threshold are **memory-bound**; operations
*above* it are **compute-bound**.

## Step 3: Interpret the roofline columns

Consider two convolution operations from a ResNet forward pass:

| Metric | Compute-bound conv | Memory-bound conv |
|--------|--------------------|-------------------|
| Input shape | `(256, 256, 2, 2)` | `(256, 64, 8, 8)` |
| Filter shape | `(256, 256, 3, 3)` | `(64, 64, 1, 1)` |
| GFLOPS | 1.21 | 0.13 |
| Data moved (MB) | 2.13 | 4.01 |
| FLOPs/byte | 542.1 | 31.9 |
| Roofline bound | COMPUTE_BOUND | MEMORY_BOUND |
| Kernel time (µs) | 33.0 | 9.2 |
| TFLOP/s | 36.6 | 14.6 |
| TB/s | 0.07 | 0.46 |
| Pct roofline | 5.6% | 8.6% |

- The 3×3 convolution performs about 9× more multiply-accumulate operations per
  input element on a small spatial size, driving arithmetic intensity to 542.1
  FLOPs/byte — above the 245 threshold — so it is compute-bound.
- The 1×1 convolution does little compute per element on a large input, giving
  31.9 FLOPs/byte — below the threshold — so it is memory-bound, and the focus
  for optimization should be memory access patterns.

## Step 4: Visualize (optional)

See the `examples/roofline_plots_example.ipynb` notebook to build roofline-style
plots for specific operators through the SDK.

## Next steps

- Compare roofline efficiency across hardware or software versions by
  [comparing two traces](./compare-traces.md).

<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->
# Architecture Performance Specs

These performance metrics are derived from the following sources:

- [Understanding Peak and Max-Achievable FLOPS](https://rocm.blogs.amd.com/software-tools-optimization/Understanding_Peak_and_Max-Achievable_FLOPS/README.html)
- [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD Instinct MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
- [AMD Instinct MI355X](https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html)

**MI355X.json caveat:** memory/vector values are AMD-published specs (288 GB HBM3E,
8 TB/s, 157.3 TFLOPS vector FP16/FP32, 78.6 TFLOPS FP64). The `matrix_*` MAF
entries are ESTIMATES derived by applying MI325X's measured MAF:peak ratios to
MI355X published peak matrix rates — replace them by running
`python -m TraceLens.PerfModel.benchmarking.microbench --device 0
--output TraceLens/Agent/Analysis/utils/arch/MI355X.json` on real MI355X
hardware (see PerfModel/benchmarking/README.md).

They represent peak performance bounds used to estimate the optimization headroom available given measured kernel performance. In practice, real kernels may not reach these bounds.

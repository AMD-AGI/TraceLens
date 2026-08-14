<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->
# Architecture Performance Specs

These performance metrics are derived from the following sources:

- [Understanding Peak and Max-Achievable FLOPS](https://rocm.blogs.amd.com/software-tools-optimization/Understanding_Peak_and_Max-Achievable_FLOPS/README.html)
- [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD Instinct MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
- Radeon 8060S (Strix Halo APU, gfx1151): benchmark-derived — max-achievable numbers measured with the
  microbench harness (`python -m TraceLens.PerfModel.benchmarking.microbench --device 0 --output ...`),
  merged max of 3 runs on the target hardware. Replace proxy arches (e.g. an Instinct part) with the
  measured file when profiling this device — proxies can be 20-25x off the real roofline.

They represent peak performance bounds used to estimate the optimization headroom available given measured kernel performance. In practice, real kernels may not reach these bounds.

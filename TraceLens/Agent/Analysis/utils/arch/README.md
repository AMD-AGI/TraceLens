<!--
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->
# Architecture Performance Specs

These performance metrics are derived from the following sources:

- [Understanding Peak and Max-Achievable FLOPS](https://rocm.blogs.amd.com/software-tools-optimization/Understanding_Peak_and_Max-Achievable_FLOPS/README.html)
- [AMD Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
- [AMD Instinct MI325X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html)
- [AMD Radeon PRO W7900](https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900.html)
- [AMD Radeon PRO W7900 Dual Slot](https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900-dual-slot.html)

They represent peak performance bounds used to estimate the optimization headroom available given measured kernel performance. In practice, real kernels may not reach these bounds.

## W7900 (`W7900.json`, gfx1100)

The platform `W7900` covers the `AMD Radeon Pro W7900` (PCI `1002:7448`) and
the `AMD Radeon Pro W7900 Dual Slot` (`1002:744A`); the microbench names both
`W7900`. Other Navi 31 (gfx1100) boards, such as the W7800, the W7800 48GB and
the Radeon RX 7900 XTX/XT/GRE, have no bundled spec.

The values are AMD's published peaks for the two boards, from the
[AMD Radeon PRO W7900](https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900.html)
and
[AMD Radeon PRO W7900 Dual Slot](https://www.amd.com/en/products/graphics/workstations/radeon-pro/w7900-dual-slot.html)
product pages. Only figures both pages agree on are used:

| Key | Value | Product page field |
|-----|-------|--------------------|
| `mem_bw_gbps` | 864 | Peak Memory Bandwidth (GB/s) |
| `memory_gb` | 48 | Dedicated Memory Size |
| `matrix_fp16` | 123 | Peak Half Precision (FP16 Matrix) Performance (TFLOPs) |
| `matrix_int8` | 123 | Peak 8-bit Precision (INT8 Matrix) Performance (TOPs) |
| `vector_fp32` | 61.3 | Peak Single Precision (FP32 Vector) Performance (TFLOPs) |

These keys have no ceiling, so rows that need them get no roofline:

- `matrix_bf16`, `vector_bf16`, `vector_fp64`: AMD lists no figure for them.
  BF16 GEMMs therefore get no roofline with this spec.
- `vector_fp16`: the two product pages disagree (61.3 TFLOPs for the W7900,
  123 TFLOPs for the W7900 Dual Slot).
- `matrix_fp8`, `matrix_fp4`, `matrix_fp6`, `matrix_fp32`, `matrix_fp64`: RDNA3
  WMMA has no such path.

The published INT4 matrix figure (245 TOPs) has no key in the schema.

For a ceiling on the keys left out, measure the board with
`TraceLens/PerfModel/benchmarking/microbench.py` and pass its JSON with
`--gpu_arch_json_path`.

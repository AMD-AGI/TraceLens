<!--
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

See LICENSE for license information.
-->

# GPU Architecture Specifications

GPU architecture JSON files define specifications used for roofline analysis in TraceLens. You can create your own based on the format below.

## File Format

```json
{
    "name": "MI300X",
    "mem_bw_gbps": 5300,
    "max_achievable_tflops": {
        "matrix_fp16": 654,
        "matrix_bf16": 708,
        "matrix_fp32": 163,
        "matrix_fp64": 81,
        "matrix_fp8": 1273,
        "matrix_int8": 2600,
        "vector_fp16": 163,
        "vector_bf16": 163,
        "vector_fp32": 81,
        "vector_fp64": 40
    },
    "_reference": "https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html#amd-maf-results"
}
```

### Fields

| Field | Description |
|-------|-------------|
| `name` | GPU model name |
| `mem_bw_gbps` | Memory bandwidth in GB/s |
| `max_achievable_tflops` | Max achievable TFLOPS by compute type and precision |
| `_reference` | (Optional) Source reference for the values |

### Compute Spec Keys

The `max_achievable_tflops` keys combine compute type and precision:

- **Compute Type**:
  - `matrix_*`: Matrix compute units (tensor cores, matrix cores) - used by GEMM, CONV, SDPA
  - `vector_*`: Vector compute units (SIMD) - used by elementwise ops

- **Precision**:
  - `fp8`: 8-bit floating point
  - `fp16`: Half precision (16-bit)
  - `bf16`: BFloat16
  - `fp32`: Single precision (32-bit)
  - `fp64`: Double precision (64-bit)
  - `int8`: 8-bit integer

## Max Achievable FLOPS (MAF)

The values represent **Max Achievable FLOPS (MAF)**, not theoretical peak FLOPS. MAF is a more realistic performance target that accounts for real-world constraints.

For methodology and official AMD MAF measurements, see:
- [Measuring Max-Achievable FLOPS - Part 2](https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html#amd-maf-results)
- [Understanding Peak and Max-Achievable FLOPS](https://rocm.blogs.amd.com/software-tools-optimization/Understanding_Peak_and_Max-Achievable_FLOPS/README.html)

## Usage

Create a GPU arch JSON file (e.g., `mi300x.json`) and pass it when generating performance reports:

```bash
TraceLens_generate_perf_report_pytorch \
    --profile_json_path trace.json \
    --output_xlsx_path report.xlsx \
    --gpu_arch_json_path /path/to/mi300x.json
```

The **Compute Spec** column in the report shows the combined type (e.g., `matrix_bf16`) and uses it to look up the appropriate MAF value for roofline calculations.

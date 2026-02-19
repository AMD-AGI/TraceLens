"""Platform specifications and category mappings for TraceLens AgenticMode analysis.

Platform specs use the TraceLens arch dict format with max_achievable_tflops
(MAF) keyed by compute spec (e.g. matrix_bf16, matrix_fp8).
See TraceLens/examples/gpu_arch_example.md for format reference.

MI300X MAF values from measured benchmarks:
  https://rocm.blogs.amd.com/software-tools-optimization/measuring-max-achievable-flops-part2/README.html
MI325X shares gfx942 arch with MI300X (same CU count), higher HBM BW.
MI355X/MI400 values are theoretical peak -- TODO: replace with measured MAF.
"""

PLATFORM_SPECS = {
    "MI300X": {
        "name": "MI300X",
        "mem_bw_gbps": 5300,
        "memory_gb": 192,
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
            "vector_fp64": 40,
        },
    },
    "MI325X": {
        "name": "MI325X",
        "mem_bw_gbps": 6000,
        "memory_gb": 256,
        "max_achievable_tflops": {
            "matrix_fp16": 794,
            "matrix_bf16": 843,
            "matrix_fp32": 194,
            "matrix_fp64": 97,
            "matrix_fp8": 1519,
            "matrix_int8": 3094,
            "vector_fp16": 194,
            "vector_bf16": 194,
            "vector_fp32": 97,
            "vector_fp64": 48
        },
    },
    "MI355X": {
        "name": "MI355X",
        "mem_bw_gbps": 8000,
        "memory_gb": 288,
        "max_achievable_tflops": {
            "matrix_fp16": 1686,
            "matrix_bf16": 1686,
            "matrix_fp32": 137,
            "matrix_fp64": 68,
            "matrix_fp8": 3567,
            "matrix_fp6": 4574,
            "matrix_fp4": 5663,
            "matrix_int8": 7134,
            "vector_fp16": 274,
            "vector_bf16": 274,
            "vector_fp32": 137,
            "vector_fp64": 68
        },
    },
    "MI400": {
        "name": "MI400",
        "mem_bw_gbps": 19600,
        "memory_gb": 432,
        "max_achievable_tflops": {
            # CDNA Next -- TODO: replace with measured MAF
            "matrix_fp16": 2500,
            "matrix_bf16": 2500,
            "matrix_fp32": 1250,
            "matrix_fp64": 625,
            "matrix_fp8": 20000,
            "matrix_fp4": 40000,
            "matrix_int8": 20000,
            "vector_fp16": 625,
            "vector_bf16": 625,
            "vector_fp32": 312,
            "vector_fp64": 156,
        },
    },
}

CATEGORY_SKILL_MAP = {
    "cpu_idle": "cpu-idle-analysis",
    "gemm": "gemm-analysis",
    "moe_fused": "moe-analysis",
    "sdpa_fwd": "sdpa-analysis",
    "elementwise": "elementwise-analysis",
    "reduce": "reduce-analysis",
    "triton": "triton-analysis",
    "batchnorm": "batchnorm-analysis",
    "convolution": "convolution-analysis",
    "other": "generic-op-analysis"
}

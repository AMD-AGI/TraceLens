"""Platform specifications and category mappings for TraceLens Jarvis analysis."""

PLATFORM_SPECS = {
    "MI300X": {
        "peak_hbm_bw_tbs": 5.3,
        "peak_bf16_maf_tflops": 708,
        "memory_gb": 192
    },
    "MI325X": {
        "peak_hbm_bw_tbs": 6.0,
        "peak_bf16_maf_tflops": 708,
        "memory_gb": 256
    },
    "MI355X": {
        "peak_hbm_bw_tbs": 6.5,
        "peak_bf16_maf_tflops": 850,
        "memory_gb": 288
    },
    "MI400": {
        "peak_hbm_bw_tbs": 7.0,
        "peak_bf16_maf_tflops": 1000,
        "memory_gb": 320
    },
}

CATEGORY_SKILL_MAP = {
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

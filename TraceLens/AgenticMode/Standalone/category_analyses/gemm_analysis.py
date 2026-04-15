#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""GEMM Analysis - Matrix Multiplications

Computes metrics for GEMM operations and outputs JSON for subagent processing.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    get_peak_specs,
    run_category_analysis,
)


def detect_quantized_gemm(op_name: str) -> bool:
    """Check if GEMM operation is quantized."""
    quantized_markers = ["w8a8", "int8", "fp8", "w4a16", "w4a4", "fp4", "mxfp4"]
    return any(marker in op_name.lower() for marker in quantized_markers)


def classify_gemm_operation(op_name: str, row) -> dict:
    """Classify GEMM operation type."""
    is_quantized = detect_quantized_gemm(op_name)
    return {
        "is_quantized": is_quantized,
        "gemm_type": "quantized" if is_quantized else "regular",
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract GEMM-specific aggregate metrics."""
    quantized_count = sum(
        1 for name in ops_df["name"] if detect_quantized_gemm(str(name))
    )

    missing_perf_model = 0
    if "TFLOPS/s_mean" in ops_df.columns:
        import pandas as pd

        missing_perf_model = ops_df["TFLOPS/s_mean"].isna().sum()

    return {
        "quantized_count": int(quantized_count),
        "missing_perf_model_count": int(missing_perf_model),
        **get_peak_specs(metadata),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze GEMM operations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    run_category_analysis(
        category="gemm",
        output_dir=args.output_dir,
        config={
            "extra_fields": ["Input Dims", "has_perf_model"],
            "operation_classifier": classify_gemm_operation,
        },
        extract_fn=extract_category_specific,
    )


if __name__ == "__main__":
    main()

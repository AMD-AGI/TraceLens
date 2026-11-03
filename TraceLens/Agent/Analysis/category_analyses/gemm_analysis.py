#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

# Constants
QUANTIZED_COMPUTE_SPECS = frozenset({"matrix_fp8", "matrix_fp4", "matrix_int8"})


def detect_quantized_gemm(compute_spec: str) -> bool:
    """Check if a GEMM's Compute Spec is a quantized precision."""
    return compute_spec in QUANTIZED_COMPUTE_SPECS


def classify_gemm_operation(op_name: str, row) -> dict:
    """Classify GEMM operation type."""
    is_quantized = detect_quantized_gemm(row["Compute Spec"])
    return {
        "is_quantized": is_quantized,
        "gemm_type": "quantized" if is_quantized else "regular",
    }


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract GEMM-specific aggregate metrics."""
    quantized_count = ops_df["Compute Spec"].isin(QUANTIZED_COMPUTE_SPECS).sum()

    missing_perf_model = 0
    if "TFLOPS/s_mean" in ops_df.columns:

        missing_perf_model = ops_df["TFLOPS/s_mean"].isna().sum()

    return {
        "quantized_count": int(quantized_count),
        "missing_perf_model_count": int(missing_perf_model),
        **get_peak_specs(metadata),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze GEMM operations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--category",
        required=True,
        help="Manifest category name; selects <category>_ops.csv",
    )
    parser.add_argument(
        "--comparison_scope",
        choices=("standalone", "comparative"),
        default="standalone",
        help=(
            "standalone: roofline Pct Roofline in operations[].efficiency; "
            "comparative: 100*t2/t1 in operations[].efficiency (needs TraceDiff CSV columns)"
        ),
    )
    args = parser.parse_args()

    run_category_analysis(
        category=args.category,
        output_dir=args.output_dir,
        config={
            "extra_fields": ["Input Dims", "Input type", "has_perf_model"],
            "operation_classifier": classify_gemm_operation,
        },
        extract_fn=extract_category_specific,
        comparison_scope=args.comparison_scope,
    )


if __name__ == "__main__":
    main()

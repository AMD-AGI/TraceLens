#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Convolution Operations Analysis

Computes metrics for Convolution operations and outputs JSON for subagent processing.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    get_peak_specs,
    run_category_analysis,
)


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract Convolution-specific aggregate metrics."""
    transpose_ops = ops_df[
        ops_df["name"].str.contains("transpose", case=False, na=False)
    ]
    transpose_time_ms = (
        transpose_ops["Kernel Time (µs)_sum"].sum() / 1000
        if len(transpose_ops) > 0
        else 0
    )

    total_time_ms = 0
    if "Kernel Time (µs)_sum" in ops_df.columns:
        total_time_ms = ops_df["Kernel Time (µs)_sum"].sum() / 1000

    transpose_overhead_pct = (
        (transpose_time_ms / total_time_ms * 100) if total_time_ms > 0 else 0
    )

    return {
        "transpose_count": len(transpose_ops),
        "transpose_time_ms": round(transpose_time_ms, 3),
        "transpose_overhead_percent": round(transpose_overhead_pct, 2),
        **get_peak_specs(metadata),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Convolution operations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    run_category_analysis(
        category="convolution",
        output_dir=args.output_dir,
        config={"extra_fields": []},
        extract_fn=extract_category_specific,
    )


if __name__ == "__main__":
    main()

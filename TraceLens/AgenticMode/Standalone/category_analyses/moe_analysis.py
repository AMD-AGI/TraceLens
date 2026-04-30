#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""MoE (Mixture of Experts) Analysis

Computes metrics for MoE fused operations and outputs JSON for subagent processing.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    get_peak_specs,
    run_category_analysis,
)


def _check_moe_data(output_dir, category):
    """Return NO_DATA metrics if MoE CSV is absent (expected for non-MoE traces)."""
    moe_csv = f"{output_dir}/category_data/{category}_ops.csv"
    if not os.path.exists(moe_csv):
        return {
            "category": category,
            "status": "NO_DATA",
            "message": f"No MoE operations detected in this trace ({category} bucket)",
            "total_time_ms": 0,
            "percent_of_compute": 0,
            "operation_count": 0,
            "operations": [],
            "category_specific": {},
        }
    return None


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract MoE-specific aggregate metrics."""
    return get_peak_specs(metadata)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MoE fused or unfused operations"
    )
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--category",
        default="moe_fused",
        choices=("moe_fused", "moe_unfused"),
        help="MoE bucket to analyze (default: moe_fused).",
    )
    args = parser.parse_args()

    run_category_analysis(
        category=args.category,
        output_dir=args.output_dir,
        config={"extra_fields": ["Input Dims", "Input type"]},
        extract_fn=extract_category_specific,
        no_data_check_fn=_check_moe_data,
    )


if __name__ == "__main__":
    main()

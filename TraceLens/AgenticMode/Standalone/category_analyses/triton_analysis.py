#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Triton Kernel Analysis

Computes metrics for Triton kernels and outputs JSON for subagent processing.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import (
    get_peak_specs,
    run_category_analysis,
)


def classify_triton_operation(op_name: str, row) -> dict:
    """Classify Triton kernel type from name prefix."""
    op_lower = op_name.lower()
    if op_lower.startswith("triton_poi_"):
        kernel_type = "pointwise"
    elif op_lower.startswith("triton_red_"):
        kernel_type = "reduction"
    elif op_lower.startswith("triton_per_"):
        kernel_type = "persistent"
    else:
        kernel_type = "other"
    return {"kernel_type": kernel_type}


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract Triton-specific aggregate metrics."""
    type_counts = {"pointwise": 0, "reduction": 0, "persistent": 0, "other": 0}
    for name in ops_df["name"]:
        classified = classify_triton_operation(str(name), None)
        type_counts[classified["kernel_type"]] += 1
    return {
        **{f"{k}_count": v for k, v in type_counts.items()},
        **get_peak_specs(metadata),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Triton kernels")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    parser.add_argument(
        "--comparison_scope",
        choices=("standalone", "comparative"),
        default="standalone",
        help=(
            "standalone: roofline efficiency in operations[].efficiency; "
            "comparative: 100*t2/t1 (needs TraceDiff CSV columns)"
        ),
    )
    args = parser.parse_args()

    run_category_analysis(
        category="triton",
        output_dir=args.output_dir,
        config={
            "extra_fields": [],
            "operation_classifier": classify_triton_operation,
        },
        extract_fn=extract_category_specific,
        compute_impact=False,
        comparison_scope=args.comparison_scope,
    )


if __name__ == "__main__":
    main()

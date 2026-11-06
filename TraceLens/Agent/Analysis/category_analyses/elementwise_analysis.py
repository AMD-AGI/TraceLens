#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Elementwise Operations Analysis

Computes metrics for elementwise operations and outputs JSON for subagent processing.
"""

import argparse

from TraceLens.Agent.Analysis.category_analyses.analysis_utils import (
    PEAK_MEM_BW_KEY,
    run_category_analysis,
)


def extract_category_specific(_ops_df, metadata) -> dict:
    """Extract elementwise-specific aggregate metrics."""
    return {PEAK_MEM_BW_KEY: metadata[PEAK_MEM_BW_KEY]}


def main():
    parser = argparse.ArgumentParser(description="Analyze elementwise operations")
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
            "standalone: roofline efficiency in operations[].efficiency; "
            "comparative: 100*t2/t1 (needs TraceDiff CSV columns)"
        ),
    )
    args = parser.parse_args()

    run_category_analysis(
        category=args.category,
        output_dir=args.output_dir,
        config={"extra_fields": ["Input Dims", "Input type", "FLOPS/Byte"]},
        extract_fn=extract_category_specific,
        comparison_scope=args.comparison_scope,
    )


if __name__ == "__main__":
    main()

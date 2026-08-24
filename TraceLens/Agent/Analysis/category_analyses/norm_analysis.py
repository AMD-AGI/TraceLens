#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Normalization Operations Analysis

Computes metrics for normalization operations (BatchNorm, LayerNorm, GroupNorm, etc.)
and outputs JSON for subagent processing.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis_utils import run_category_analysis


def extract_category_specific(ops_df, metadata) -> dict:
    """Extract normalization-specific aggregate metrics."""
    return {"peak_hbm_bw_tbs": metadata.get("peak_hbm_bw_tbs")}


def main():
    parser = argparse.ArgumentParser(description="Analyze normalization operations")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--category",
        default="norm",
        help="Category name prefix used for input/output files (default: norm)",
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
        config={"extra_fields": ["Input Dims", "Input type"]},
        extract_fn=extract_category_specific,
        comparison_scope=args.comparison_scope,
    )


if __name__ == "__main__":
    main()
